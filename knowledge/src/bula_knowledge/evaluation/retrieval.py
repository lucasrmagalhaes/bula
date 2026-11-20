"""Avaliação da busca contra um conjunto ouro, sem LLM.

Cada caso tem uma pergunta e os documentos em que a resposta está. Mede se o
documento certo aparece entre os k primeiros (hit rate), em que posição (MRR) e
quanto a busca demora. Roda igual para os três modos, o que mostra em números o
que a busca híbrida ganha sobre cada perna sozinha.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bula_knowledge.rag.workflow import RagWorkflow
from bula_knowledge.schemas import Chunk, SearchMode
from bula_knowledge.search import Searcher


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    lang: str
    # Cada item é um trecho do caminho do documento esperado, como "Reference/Hooks.md".
    expect: tuple[str, ...]
    kind: str = "paraphrase"


@dataclass
class CaseResult:
    id: str
    question: str
    lang: str
    kind: str
    rank: int | None
    took_ms: float
    top: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    mode: SearchMode
    k: int
    metrics: dict[str, float]
    cases: list[CaseResult]


def load_golden(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            raw = json.loads(line)
            cases.append(
                GoldenCase(
                    id=raw["id"],
                    question=raw["question"],
                    lang=raw.get("lang", "en"),
                    expect=tuple(raw["expect"]),
                    kind=raw.get("kind", "paraphrase"),
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path.name}, linha {number}: caso inválido ({exc})") from exc
    return cases


def first_hit_rank(chunks: list[Chunk], expect: tuple[str, ...]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(fragment in chunk.url for fragment in expect):
            return rank
    return None


def evaluate(
    searcher: Searcher, cases: list[GoldenCase], *, k: int, mode: SearchMode
) -> EvalResult:
    results: list[CaseResult] = []
    for case in cases:
        chunks, took_ms = searcher.search(case.question, k=k, mode=mode)
        results.append(
            CaseResult(
                id=case.id,
                question=case.question,
                lang=case.lang,
                kind=case.kind,
                rank=first_hit_rank(chunks, case.expect),
                took_ms=took_ms,
                top=[chunk.url for chunk in chunks[:3]],
            )
        )
    return EvalResult(mode=mode, k=k, metrics=_metrics(results, k), cases=results)


def _metrics(results: list[CaseResult], k: int) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {}
    latencies = sorted(result.took_ms for result in results)
    metrics = {
        f"hit_rate_at_{k}": sum(r.rank is not None for r in results) / total,
        f"mrr_at_{k}": sum(1 / r.rank for r in results if r.rank) / total,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": latencies[min(total - 1, int(total * 0.95))],
        "cases": float(total),
    }
    # Recorte por idioma e por tipo de pergunta: é onde os modos se separam.
    for attribute in ("lang", "kind"):
        for value in sorted({getattr(r, attribute) for r in results}):
            group = [r for r in results if getattr(r, attribute) == value]
            metrics[f"hit_rate_at_{k}_{attribute}_{value}"] = sum(
                r.rank is not None for r in group
            ) / len(group)
    return {name: round(value, 4) for name, value in metrics.items()}


@dataclass
class WorkflowCaseResult:
    id: str
    question: str
    lang: str
    kind: str
    # Posição do primeiro trecho de um documento esperado em cada etapa (None = fora dos k).
    rank_raw: int | None  # busca híbrida com a pergunta como veio
    rank_rewritten: int | None  # depois da reescrita, antes da triagem
    rank_graded: int | None  # no que a triagem aprovou
    kept: int
    attempts: int
    queries: list[str]
    degraded: bool
    took_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int


@dataclass
class WorkflowEvalResult:
    k: int
    metrics: dict[str, float]
    cases: list[WorkflowCaseResult]


class EvalAbortedError(RuntimeError):
    """As chamadas ao modelo estão falhando: não adianta gastar o conjunto inteiro."""


def evaluate_workflow(
    workflow: RagWorkflow,
    searcher: Searcher,
    cases: list[GoldenCase],
    *,
    k: int,
    on_case: Callable[[int, WorkflowCaseResult], None] | None = None,
) -> WorkflowEvalResult:
    """Mede o workflow de RAG com LLM, etapa por etapa, contra a busca sem LLM.

    Para cada pergunta compara três listas: a busca híbrida com a pergunta crua, os
    candidatos depois da reescrita e o que sobrou depois da triagem. Assim dá para ver
    separado o que a reescrita ganha e o que a triagem perde.
    """
    results: list[WorkflowCaseResult] = []
    for number, case in enumerate(cases, start=1):
        raw_chunks, _ = searcher.search(case.question, k=k)
        started = time.perf_counter()
        state = workflow.invoke(case.question)
        took_ms = round((time.perf_counter() - started) * 1000, 1)
        usage = state.get("usage", {})
        relevant = state.get("relevant", [])
        result = WorkflowCaseResult(
            id=case.id,
            question=case.question,
            lang=case.lang,
            kind=case.kind,
            rank_raw=first_hit_rank(raw_chunks, case.expect),
            rank_rewritten=first_hit_rank(state.get("candidates", [])[:k], case.expect),
            rank_graded=first_hit_rank(relevant[:k], case.expect),
            kept=len(relevant),
            attempts=state.get("attempts", 0),
            queries=state.get("tried", []),
            degraded=state.get("degraded", False),
            took_ms=took_ms,
            llm_calls=int(usage.get("llm_calls", 0)),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
        results.append(result)
        if on_case is not None:
            on_case(number, result)
        if number >= 3 and all(item.degraded for item in results):
            raise EvalAbortedError(
                "as três primeiras perguntas caíram no modo degradado: o modelo não está "
                "respondendo. Veja o aviso no log antes de rodar o conjunto inteiro."
            )
    return WorkflowEvalResult(k=k, metrics=_workflow_metrics(results, k), cases=results)


def _workflow_metrics(results: list[WorkflowCaseResult], k: int) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {}
    latencies = sorted(result.took_ms for result in results)
    metrics: dict[str, float] = {
        "cases": float(total),
        "no_context_rate": sum(r.kept == 0 for r in results) / total,
        "degraded_rate": sum(r.degraded for r in results) / total,
        "second_attempt_rate": sum(r.attempts > 1 for r in results) / total,
        "avg_kept": sum(r.kept for r in results) / total,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": latencies[min(total - 1, int(total * 0.95))],
        "llm_calls": float(sum(r.llm_calls for r in results)),
        "input_tokens": float(sum(r.input_tokens for r in results)),
        "output_tokens": float(sum(r.output_tokens for r in results)),
    }
    groups: dict[str, list[WorkflowCaseResult]] = {"": results}
    for attribute in ("lang", "kind"):
        for value in sorted({getattr(r, attribute) for r in results}):
            groups[f"_{attribute}_{value}"] = [r for r in results if getattr(r, attribute) == value]
    for stage in ("raw", "rewritten", "graded"):
        for suffix, group in groups.items():
            ranks = [getattr(r, f"rank_{stage}") for r in group]
            metrics[f"{stage}_hit_rate_at_{k}{suffix}"] = sum(
                rank is not None for rank in ranks
            ) / len(group)
        all_ranks = [getattr(r, f"rank_{stage}") for r in results]
        metrics[f"{stage}_mrr_at_{k}"] = sum(1 / rank for rank in all_ranks if rank) / total
    return {name: round(value, 4) for name, value in metrics.items()}


def log_workflow_to_mlflow(result: WorkflowEvalResult, params: dict[str, str]) -> str:
    import mlflow

    with mlflow.start_run(run_name="workflow-llm") as run:
        mlflow.log_params({"mode": "workflow", "k": result.k, **params})
        mlflow.log_metrics(result.metrics)
        mlflow.log_dict({"cases": [asdict(case) for case in result.cases]}, "cases.json")
        return str(run.info.run_id)


def log_to_mlflow(result: EvalResult, params: dict[str, str]) -> str:
    """Grava um run com parâmetros, métricas e o detalhe por caso. Devolve o id do run."""
    import mlflow

    with mlflow.start_run(run_name=f"retrieval-{result.mode}") as run:
        mlflow.log_params({"mode": result.mode, "k": result.k, **params})
        mlflow.log_metrics(result.metrics)
        mlflow.log_dict({"cases": [asdict(case) for case in result.cases]}, "cases.json")
        return str(run.info.run_id)
