from __future__ import annotations

import re

import pytest
from fakes import ScriptedLlm

from bula_knowledge.config import Settings
from bula_knowledge.evaluation.retrieval import (
    EvalAbortedError,
    GoldenCase,
    WorkflowCaseResult,
    evaluate_workflow,
)
from bula_knowledge.rag.workflow import RagWorkflow
from bula_knowledge.search import Searcher

pytestmark = pytest.mark.db

_PASSAGE = re.compile(r'<passage id="(\d+)" source="[^"]*" heading="([^"]*)">')

CASES = [
    GoldenCase("retry", "quantas tentativas por padrão?", "pt", ("retries.md",)),
    GoldenCase("codigo", "KQ1042", "en", ("errors.mdx",), kind="identifier"),
    GoldenCase("ausente", "como configuro o ingress?", "pt", ("nao-existe.md",)),
]


def _llm() -> ScriptedLlm:
    """Reescreve as perguntas conhecidas para inglês e só aprova o trecho que responde."""
    rewrites = {"tentativas": "retryLimit default attempts", "KQ1042": "KQ1042 lock"}
    approve = {"tentativas": "retryLimit", "KQ1042": "KQ1042"}

    def plan(user: str) -> dict:
        query = next((q for word, q in rewrites.items() if word in user), "kubernetes ingress")
        return {"queries": [query]}

    def grade(user: str) -> dict:
        # Decide só pela pergunta: o texto dos trechos também vem no pedido.
        question = user.split("\n\nPassages:")[0]
        wanted = next((text for word, text in approve.items() if word in question), None)
        ids = [int(pid) for pid, heading in _PASSAGE.findall(user) if wanted and wanted in heading]
        return {"relevant_ids": ids}

    return ScriptedLlm({"SearchPlan": plan, "Grade": grade})


def test_measures_each_stage_of_the_workflow(searcher: Searcher, settings: Settings) -> None:
    seen: list[tuple[int, WorkflowCaseResult]] = []
    result = evaluate_workflow(
        RagWorkflow(searcher, _llm(), settings),
        searcher,
        CASES,
        k=5,
        on_case=lambda number, case: seen.append((number, case)),
    )

    assert [number for number, _ in seen] == [1, 2, 3]
    retry, code, absent = result.cases

    # A reescrita leva a pergunta em português ao trecho certo, e a triagem fica só com ele.
    assert retry.queries == ["retryLimit default attempts"]
    assert retry.rank_rewritten == 1
    assert (retry.rank_graded, retry.kept, retry.attempts) == (1, 1, 1)
    assert (retry.llm_calls, retry.input_tokens, retry.output_tokens) == (2, 20, 10)

    assert (code.rank_raw, code.rank_graded) == (1, 1)

    # Sem resposta no corpus: duas tentativas, nada aprovado.
    assert (absent.rank_graded, absent.kept, absent.attempts) == (None, 0, 2)
    assert absent.llm_calls == 4

    metrics = result.metrics
    assert metrics["cases"] == 3
    assert metrics["graded_hit_rate_at_5"] == round(2 / 3, 4)
    assert metrics["graded_hit_rate_at_5_lang_pt"] == 0.5
    assert metrics["graded_hit_rate_at_5_kind_identifier"] == 1.0
    assert metrics["no_context_rate"] == round(1 / 3, 4)
    assert metrics["second_attempt_rate"] == round(1 / 3, 4)
    assert metrics["degraded_rate"] == 0
    assert metrics["llm_calls"] == 8
    assert metrics["input_tokens"] == 80
    assert {"raw_mrr_at_5", "rewritten_mrr_at_5", "graded_mrr_at_5"} <= set(metrics)


def test_aborts_when_the_model_is_not_answering(searcher: Searcher, settings: Settings) -> None:
    def boom(_: str) -> dict:
        raise RuntimeError("provedor fora do ar")

    calls: list[int] = []
    workflow = RagWorkflow(searcher, ScriptedLlm({"SearchPlan": boom, "Grade": boom}), settings)
    with pytest.raises(EvalAbortedError, match="modo degradado"):
        evaluate_workflow(
            workflow, searcher, CASES * 4, k=5, on_case=lambda number, _: calls.append(number)
        )
    # Para na terceira pergunta, em vez de gastar o conjunto inteiro.
    assert calls == [1, 2, 3]
