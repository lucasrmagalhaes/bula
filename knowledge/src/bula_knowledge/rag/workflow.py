"""Workflow de RAG em LangGraph: reescreve a pergunta, busca, tria e tenta de novo.

É um workflow e não um agente: o caminho está escrito no grafo e o modelo só
preenche duas decisões tipadas (as consultas e os ids relevantes). O limite de
tentativas é do código.

    START -> rewrite -> retrieve -> grade -+-> END
                ^                          |
                +---- nada relevante ------+

Sem LLM (LLM_PROVIDER=offline) o grafo é o mesmo, mas a reescrita devolve a própria
pergunta e a triagem vira um corte por distância vetorial.
"""

from __future__ import annotations

import logging
import time
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from bula_knowledge.config import Settings
from bula_knowledge.llm import StructuredLlm
from bula_knowledge.rag.prompts import GRADE_SYSTEM, REWRITE_SYSTEM
from bula_knowledge.schemas import Chunk, ContextResponse, Usage
from bula_knowledge.search import Searcher

log = logging.getLogger("bula.knowledge.rag")


class SearchPlan(BaseModel):
    """Consultas de busca derivadas da pergunta."""

    queries: list[str] = Field(
        min_length=1, max_length=3, description="Search queries in English, most specific first."
    )


class Grade(BaseModel):
    """Triagem dos trechos recuperados."""

    relevant_ids: list[int] = Field(
        description="Ids of the passages that help answer the question. Empty when none does."
    )


def _sum(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged


class RagState(TypedDict, total=False):
    question: str
    sources: list[str] | None
    queries: list[str]
    tried: Annotated[list[str], add]
    attempts: int
    candidates: list[Chunk]
    relevant: list[Chunk]
    degraded: bool
    usage: Annotated[dict[str, float], _sum]
    timings: Annotated[dict[str, float], _sum]


class RagWorkflow:
    def __init__(
        self,
        searcher: Searcher,
        llm: StructuredLlm | None,
        settings: Settings,
        *,
        tracing: bool = False,
    ) -> None:
        self._searcher = searcher
        self._llm = llm
        self._settings = settings
        self._tracing = tracing
        self._graph = self._build()

    @property
    def mode(self) -> Literal["llm", "offline"]:
        return "llm" if self._llm is not None else "offline"

    def _build(self) -> Any:
        graph = StateGraph(RagState)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self._next, {"rewrite": "rewrite", "finish": END})
        return graph.compile()

    # --- nós ---------------------------------------------------------------

    def _rewrite(self, state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        queries = [question]
        usage: dict[str, float] = {}
        degraded = False
        if self._llm is not None:
            tried = state.get("tried", [])
            user = f"Question:\n{question}"
            if tried:
                listed = "\n".join(f"- {query}" for query in tried)
                user += f"\n\nQueries already tried without useful results:\n{listed}"
            try:
                plan, spent = self._llm.structured(SearchPlan, system=REWRITE_SYSTEM, user=user)
                cleaned = list(dict.fromkeys(q.strip() for q in plan.queries if q.strip()))
                queries = cleaned[:3] or [question]
                usage = _usage(spent)
            except Exception:
                # A reescrita é uma melhoria, não um requisito: sem ela, busca a pergunta crua.
                log.warning("rewrite falhou; buscando a pergunta original", exc_info=True)
                degraded = True
        return {
            "queries": queries,
            "tried": queries,
            "attempts": state.get("attempts", 0) + 1,
            "degraded": state.get("degraded", False) or degraded,
            "usage": usage,
            "timings": {"rewrite": _elapsed(started)},
        }

    def _retrieve(self, state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        limit = self._settings.search_k * 2
        scores: dict[int, float] = {}
        found: dict[int, Chunk] = {}
        # RRF também entre as consultas: um trecho bem colocado em mais de uma sobe.
        for query in state["queries"]:
            chunks, _ = self._searcher.search(query, k=limit, sources=state.get("sources"))
            for rank, chunk in enumerate(chunks, start=1):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (self._settings.rrf_k + rank)
                found.setdefault(chunk.id, chunk)
        ordered = sorted(found.values(), key=lambda chunk: (-scores[chunk.id], chunk.id))[:limit]
        candidates = [chunk.model_copy(update={"score": scores[chunk.id]}) for chunk in ordered]
        return {"candidates": candidates, "timings": {"retrieve": _elapsed(started)}}

    def _grade(self, state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        candidates = state.get("candidates", [])
        top = self._settings.search_k
        relevant = self._plausible(candidates)[:top]
        usage: dict[str, float] = {}
        degraded = False
        if self._llm is not None and candidates:
            limit = self._settings.rag_passage_chars
            passages = "\n\n".join(
                f'<passage id="{c.id}" source="{c.source}" heading="{c.heading}">\n'
                f"{c.content[:limit]}\n</passage>"
                for c in candidates
            )
            user = f"Question:\n{state['question']}\n\nPassages:\n{passages}"
            try:
                grade, spent = self._llm.structured(Grade, system=GRADE_SYSTEM, user=user)
                # O modelo só escolhe entre os ids que recebeu; id desconhecido é descartado.
                chosen = set(grade.relevant_ids)
                relevant = [c for c in candidates if c.id in chosen][:top]
                usage = _usage(spent)
            except Exception:
                log.warning("grade falhou; devolvendo os primeiros da busca", exc_info=True)
                degraded = True
        return {
            "relevant": relevant,
            "degraded": state.get("degraded", False) or degraded,
            "usage": usage,
            "timings": {"grade": _elapsed(started)},
        }

    def _plausible(self, candidates: list[Chunk]) -> list[Chunk]:
        """Triagem sem LLM: fica o trecho que casou termo da pergunta ou que está perto dela."""
        limit = self._settings.offline_max_distance
        return [
            c
            for c in candidates
            if c.lexical_rank is not None
            or (c.vector_distance is not None and c.vector_distance <= limit)
        ]

    def _next(self, state: RagState) -> Literal["rewrite", "finish"]:
        if state.get("relevant") or self._llm is None or state.get("degraded"):
            return "finish"
        if state.get("attempts", 0) >= self._settings.rag_max_attempts:
            return "finish"
        return "rewrite"

    # --- execução ----------------------------------------------------------

    def invoke(
        self,
        question: str,
        sources: list[str] | None = None,
        *,
        request_id: str | None = None,
    ) -> RagState:
        """Roda o grafo e devolve o estado final inteiro, com os candidatos de antes da
        triagem. É o que a avaliação usa para medir cada etapa em separado."""
        config = {
            # Teto do LangGraph por cima do contador de tentativas do próprio estado.
            "recursion_limit": 3 * self._settings.rag_max_attempts + 3,
            "run_name": "rag_context",
            "metadata": {"request_id": request_id or ""},
        }
        state: RagState = self._graph.invoke(
            {"question": question, "sources": sources, "attempts": 0}, config=config
        )
        return state

    def run(
        self,
        question: str,
        sources: list[str] | None = None,
        *,
        request_id: str | None = None,
    ) -> ContextResponse:
        started = time.perf_counter()
        state = self.invoke(question, sources, request_id=request_id)
        trace_id = self._last_trace_id() if self._tracing else None
        usage = state.get("usage", {})
        timings = dict(state.get("timings", {}))
        timings["total"] = _elapsed(started)
        relevant = state.get("relevant", [])
        return ContextResponse(
            status="ok" if relevant else "no_context",
            question=question,
            queries=state.get("tried", []),
            attempts=state.get("attempts", 0),
            chunks=relevant,
            mode=self.mode,
            degraded=state.get("degraded", False),
            usage=Usage(
                llm_calls=int(usage.get("llm_calls", 0)),
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            timings_ms=timings,
            trace_id=trace_id,
        )

    @staticmethod
    def _last_trace_id() -> str | None:
        try:
            import mlflow

            return mlflow.get_last_active_trace_id(thread_local=True)
        except Exception:  # pragma: no cover - tracing nunca derruba a resposta
            return None


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _usage(spent: Any) -> dict[str, float]:
    return {
        "llm_calls": 1,
        "input_tokens": spent.input_tokens,
        "output_tokens": spent.output_tokens,
    }
