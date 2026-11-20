from __future__ import annotations

import re

import pytest
from fakes import ScriptedLlm

from bula_knowledge.config import Settings
from bula_knowledge.rag.workflow import RagWorkflow
from bula_knowledge.search import Searcher

pytestmark = pytest.mark.db

_PASSAGE = re.compile(r'<passage id="(\d+)" source="[^"]*" heading="([^"]*)">')


def _ids_where(user: str, text: str) -> list[int]:
    return [int(pid) for pid, heading in _PASSAGE.findall(user) if text in heading]


def test_offline_mode_searches_the_question_as_is(searcher: Searcher, settings: Settings) -> None:
    result = RagWorkflow(searcher, None, settings).run("what is the default retryLimit")
    assert result.status == "ok"
    assert result.mode == "offline"
    assert result.queries == ["what is the default retryLimit"]
    assert result.attempts == 1
    assert result.usage.llm_calls == 0
    assert result.chunks[0].heading == "Retries › The retryLimit option"


def test_llm_rewrites_then_keeps_only_graded_passages(
    searcher: Searcher, settings: Settings
) -> None:
    llm = ScriptedLlm(
        {
            "SearchPlan": lambda _: {
                "queries": ["retryLimit default attempts", "retry failed job"]
            },
            "Grade": lambda user: {"relevant_ids": [*_ids_where(user, "retryLimit"), 999_999]},
        }
    )
    result = RagWorkflow(searcher, llm, settings).run("quantas tentativas por padrão?")
    assert result.status == "ok"
    assert result.mode == "llm"
    assert [c.heading for c in result.chunks] == ["Retries › The retryLimit option"]
    assert result.usage.llm_calls == 2
    assert result.usage.input_tokens == 20
    assert {"rewrite", "retrieve", "grade", "total"} <= set(result.timings_ms)


def test_retries_with_new_queries_when_nothing_is_relevant(
    searcher: Searcher, settings: Settings
) -> None:
    plans = iter([{"queries": ["installation"]}, {"queries": ["KQ1042 lock"]}])
    # A primeira triagem não aprova nada; a segunda aprova o trecho do erro.
    grades = iter([lambda _: [], lambda user: _ids_where(user, "KQ1042")])
    llm = ScriptedLlm(
        {
            "SearchPlan": lambda _: next(plans),
            "Grade": lambda user: {"relevant_ids": next(grades)(user)},
        }
    )
    result = RagWorkflow(searcher, llm, settings).run("o que é o erro KQ1042?")
    assert result.status == "ok"
    assert result.attempts == 2
    assert result.queries == ["installation", "KQ1042 lock"]
    # A segunda reescrita recebe o que já foi tentado.
    assert "installation" in llm.calls[2][1]


def test_gives_up_with_no_context_at_the_attempt_cap(
    searcher: Searcher, settings: Settings
) -> None:
    llm = ScriptedLlm(
        {
            "SearchPlan": lambda _: {"queries": ["kubernetes ingress"]},
            "Grade": lambda _: {"relevant_ids": []},
        }
    )
    result = RagWorkflow(searcher, llm, settings).run("como configuro o ingress do kubernetes?")
    assert result.status == "no_context"
    assert result.chunks == []
    assert result.attempts == settings.rag_max_attempts
    assert [name for name, _ in llm.calls].count("SearchPlan") == settings.rag_max_attempts


def test_llm_failure_degrades_to_plain_search_instead_of_failing(
    searcher: Searcher, settings: Settings
) -> None:
    def boom(_: str) -> dict:
        raise RuntimeError("provedor fora do ar")

    llm = ScriptedLlm({"SearchPlan": boom, "Grade": boom})
    result = RagWorkflow(searcher, llm, settings).run("default retryLimit")
    assert result.status == "ok"
    assert result.degraded is True
    assert result.attempts == 1
    assert result.chunks


def test_offline_mode_returns_no_context_when_nothing_is_close(
    searcher: Searcher, settings: Settings
) -> None:
    # Nenhuma palavra em comum com o corpus: sem termo casado e longe demais no vetor.
    result = RagWorkflow(searcher, None, settings).run("receita de bolo de cenoura")
    assert result.status == "no_context"
    assert result.chunks == []
