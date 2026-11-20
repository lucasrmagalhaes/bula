from __future__ import annotations

import pytest
from conftest import FIXTURES, KITEQ
from fakes import HashEmbedder

from bula_knowledge.db import Database, IndexMismatchError
from bula_knowledge.ingest.fetch import RawDocument, load_local
from bula_knowledge.ingest.pipeline import ingest_source
from bula_knowledge.ingest.sources import SOURCES, Source
from bula_knowledge.search import Searcher

pytestmark = pytest.mark.db


def test_hybrid_search_finds_the_section_that_answers(searcher: Searcher) -> None:
    chunks, _ = searcher.search("is a failed job retried automatically", k=3)
    assert chunks[0].heading.startswith("Retries")
    assert chunks[0].url.endswith("retries.md#retries") or "retries.md#" in chunks[0].url


def test_exact_identifier_is_found_by_the_lexical_leg(searcher: Searcher) -> None:
    chunks, _ = searcher.search("KQ1042", k=3, mode="lexical")
    assert chunks, "o código de erro tem que casar por termo exato"
    assert chunks[0].heading == "Error codes › KQ1042"
    assert chunks[0].lexical_rank == 1
    assert chunks[0].vector_rank is None


def test_modes_report_which_leg_ranked_each_chunk(searcher: Searcher) -> None:
    vector, _ = searcher.search("exponential backoff delay", k=3, mode="vector")
    assert all(c.vector_rank is not None and c.lexical_rank is None for c in vector)
    hybrid, _ = searcher.search("exponential backoff delay", k=3, mode="hybrid")
    assert hybrid[0].heading == "Retries › Backoff"
    assert hybrid[0].vector_rank is not None and hybrid[0].lexical_rank is not None


def test_query_with_no_lexical_match_still_returns_vector_results(searcher: Searcher) -> None:
    # Só stopwords: a perna textual fica vazia e a consulta não pode quebrar.
    chunks, _ = searcher.search("the of and", k=2)
    assert len(chunks) == 2


def test_source_filter_limits_the_results(searcher: Searcher) -> None:
    assert searcher.search("retry", k=5, sources=["kiteq"])[0]
    assert searcher.search("retry", k=5, sources=["outra-fonte"])[0] == []


def test_citation_url_carries_the_section_anchor(searcher: Searcher) -> None:
    chunks, _ = searcher.search("dead letter queue", k=1)
    assert chunks[0].url == "retries.md#dead-letter-queue"


def test_reingest_is_idempotent_and_tracks_changes(db: Database, embedder: HashEmbedder) -> None:
    documents = load_local(FIXTURES)
    first = ingest_source(db, embedder, KITEQ, documents, None)
    assert (first.updated, first.unchanged) == (3, 0)

    second = ingest_source(db, embedder, KITEQ, documents, None)
    assert (second.updated, second.unchanged, second.chunks_embedded) == (0, 3, 0)

    edited = [
        RawDocument(d.path, d.text + "\n\n## New\n\nAdded section.") if "retries" in d.path else d
        for d in documents
        if "errors" not in d.path
    ]
    third = ingest_source(db, embedder, KITEQ, edited, None)
    assert (third.updated, third.unchanged, third.removed) == (1, 1, 1)

    searcher = Searcher(db, embedder, candidates=30, rrf_k=60)
    assert searcher.search("KQ1042", k=3, mode="lexical")[0] == []
    assert searcher.search("added section", k=1)[0][0].heading == "Retries › New"


def test_index_built_with_another_embedding_model_is_refused(database: Database) -> None:
    with pytest.raises(IndexMismatchError, match="--rebuild"):
        database.migrate(embedding_model="outro-modelo", embedding_dim=384)


def test_rare_term_outranks_a_term_that_is_everywhere(searcher: Searcher) -> None:
    # "kiteq" aparece em quase todo trecho; "backoff", em um só. O IDF decide.
    chunks, _ = searcher.search("kiteq backoff", k=1, mode="lexical")
    assert chunks[0].heading == "Retries › Backoff"


def test_lexical_leg_stays_silent_without_enough_evidence(searcher: Searcher) -> None:
    # Pergunta em português num corpus em inglês: só "kiteq" casa, e isso é uma fração
    # pequena do peso da pergunta. Melhor não devolver nada do que devolver ruído.
    question = "como configurar a quantidade de tentativas no kiteq"
    assert searcher.search(question, k=3, mode="lexical")[0] == []
    assert len(searcher.search(question, k=3, mode="hybrid")[0]) == 3


def test_citation_url_by_kind_of_source() -> None:
    fastify = SOURCES["fastify"]
    assert fastify.url_for("docs/Reference/Hooks.md", "abc123") == (
        "https://github.com/fastify/fastify/blob/abc123/docs/Reference/Hooks.md"
    )
    published = Source(
        id="x",
        title="X",
        repo="local",
        ref="local",
        license="-",
        prefix="",
        base_url="https://docs.example.test/",
    )
    assert published.url_for("guia/inicio.md", None) == "https://docs.example.test/guia/inicio.md"
