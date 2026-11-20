from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakes import HashEmbedder
from fastapi.testclient import TestClient
from pydantic import SecretStr

from bula_knowledge.api import Deps, create_app
from bula_knowledge.config import Settings
from bula_knowledge.db import Database
from bula_knowledge.rag.workflow import RagWorkflow
from bula_knowledge.search import Searcher

pytestmark = pytest.mark.db


def _client(indexed: Database, embedder: HashEmbedder, settings: Settings) -> TestClient:
    searcher = Searcher(indexed, embedder, candidates=30, rrf_k=60)
    deps = Deps(settings, indexed, embedder, searcher, RagWorkflow(searcher, None, settings))
    return TestClient(create_app(deps))


@pytest.fixture
def client(indexed: Database, embedder: HashEmbedder, settings: Settings) -> Iterator[TestClient]:
    with _client(indexed, embedder, settings) as test_client:
        yield test_client


def test_health_reports_index_and_mode(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "offline"
    assert body["chunks"] > 0


def test_search_returns_chunks_with_citation_fields(client: TestClient) -> None:
    response = client.post("/v1/search", json={"query": "retryLimit", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert len(body["chunks"]) == 2
    assert {"id", "source", "title", "heading", "url", "content", "score"} <= set(body["chunks"][0])


def test_context_runs_the_workflow(client: TestClient) -> None:
    response = client.post("/v1/context", json={"question": "what is the default retryLimit"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["chunks"][0]["heading"] == "Retries › The retryLimit option"


def test_sources_lists_what_is_indexed(client: TestClient) -> None:
    sources = client.get("/v1/sources").json()["sources"]
    assert [(s["id"], s["documents"], s["license"]) for s in sources] == [("kiteq", 3, "fictícia")]


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    assert client.get("/health", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"
    assert len(client.get("/health").headers["x-request-id"]) == 32


def test_invalid_and_oversized_input_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "k": 500}).status_code == 422
    assert client.post("/v1/context", json={"question": "x" * 5000}).status_code == 413


def test_token_is_required_when_configured(
    indexed: Database, embedder: HashEmbedder, settings: Settings
) -> None:
    protected = settings.model_copy(update={"knowledge_api_token": SecretStr("segredo-de-teste")})
    with _client(indexed, embedder, protected) as client:
        assert client.post("/v1/search", json={"query": "x"}).status_code == 401
        wrong = {"authorization": "Bearer errado"}
        assert client.post("/v1/search", json={"query": "x"}, headers=wrong).status_code == 401
        right = {"authorization": "Bearer segredo-de-teste"}
        assert client.post("/v1/search", json={"query": "x"}, headers=right).status_code == 200
        # A sonda de saúde fica aberta para o orquestrador.
        assert client.get("/health").status_code == 200
