"""Contrato HTTP do serviço. O cliente do agente (Node) valida as respostas contra isto."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SearchMode = Literal["hybrid", "vector", "lexical"]


class Chunk(BaseModel):
    id: int
    source: str
    title: str
    heading: str
    url: str
    content: str
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    # Distância de cosseno até a pergunta (0 = igual). Ausente se o trecho só veio da busca textual.
    vector_distance: float | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=20)
    mode: SearchMode = "hybrid"
    sources: list[str] | None = None


class SearchResponse(BaseModel):
    chunks: list[Chunk]
    mode: SearchMode
    embedding_model: str
    took_ms: float


class ContextRequest(BaseModel):
    question: str = Field(min_length=1)
    sources: list[str] | None = None


class Usage(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ContextResponse(BaseModel):
    """Saída do workflow de RAG: os trechos que sustentam uma resposta, ou nenhum."""

    status: Literal["ok", "no_context"]
    question: str
    queries: list[str]
    attempts: int
    chunks: list[Chunk]
    mode: Literal["llm", "offline"]
    # True quando uma chamada de LLM falhou e o passo caiu no comportamento offline.
    degraded: bool = False
    usage: Usage
    timings_ms: dict[str, float]
    trace_id: str | None = None


class SourceInfo(BaseModel):
    id: str
    title: str
    repo: str
    ref: str
    commit_sha: str | None
    license: str
    documents: int
    chunks: int
    ingested_at: str


class SourcesResponse(BaseModel):
    embedding_model: str
    sources: list[SourceInfo]


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    embedding_model: str
    llm_mode: Literal["llm", "offline"]
    chunks: int
