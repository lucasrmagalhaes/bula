"""API HTTP do serviço de conhecimento."""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from bula_knowledge import __version__
from bula_knowledge.config import Settings, get_settings
from bula_knowledge.db import Database
from bula_knowledge.embeddings import Embedder, FastEmbedEmbedder
from bula_knowledge.llm import build_structured_llm
from bula_knowledge.rag.workflow import RagWorkflow
from bula_knowledge.schemas import (
    ContextRequest,
    ContextResponse,
    Health,
    SearchRequest,
    SearchResponse,
    SourceInfo,
    SourcesResponse,
)
from bula_knowledge.search import Searcher
from bula_knowledge.tracking import setup_tracing

log = logging.getLogger("bula.knowledge.api")


@dataclass
class Deps:
    settings: Settings
    db: Database
    embedder: Embedder
    searcher: Searcher
    workflow: RagWorkflow


def build_deps(settings: Settings) -> Deps:
    db = Database(settings.database_url, settings.data_dir)
    embedder = FastEmbedEmbedder(
        settings.embeddings_model, settings.embeddings_dim, settings.data_dir / "models"
    )
    db.migrate(embedding_model=embedder.name, embedding_dim=embedder.dim)
    searcher = Searcher(
        db,
        embedder,
        candidates=settings.search_candidates,
        rrf_k=settings.rrf_k,
        min_coverage=settings.lexical_min_coverage,
    )
    workflow = RagWorkflow(
        searcher, build_structured_llm(settings), settings, tracing=setup_tracing(settings)
    )
    return Deps(settings, db, embedder, searcher, workflow)


def create_app(deps: Deps | None = None) -> FastAPI:
    """Com `deps`, a aplicação usa as dependências recebidas (testes). Sem, monta as reais."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = deps is None
        app.state.deps = deps or build_deps(get_settings())
        if owned and isinstance(app.state.deps.embedder, FastEmbedEmbedder):
            # Carrega o modelo na subida para a primeira busca não pagar por isso.
            app.state.deps.embedder.warm_up()
        yield
        if owned:
            app.state.deps.db.close()

    app = FastAPI(title="Bula Knowledge", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("erro não tratado", extra={"request_id": request_id})
            response = JSONResponse(
                {"detail": "erro interno", "request_id": request_id}, status_code=500
            )
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response

    def get_deps(request: Request) -> Deps:
        return request.app.state.deps

    def authorize(request: Request, deps: Deps = Depends(get_deps)) -> None:
        expected = deps.settings.knowledge_api_token
        if expected is None:
            return
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        valid = scheme.lower() == "bearer" and hmac.compare_digest(
            token.encode(), expected.get_secret_value().encode()
        )
        if not valid:
            raise HTTPException(status_code=401, detail="token ausente ou inválido")

    @app.get("/health", response_model=Health)
    def health(deps: Deps = Depends(get_deps)) -> Health:
        return Health(
            status="ok",
            version=__version__,
            embedding_model=deps.embedder.name,
            llm_mode=deps.workflow.mode,
            chunks=deps.searcher.count_chunks(),
        )

    @app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(authorize)])
    def search(body: SearchRequest, deps: Deps = Depends(get_deps)) -> SearchResponse:
        _check_length(body.query, deps.settings.max_question_chars)
        chunks, took_ms = deps.searcher.search(
            body.query, k=body.k or deps.settings.search_k, mode=body.mode, sources=body.sources
        )
        return SearchResponse(
            chunks=chunks, mode=body.mode, embedding_model=deps.embedder.name, took_ms=took_ms
        )

    @app.post("/v1/context", response_model=ContextResponse, dependencies=[Depends(authorize)])
    def context(
        body: ContextRequest, request: Request, deps: Deps = Depends(get_deps)
    ) -> ContextResponse:
        _check_length(body.question, deps.settings.max_question_chars)
        result = deps.workflow.run(body.question, body.sources, request_id=request.state.request_id)
        log.info(
            "rag_context",
            extra={
                "request_id": request.state.request_id,
                "status": result.status,
                "attempts": result.attempts,
                "chunks": len(result.chunks),
                "degraded": result.degraded,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "timings_ms": result.timings_ms,
            },
        )
        return result

    @app.get("/v1/sources", response_model=SourcesResponse, dependencies=[Depends(authorize)])
    def sources(deps: Deps = Depends(get_deps)) -> SourcesResponse:
        return SourcesResponse(embedding_model=deps.embedder.name, sources=list_sources(deps.db))

    return app


def list_sources(db: Database) -> list[SourceInfo]:
    with db.pool.connection() as conn:
        rows = conn.execute(
            """
            select s.id, s.title, s.repo, s.ref, s.commit_sha, s.license,
                   count(distinct d.id), count(c.id), s.ingested_at
            from sources s
            left join documents d on d.source_id = s.id
            left join chunks c on c.document_id = d.id
            group by s.id
            order by s.id
            """
        ).fetchall()
    return [
        SourceInfo(
            id=row[0],
            title=row[1],
            repo=row[2],
            ref=row[3],
            commit_sha=row[4],
            license=row[5],
            documents=row[6],
            chunks=row[7],
            ingested_at=row[8].isoformat(timespec="seconds"),
        )
        for row in rows
    ]


def _check_length(text: str, limit: int) -> None:
    if len(text) > limit:
        raise HTTPException(status_code=413, detail=f"texto acima de {limit} caracteres")
