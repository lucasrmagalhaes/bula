from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import HashEmbedder

from bula_knowledge.config import Settings
from bula_knowledge.db import Database
from bula_knowledge.ingest.fetch import load_local
from bula_knowledge.ingest.pipeline import ingest_source
from bula_knowledge.ingest.sources import Source
from bula_knowledge.search import Searcher

FIXTURES = Path(__file__).parent / "fixtures" / "docs"

KITEQ = Source(id="kiteq", title="Kiteq", repo="local", ref="local", license="fictícia", prefix="")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # `_env_file=None` isola os testes do .env real da máquina.
    return Settings(_env_file=None, data_dir=tmp_path, llm_provider="offline")


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture(scope="session")
def database(
    tmp_path_factory: pytest.TempPathFactory, embedder: HashEmbedder
) -> Iterator[Database]:
    """Um Postgres embutido por sessão de testes."""
    db = Database(None, tmp_path_factory.mktemp("pg"))
    db.migrate(embedding_model=embedder.name, embedding_dim=embedder.dim)
    yield db
    db.close()


@pytest.fixture
def db(database: Database) -> Iterator[Database]:
    yield database
    with database.pool.connection() as conn:
        conn.execute("truncate sources cascade")


@pytest.fixture
def indexed(db: Database, embedder: HashEmbedder) -> Database:
    ingest_source(db, embedder, KITEQ, load_local(FIXTURES), None)
    return db


@pytest.fixture
def searcher(indexed: Database, embedder: HashEmbedder) -> Searcher:
    return Searcher(indexed, embedder, candidates=30, rrf_k=60)
