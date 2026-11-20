"""Acesso ao Postgres com pgvector.

O mesmo SQL roda nos dois modos: Postgres externo (DATABASE_URL) ou um Postgres
embutido pelo pacote opcional `pgserver`, que serve para desenvolver e testar
sem Docker.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

log = logging.getLogger("bula.knowledge.db")

SCHEMA = """
create extension if not exists vector;

create table if not exists meta (
  key   text primary key,
  value text not null
);

create table if not exists sources (
  id          text primary key,
  title       text not null,
  repo        text not null,
  ref         text not null,
  commit_sha  text,
  license     text not null,
  ingested_at timestamptz not null default now()
);

create table if not exists documents (
  id           bigserial primary key,
  source_id    text not null references sources(id) on delete cascade,
  path         text not null,
  title        text not null,
  url          text not null,
  content_hash text not null,
  unique (source_id, path)
);

create table if not exists chunks (
  id             bigserial primary key,
  document_id    bigint not null references documents(id) on delete cascade,
  ordinal        int not null,
  heading_path   text not null,
  anchor         text,
  content        text not null,
  token_estimate int not null,
  embedding      vector({dim}) not null,
  tsv            tsvector generated always as (
                   setweight(to_tsvector('english', heading_path), 'A') ||
                   setweight(to_tsvector('english', content), 'B')
                 ) stored,
  unique (document_id, ordinal)
);

-- Em quantos trechos cada radical aparece: o IDF da busca textual sai daqui.
create table if not exists lexeme_stats (
  lexeme text primary key,
  ndoc   int not null
);

create index if not exists chunks_tsv_idx on chunks using gin (tsv);
create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
"""


class IndexMismatchError(RuntimeError):
    """O índice foi construído com outro modelo de embedding."""


def drop_dead_handles(pgdata: Path) -> None:
    """Tira da lista do `pgserver` os processos que não existem mais.

    O `pgserver` só para o Postgres quando sai o último processo que o usa, e guarda os
    PIDs em `.handle_pids.json`. Um processo morto à força nunca sai dessa lista, e a
    partir daí o Postgres embutido não para mais sozinho. Isto roda antes de cada subida.
    """
    path = pgdata / ".handle_pids.json"
    try:
        pids = json.loads(path.read_text())
        alive = [pid for pid in pids if holds_pgserver(pid)]
        if alive != pids:
            path.write_text(json.dumps(alive))
    except (OSError, ValueError):
        # Sem o arquivo, ou ilegível: nada a limpar, e o pgserver cuida do resto.
        return


def holds_pgserver(pid: object) -> bool:
    """Um PID da lista só conta se ainda for um processo Python.

    Não basta o PID existir: o Windows reaproveita os números, e o de um serviço morto
    já apareceu aqui pertencendo ao menu Iniciar, o que mantinha o Postgres no ar.
    """
    if not isinstance(pid, int):
        return False
    try:
        import psutil

        # O executável, e não o nome do processo: no Linux um serviço iniciado pelo
        # script `bula-knowledge` ou pelo `pytest` leva o nome do script, mas roda no
        # interpretador Python.
        executable = Path(psutil.Process(pid).exe()).name.lower()
        return executable.startswith(("python", "pypy"))
    except Exception:
        # Processo inexistente, sem permissão para ler ou psutil ausente: não conta.
        return False


class Database:
    def __init__(self, database_url: str | None, data_dir: Path) -> None:
        self._server: Any = None
        uri = database_url or self._start_embedded(data_dir)
        # A extensão precisa existir antes de o pool registrar o tipo `vector`.
        with psycopg.connect(uri, autocommit=True) as conn:
            conn.execute("create extension if not exists vector")
        self.pool = ConnectionPool(
            uri,
            min_size=1,
            max_size=5,
            configure=register_vector,
            open=True,
        )

    def _start_embedded(self, data_dir: Path) -> str:
        try:
            import pgserver
        except ImportError as exc:  # pragma: no cover - depende da instalação
            raise RuntimeError(
                "Sem DATABASE_URL, o serviço usa o Postgres embutido. "
                "Instale o extra: uv sync --extra embedded"
            ) from exc
        pgdata = data_dir / "pg"
        pgdata.mkdir(parents=True, exist_ok=True)
        drop_dead_handles(pgdata)
        # Depois de uma parada abrupta (Ctrl+C no Windows atinge o Postgres também), ele
        # refaz o WAL na subida e isso pode passar do prazo de 10 s do pgserver. O servidor
        # continua subindo, então a tentativa seguinte o encontra pronto.
        for remaining in (3, 2, 1, 0):
            try:
                self._server = pgserver.get_server(pgdata, cleanup_mode="stop")
                break
            except subprocess.TimeoutExpired:
                if remaining == 0:
                    raise RuntimeError(
                        f"O Postgres embutido não subiu a tempo. Veja o log em {pgdata / 'log'}."
                    ) from None
                log.warning("Postgres embutido em recuperação depois de parada abrupta; aguardando")
                time.sleep(15)
        return str(self._server.get_uri())

    def migrate(self, *, embedding_model: str, embedding_dim: int) -> None:
        """Cria o esquema e confere se o índice é do modelo de embedding configurado."""
        with self.pool.connection() as conn:
            conn.execute(SCHEMA.format(dim=int(embedding_dim)))
            stored = dict(
                conn.execute(
                    "select key, value from meta where key in ('embedding_model', 'embedding_dim')"
                ).fetchall()
            )
            if not stored:
                conn.execute(
                    "insert into meta (key, value) values (%s, %s), (%s, %s)",
                    ("embedding_model", embedding_model, "embedding_dim", str(embedding_dim)),
                )
                return
            if stored.get("embedding_model") != embedding_model or stored.get(
                "embedding_dim"
            ) != str(embedding_dim):
                raise IndexMismatchError(
                    f"O índice foi construído com {stored.get('embedding_model')} "
                    f"({stored.get('embedding_dim')}d) e a configuração pede {embedding_model} "
                    f"({embedding_dim}d). Vetores de modelos diferentes não se comparam: "
                    "rode `bula-knowledge ingest --rebuild`."
                )

    def reset(self) -> None:
        """Apaga o índice inteiro. Usado ao trocar o modelo de embedding."""
        with self.pool.connection() as conn:
            conn.execute(
                "drop table if exists chunks, documents, sources, meta, lexeme_stats cascade"
            )

    def close(self) -> None:
        self.pool.close()
        if self._server is not None:
            self._server.cleanup()
            self._server = None
