"""Ingestão: documentos brutos viram trechos vetorizados no Postgres.

É idempotente. Documento com o mesmo hash não é vetorizado de novo, e documento
que saiu da fonte sai do índice. Cada fonte entra numa transação só, então quem
está buscando vê o índice antigo ou o novo, nunca um pela metade.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from bula_knowledge.db import Database
from bula_knowledge.embeddings import Embedder
from bula_knowledge.ingest.chunk import ChunkDraft, full_heading, parse_document
from bula_knowledge.ingest.fetch import RawDocument
from bula_knowledge.ingest.sources import Source

# Entra no hash de cada documento. Mudou a divisão em trechos ou o texto vetorizado?
# Suba este número: a próxima ingestão refaz tudo, em vez de pular o que "não mudou".
PIPELINE_VERSION = 2


@dataclass
class IngestReport:
    source: str
    documents: int = 0
    unchanged: int = 0
    updated: int = 0
    removed: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0


def ingest_source(
    db: Database,
    embedder: Embedder,
    source: Source,
    documents: list[RawDocument],
    commit_sha: str | None,
) -> IngestReport:
    started = time.perf_counter()
    report = IngestReport(source=source.id, documents=len(documents))

    with db.pool.connection() as conn:
        existing = dict(
            conn.execute(
                "select path, content_hash from documents where source_id = %s", (source.id,)
            ).fetchall()
        )

    # Vetoriza fora da transação: é a parte lenta e não precisa segurar conexão.
    changed: list[tuple[RawDocument, str, str, list[ChunkDraft], list[list[float]]]] = []
    for document in documents:
        digest = hashlib.sha256(f"{PIPELINE_VERSION}\n{document.text}".encode()).hexdigest()
        if existing.get(document.path) == digest:
            report.unchanged += 1
            continue
        parsed = parse_document(document.text, fallback_title=_title_from_path(document.path))
        texts = [
            f"{full_heading(parsed.title, c.heading_path)}\n\n{c.content}" for c in parsed.chunks
        ]
        vectors = embedder.embed_documents(texts)
        changed.append((document, digest, parsed.title, parsed.chunks, vectors))
        report.chunks_embedded += len(vectors)

    current_paths = [document.path for document in documents]
    with db.pool.connection() as conn, conn.transaction():
        conn.execute(
            """
            insert into sources (id, title, repo, ref, commit_sha, license, ingested_at)
            values (%s, %s, %s, %s, %s, %s, now())
            on conflict (id) do update set
              title = excluded.title, repo = excluded.repo, ref = excluded.ref,
              commit_sha = excluded.commit_sha, license = excluded.license,
              ingested_at = excluded.ingested_at
            """,
            (source.id, source.title, source.repo, source.ref, commit_sha, source.license),
        )
        removed = conn.execute(
            "delete from documents where source_id = %s and not (path = any(%s))",
            (source.id, current_paths),
        )
        report.removed = removed.rowcount
        for document, digest, title, chunks, vectors in changed:
            conn.execute(
                "delete from documents where source_id = %s and path = %s",
                (source.id, document.path),
            )
            row = conn.execute(
                """
                insert into documents (source_id, path, title, url, content_hash)
                values (%s, %s, %s, %s, %s) returning id
                """,
                (
                    source.id,
                    document.path,
                    title,
                    source.url_for(document.path, commit_sha),
                    digest,
                ),
            ).fetchone()
            assert row is not None
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    insert into chunks
                      (document_id, ordinal, heading_path, anchor, content,
                       token_estimate, embedding)
                    values (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            row[0],
                            ordinal,
                            chunk.heading_path,
                            chunk.anchor,
                            chunk.content,
                            max(1, len(chunk.content) // 4),
                            np.asarray(vector, dtype=np.float32),
                        )
                        for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                    ],
                )
            report.updated += 1
        if changed or report.removed:
            # O IDF muda com o corpus; refaz na mesma transação para a busca ver tudo junto.
            conn.execute("truncate lexeme_stats")
            conn.execute(
                "insert into lexeme_stats (lexeme, ndoc) "
                "select word, ndoc from ts_stat('select tsv from chunks')"
            )

    if changed or report.removed:
        # Depois de uma carga o planejador ainda não tem estatísticas das tabelas, e a
        # busca textual cai num plano ruim até o autovacuum passar. Não espera por ele.
        with db.pool.connection() as conn:
            conn.execute("analyze chunks")
            conn.execute("analyze lexeme_stats")

    report.seconds = round(time.perf_counter() - started, 2)
    return report


def _title_from_path(path: str) -> str:
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").replace("_", " ").strip().title() or path
