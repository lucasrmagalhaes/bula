"""Busca híbrida numa consulta SQL: vetorial (pgvector) e textual (tsvector), fundidas por RRF.

A busca vetorial sozinha erra termo exato, como nome de função, opção ou código de
erro. A textual sozinha erra paráfrase e pergunta em outro idioma. O RRF soma
1 / (k + posição) de cada lista, então dispensa normalizar notas que não se comparam.

A perna textual não usa o `ts_rank` puro do Postgres, que ignora a raridade do termo:
numa pergunta em linguagem natural, o nome da biblioteca casava com quase todo trecho
e dominava a lista. Aqui cada termo casado vale o seu IDF (tabela `lexeme_stats`,
refeita na ingestão), com bônus de frequência e de título vindo do `ts_rank`. Um
trecho só entra se os termos casados cobrirem uma fração mínima do peso da pergunta;
termo que não existe no corpus conta com o peso máximo. É isso que cala a perna
textual quando a pergunta está em português e o corpus em inglês, em vez de deixá-la
devolver ruído. A fórmula e o limiar foram escolhidos medindo no conjunto ouro
(`bula-knowledge eval`).
"""

from __future__ import annotations

import time

import numpy as np

from bula_knowledge.db import Database
from bula_knowledge.embeddings import Embedder
from bula_knowledge.schemas import Chunk, SearchMode

_SQL = """
with scope as (
  select id from documents
  where %(sources)s::text[] is null or source_id = any(%(sources)s::text[])
),
corpus as (
  select greatest(count(*), 1)::float8 as size from chunks
),
terms as (
  select t.lexeme,
         s.ndoc is not null as known,
         ln(1 + (corpus.size - coalesce(s.ndoc, 0) + 0.5) / (coalesce(s.ndoc, 0) + 0.5)) as idf
  from unnest(tsvector_to_array(to_tsvector('english', %(text)s))) as t(lexeme)
  left join lexeme_stats s on s.lexeme = t.lexeme
  cross join corpus
),
question as (
  select coalesce(sum(idf), 0) as weight from terms
),
vec as (
  select c.id,
         c.embedding <=> %(embedding)s::vector as distance,
         row_number() over (order by c.embedding <=> %(embedding)s::vector) as rank
  from chunks c
  where %(use_vector)s and c.document_id in (select id from scope)
  order by c.embedding <=> %(embedding)s::vector
  limit %(candidates)s
),
matched as (
  select c.id,
         sum(t.idf) as weight,
         sum(t.idf * (1 + ts_rank(c.tsv, quote_literal(t.lexeme)::tsquery, 1))) as score
  from terms t
  join chunks c on c.tsv @@ quote_literal(t.lexeme)::tsquery
  where %(use_lexical)s and t.known and c.document_id in (select id from scope)
  group by c.id
),
lex as (
  select m.id, row_number() over (order by m.score desc, m.id) as rank
  from matched m, question
  where m.weight >= %(min_coverage)s * question.weight
  order by m.score desc, m.id
  limit %(candidates)s
),
fused as (
  select id,
         sum(1.0 / (%(rrf_k)s + rank)) as score,
         max(rank) filter (where kind = 'vec') as vector_rank,
         max(rank) filter (where kind = 'lex') as lexical_rank,
         min(distance) as vector_distance
  from (
    select id, rank, distance, 'vec' as kind from vec
    union all
    select id, rank, null::float8, 'lex' as kind from lex
  ) ranked
  group by id
)
select c.id, d.source_id, d.title, c.heading_path,
       d.url || coalesce('#' || c.anchor, '') as url,
       c.content, f.score::float8, f.vector_rank, f.lexical_rank, f.vector_distance::float8
from fused f
join chunks c on c.id = f.id
join documents d on d.id = c.document_id
order by f.score desc, c.id
limit %(k)s
"""


class Searcher:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        *,
        candidates: int,
        rrf_k: int,
        min_coverage: float = 0.35,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._min_coverage = min_coverage

    @property
    def embedding_model(self) -> str:
        return self._embedder.name

    def search(
        self,
        query: str,
        *,
        k: int,
        mode: SearchMode = "hybrid",
        sources: list[str] | None = None,
    ) -> tuple[list[Chunk], float]:
        started = time.perf_counter()
        params = {
            "embedding": np.asarray(self._embedder.embed_query(query), dtype=np.float32),
            "text": query,
            "sources": sources or None,
            "use_vector": mode in ("hybrid", "vector"),
            "use_lexical": mode in ("hybrid", "lexical"),
            "candidates": self._candidates,
            "rrf_k": self._rrf_k,
            "min_coverage": self._min_coverage,
            "k": k,
        }
        with self._db.pool.connection() as conn:
            rows = conn.execute(_SQL, params).fetchall()
        chunks = [
            Chunk(
                id=row[0],
                source=row[1],
                title=row[2],
                heading=row[3],
                url=row[4],
                content=row[5],
                score=row[6],
                vector_rank=row[7],
                lexical_rank=row[8],
                vector_distance=row[9],
            )
            for row in rows
        ]
        return chunks, round((time.perf_counter() - started) * 1000, 1)

    def count_chunks(self) -> int:
        with self._db.pool.connection() as conn:
            row = conn.execute("select count(*) from chunks").fetchone()
        return int(row[0]) if row else 0
