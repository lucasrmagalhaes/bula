"""Embeddings locais em ONNX (fastembed): sem chave de API e sem PyTorch."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Carrega o modelo na primeira chamada; o download fica em `cache_dir`."""

    def __init__(self, model_name: str, dim: int, cache_dir: Path) -> None:
        self.name = model_name
        self.dim = dim
        self._cache_dir = cache_dir
        self._model: Any = None
        # Perguntas repetidas não pagam a inferência de novo.
        self._query_cache = lru_cache(maxsize=2048)(self._embed_query_uncached)

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.name, cache_dir=str(self._cache_dir))
        return self._model

    def warm_up(self) -> None:
        vector = self.embed_query("warm up")
        if len(vector) != self.dim:
            raise RuntimeError(
                f"{self.name} gera vetores de {len(vector)} dimensões, "
                f"mas EMBEDDINGS_DIM é {self.dim}"
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._load().embed(list(texts), batch_size=32)]

    def _embed_query_uncached(self, text: str) -> tuple[float, ...]:
        return tuple(next(iter(self._load().embed([text]))).tolist())

    def embed_query(self, text: str) -> list[float]:
        return list(self._query_cache(text))

    def clear_cache(self) -> None:
        self._query_cache.cache_clear()
