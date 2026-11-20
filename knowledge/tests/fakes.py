"""Dublês de teste: embedding determinístico e LLM roteirizado. Ficam fora de `src`."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from bula_knowledge.llm import LlmUsage

_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Saco de palavras projetado por hash: textos com palavras em comum ficam próximos."""

    def __init__(self, dim: int = 384) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ScriptedLlm:
    """Responde cada pedido estruturado com a função registrada para aquele schema."""

    def __init__(self, handlers: dict[str, Callable[[str], Any]]) -> None:
        self._handlers = handlers
        self.calls: list[tuple[str, str]] = []

    def structured(self, schema: type, *, system: str, user: str) -> tuple[Any, LlmUsage]:
        name = schema.__name__
        self.calls.append((name, user))
        payload = self._handlers[name](user)
        return schema.model_validate(payload), LlmUsage(input_tokens=10, output_tokens=5)
