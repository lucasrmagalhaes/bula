"""Única porta de saída para LLM neste serviço.

O workflow só precisa de uma operação: pedir um objeto tipado. Concentrar isso aqui
deixa num lugar só o que muda por provedor, e os testes trocam a porta inteira.

Notas da Anthropic, conferidas no `langchain-anthropic` instalado:
- a URL base é passada explícita, porque sem isso o adaptador herda
  `ANTHROPIC_BASE_URL` do ambiente e a chave iria para o host que estiver lá;
- nenhum parâmetro de amostragem é enviado: os modelos atuais recusam
  `temperature`, `top_p` e `top_k`;
- a saída estruturada usa `output_config.format` (method="json_schema"), porque o
  `tool_choice` forçado do método padrão é recusado no Claude Opus 5.5;
- a recusa do modelo chega como `stop_reason: "refusal"` com HTTP 200 e vira erro
  tipado aqui.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from bula_knowledge.config import Settings

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Modelos que aceitam o parâmetro `fallbacks` do lado do servidor.
_FALLBACK_MODELS = (
    "claude-fable-5-1",
    "claude-fable-5-2",
    "claude-opus-5",
    "claude-sonnet-5-5",
)


@dataclass(frozen=True)
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0


class LlmOutputError(RuntimeError):
    """O modelo respondeu, mas não no formato pedido."""


class LlmRefusalError(RuntimeError):
    """O modelo recusou o pedido (stop_reason "refusal")."""


class StructuredLlm(Protocol):
    def structured(self, schema: type[T], *, system: str, user: str) -> tuple[T, LlmUsage]: ...


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Modelo das etapas de reescrita e triagem: LLM_FAST_MODEL, ou LLM_MODEL se ausente."""
    model = settings.llm_fast_model or settings.llm_model
    assert model is not None  # garantido pela validação da configuração
    timeout_s = settings.llm_timeout_ms / 1000
    common: dict[str, Any] = {"max_retries": settings.llm_max_retries, "timeout": timeout_s}

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        assert settings.anthropic_api_key is not None
        params: dict[str, Any] = {
            "model": model,
            "api_key": settings.anthropic_api_key,
            "base_url": settings.anthropic_api_url,
            "max_tokens": settings.llm_max_tokens,
            **common,
        }
        if settings.llm_effort:
            params["effort"] = settings.llm_effort
        if settings.anthropic_fallbacks and model.startswith(_FALLBACK_MODELS):
            params["betas"] = [FALLBACK_BETA]
            params["model_kwargs"] = {"fallbacks": "default"}
        return ChatAnthropic(**params)

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=settings.openai_api_key, **common)

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, **common)

    raise ValueError(f"LLM_PROVIDER sem modelo de chat: {settings.llm_provider}")


class LangChainStructuredLlm:
    def __init__(self, model: BaseChatModel, *, method: str | None = None) -> None:
        self._model = model
        self._method = method
        self._runnables: dict[type, Any] = {}

    def structured(self, schema: type[T], *, system: str, user: str) -> tuple[T, LlmUsage]:
        runnable = self._runnables.get(schema)
        if runnable is None:
            kwargs: dict[str, Any] = {"include_raw": True}
            if self._method:
                kwargs["method"] = self._method
            runnable = self._model.with_structured_output(schema, **kwargs)
            self._runnables[schema] = runnable

        result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if not isinstance(result, dict):  # include_raw=True sempre devolve um dict
            raise LlmOutputError("resposta inesperada do adaptador")
        raw = result.get("raw")
        metadata = getattr(raw, "response_metadata", None) or {}
        if metadata.get("stop_reason") == "refusal":
            raise LlmRefusalError(str(metadata.get("stop_details") or "recusa do modelo"))
        parsed = result.get("parsed")
        if parsed is None:
            raise LlmOutputError(str(result.get("parsing_error") or "saída fora do schema"))
        usage = getattr(raw, "usage_metadata", None) or {}
        return parsed, LlmUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


def build_structured_llm(settings: Settings) -> StructuredLlm | None:
    if settings.llm_provider == "offline":
        return None
    method = "json_schema" if settings.llm_provider == "anthropic" else None
    return LangChainStructuredLlm(build_chat_model(settings), method=method)
