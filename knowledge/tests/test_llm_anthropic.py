"""Fixa o que este serviço manda para a API da Anthropic.

Um servidor HTTP local faz o papel da API e guarda o pedido recebido. Assim o teste
passa pelo adaptador e pelo SDK de verdade, sem chave e sem rede externa.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from bula_knowledge.config import Settings
from bula_knowledge.llm import FALLBACK_BETA, LlmRefusalError, build_structured_llm


class Plan(BaseModel):
    queries: list[str]


class _FakeAnthropic:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply: dict[str, Any] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(
                    {"path": self.path, "headers": dict(self.headers.items()), "body": body}
                )
                payload = json.dumps(outer.reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _message(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        # Como o modelo real com raciocínio adaptativo: um bloco de raciocínio antes do texto.
        "content": [
            {"type": "thinking", "thinking": "Penso nas consultas.", "signature": "sig-abc"},
            {"type": "text", "text": text},
        ]
        if text
        else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 41, "output_tokens": 7},
    }


@pytest.fixture
def api() -> Iterator[_FakeAnthropic]:
    fake = _FakeAnthropic()
    yield fake
    fake.server.shutdown()


def _settings(api: _FakeAnthropic, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key=SecretStr("chave-de-teste"),
        anthropic_api_url=api.url,
        llm_max_retries=0,
        **overrides,
    )


def test_request_uses_native_structured_output_and_no_forbidden_params(
    api: _FakeAnthropic, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Um ANTHROPIC_BASE_URL solto no ambiente não pode desviar a chamada.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/nao-usar")
    api.reply = _message('{"queries": ["fastify hooks"]}')
    llm = build_structured_llm(_settings(api))
    assert llm is not None

    plan, usage = llm.structured(Plan, system="sys", user="pergunta")

    assert plan.queries == ["fastify hooks"]
    assert (usage.input_tokens, usage.output_tokens) == (41, 7)
    (request,) = api.requests
    body = request["body"]
    assert request["path"].startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "sys"
    assert body["max_tokens"] == 16_000
    # Recusados pelos modelos atuais: amostragem e tool_choice forçado.
    assert not {"temperature", "top_p", "top_k", "tool_choice", "tools"} & set(body)
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "queries" in body["output_config"]["format"]["schema"]["properties"]
    assert body["thinking"]["type"] == "adaptive"
    # Fallback de recusa do lado do servidor, ligado por padrão.
    assert body["fallbacks"] == "default"
    headers = {name.lower(): value for name, value in request["headers"].items()}
    assert FALLBACK_BETA in headers["anthropic-beta"]
    # O SDK 2.x manda a chave de API como Bearer.
    assert headers["authorization"] == "Bearer chave-de-teste"


def test_fallbacks_can_be_turned_off_and_skip_models_without_support(
    api: _FakeAnthropic,
) -> None:
    api.reply = _message('{"queries": ["x"]}')
    for overrides in ({"anthropic_fallbacks": False}, {"llm_fast_model": "claude-haiku-5-5"}):
        api.requests.clear()
        llm = build_structured_llm(_settings(api, **overrides))
        assert llm is not None
        llm.structured(Plan, system="s", user="u")
        assert "fallbacks" not in api.requests[0]["body"]


def test_refusal_becomes_a_typed_error(api: _FakeAnthropic) -> None:
    api.reply = _message("", stop_reason="refusal")
    llm = build_structured_llm(_settings(api))
    assert llm is not None
    with pytest.raises(LlmRefusalError):
        llm.structured(Plan, system="s", user="u")


def test_offline_provider_has_no_llm() -> None:
    assert build_structured_llm(Settings(_env_file=None, llm_provider="offline")) is None


def test_missing_key_or_model_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Settings(_env_file=None, llm_provider="anthropic")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        Settings(_env_file=None, llm_provider="openai", openai_api_key=SecretStr("k"))
