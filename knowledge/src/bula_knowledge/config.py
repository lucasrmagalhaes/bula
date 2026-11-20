"""Configuração do serviço, lida do ambiente e validada na subida.

O `.env` fica na raiz do repositório e é compartilhado com o serviço Node.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

LlmProvider = Literal["anthropic", "openai", "google", "offline"]

# Só a Anthropic tem modelo padrão. Para os outros provedores o nome do modelo
# é obrigatório, porque não há um padrão que este projeto tenha verificado.
DEFAULT_MODELS: dict[str, str] = {"anthropic": "claude-opus-5-5"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Banco. Sem DATABASE_URL, sobe um Postgres embutido em DATA_DIR/pg.
    database_url: str | None = None
    data_dir: Path = REPO_ROOT / "data"

    # Embeddings locais (ONNX, sem chave de API).
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings_dim: int = Field(384, gt=0)

    # LLM do workflow de RAG. "offline" pula a reescrita e a triagem.
    llm_provider: LlmProvider = "offline"
    llm_model: str | None = None
    llm_fast_model: str | None = None
    llm_max_retries: int = Field(2, ge=0, le=5)
    llm_timeout_ms: int = Field(60_000, gt=0)
    llm_max_tokens: int = Field(16_000, gt=0)
    # Esforço das etapas de reescrita e triagem (Anthropic). Vazio usa o padrão do modelo.
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "low"
    anthropic_api_key: SecretStr | None = None
    anthropic_api_url: str = "https://api.anthropic.com"
    # Reencaminha recusas para outro modelo, no servidor (beta da Anthropic).
    anthropic_fallbacks: bool = True
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # Busca e workflow.
    search_k: int = Field(6, ge=1, le=20)
    search_candidates: int = Field(30, ge=5, le=200)
    rrf_k: int = Field(60, ge=1)
    # Fração mínima do peso (IDF) da pergunta que um trecho precisa casar na busca textual.
    lexical_min_coverage: float = Field(0.35, ge=0, le=1)
    rag_max_attempts: int = Field(2, ge=1, le=4)
    rag_passage_chars: int = Field(900, ge=200)
    # Sem LLM não há triagem, então o corte é por distância: trecho sem termo em comum e
    # mais distante que isto não entra. Calibrado no conjunto ouro (perguntas do corpus
    # ficam em 0,34 de mediana e 0,50 no percentil 90; fora do corpus, de 0,57 para cima).
    offline_max_distance: float = Field(0.6, ge=0, le=2)

    # API.
    knowledge_host: str = "127.0.0.1"
    knowledge_port: int = Field(8100, ge=1, le=65535)
    knowledge_api_token: SecretStr | None = None
    max_question_chars: int = Field(2_000, ge=50)

    # MLflow. Sem MLFLOW_TRACKING_URI, usa um SQLite em DATA_DIR.
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "bula"
    mlflow_tracing: bool = False

    # Opcional: aumenta o limite da API do GitHub na ingestão.
    github_token: SecretStr | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _empty_as_unset(cls, value: object) -> object:
        # `CHAVE=` no .env chega como string vazia; trata como ausente.
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _check_llm(self) -> Settings:
        if self.llm_provider == "offline":
            return self
        if self.llm_model is None:
            default = DEFAULT_MODELS.get(self.llm_provider)
            if default is None:
                raise ValueError(f"LLM_MODEL é obrigatório com LLM_PROVIDER={self.llm_provider}")
            self.llm_model = default
        key = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
        }[self.llm_provider]
        if key is None:
            name = f"{self.llm_provider.upper()}_API_KEY"
            raise ValueError(f"{name} é obrigatória com LLM_PROVIDER={self.llm_provider}")
        return self

    @property
    def effective_mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{(self.data_dir / 'mlflow.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
