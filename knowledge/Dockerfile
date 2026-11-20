# Serviço de conhecimento. Em contêiner o banco é sempre externo (DATABASE_URL):
# o extra "embedded" não entra na imagem.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
WORKDIR /app/knowledge
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra openai --extra google --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra openai --extra google

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 bula
# O layout /app/knowledge espelha o repositório: a configuração resolve a raiz a partir daqui.
WORKDIR /app/knowledge
COPY --from=build --chown=bula:bula /app/knowledge /app/knowledge
RUN mkdir -p /app/data && chown bula:bula /app/data
USER bula
ENV PATH="/app/knowledge/.venv/bin:$PATH" \
    DATA_DIR=/app/data \
    KNOWLEDGE_HOST=0.0.0.0 \
    KNOWLEDGE_PORT=8100
EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=2)"
CMD ["bula-knowledge", "serve"]
