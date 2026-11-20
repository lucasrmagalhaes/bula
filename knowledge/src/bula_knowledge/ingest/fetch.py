"""Baixa os arquivos de documentação de uma fonte, do GitHub ou de uma pasta local."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from bula_knowledge.ingest.sources import Source

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
MAX_FILE_BYTES = 400_000


@dataclass(frozen=True)
class RawDocument:
    path: str
    text: str


class FetchError(RuntimeError):
    pass


def fetch_github(source: Source, *, token: str | None = None) -> tuple[str, list[RawDocument]]:
    """Devolve o commit da tag e os documentos. Usa duas chamadas à API do GitHub por fonte;
    o conteúdo vem do raw.githubusercontent.com, que não consome o limite da API."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "bula-knowledge"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        commit = _get_json(client, f"{API}/repos/{source.repo}/commits/{source.ref}")
        commit_sha = str(commit["sha"])
        tree = _get_json(client, f"{API}/repos/{source.repo}/git/trees/{commit_sha}?recursive=1")
        if tree.get("truncated"):
            raise FetchError(f"{source.repo}: a árvore veio truncada pela API do GitHub")
        paths = sorted(
            entry["path"]
            for entry in tree["tree"]
            if entry["type"] == "blob"
            and source.matches(entry["path"])
            and entry.get("size", 0) <= MAX_FILE_BYTES
        )
        if not paths:
            raise FetchError(f"{source.repo}@{source.ref}: nenhum arquivo em {source.prefix}")

        def download(path: str) -> RawDocument:
            response = client.get(f"{RAW}/{source.repo}/{commit_sha}/{path}")
            response.raise_for_status()
            return RawDocument(path=path, text=response.text)

        with ThreadPoolExecutor(max_workers=8) as pool:
            documents = list(pool.map(download, paths))
    return commit_sha, documents


def load_local(directory: Path, suffixes: tuple[str, ...] = (".md", ".mdx")) -> list[RawDocument]:
    """Lê uma pasta local de Markdown: seus próprios documentos ou as fixtures de teste."""
    if not directory.is_dir():
        raise FetchError(f"{directory} não é uma pasta")
    return [
        RawDocument(path=path.relative_to(directory).as_posix(), text=path.read_text("utf-8"))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]


def _get_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
        raise FetchError(
            "Limite da API do GitHub atingido. Defina GITHUB_TOKEN ou tente de novo mais tarde."
        )
    if response.status_code == 404:
        raise FetchError(f"Não encontrado no GitHub: {url}")
    response.raise_for_status()
    return response.json()
