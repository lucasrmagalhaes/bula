"""Catálogo das fontes do corpus.

Nenhuma documentação de terceiros é versionada neste repositório. A ingestão
baixa cada fonte do GitHub numa tag fixa e grava a tag, o commit e a licença
junto do índice, para que toda citação aponte para uma versão conferível.
"""

from __future__ import annotations

from dataclasses import dataclass

LOCAL = "local"


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    repo: str
    ref: str
    license: str
    prefix: str
    suffixes: tuple[str, ...] = (".md", ".mdx")
    exclude: tuple[str, ...] = ()
    # Só para fonte local (`ingest --path`): onde os documentos são publicados, se forem.
    base_url: str | None = None

    def url_for(self, path: str, commit_sha: str | None) -> str:
        if self.repo == LOCAL:
            # Sem endereço público, a citação é o caminho do arquivo dentro da pasta.
            return f"{self.base_url.rstrip('/')}/{path}" if self.base_url else path
        return f"https://github.com/{self.repo}/blob/{commit_sha or self.ref}/{path}"

    def matches(self, path: str) -> bool:
        return (
            path.startswith(self.prefix)
            and path.endswith(self.suffixes)
            and not any(part in path for part in self.exclude)
        )


SOURCES: dict[str, Source] = {
    source.id: source
    for source in (
        Source(
            id="fastify",
            title="Fastify",
            repo="fastify/fastify",
            ref="v5.12.8",
            license="MIT",
            prefix="docs/",
        ),
        Source(
            id="zod",
            title="Zod",
            repo="colinhacks/zod",
            ref="v4.6.7",
            license="MIT",
            prefix="packages/docs/content/",
            suffixes=(".mdx",),
            exclude=("/blog/",),
        ),
    )
}
