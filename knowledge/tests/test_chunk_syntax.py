"""Sintaxe que não é conteúdo: comentários de MDX e marcação dentro de títulos.

Os dois casos vieram do índice real. No Zod, um comentário `{/* ... */}` de várias
linhas escondia uma seção antiga e ela era indexada, com o fecho `*/}` no título. No
Fastify, títulos da referência de TypeScript traziam link por referência e `\\<` escapado.
"""

from bula_knowledge.ingest.chunk import parse_document


def _parse(text: str):
    return parse_document(text, fallback_title="x").chunks


def test_single_line_mdx_comment_is_removed() -> None:
    chunks = _parse(
        "# T\n\nAntes. {/* nota interna */} Depois.\n\n{/* linha só de comentário */}\n\nFim."
    )
    assert [chunk.content for chunk in chunks] == ["Antes.  Depois.\n\nFim."]


def test_commented_out_section_is_not_indexed() -> None:
    text = (
        "# API\n\n## .refine()\n\nVersão atual.\n\n"
        "{/* ## Refinements antigos\n\nTexto antigo.\n\n### `.refine()` */}\n\n"
        "## .superRefine()\n\nOutra seção."
    )
    chunks = _parse(text)
    assert [chunk.heading_path for chunk in chunks] == ["API › .refine()", "API › .superRefine()"]
    assert all("antigo" not in chunk.content for chunk in chunks)


def test_comment_syntax_inside_a_code_fence_is_kept() -> None:
    text = "# T\n\n```tsx\nreturn <div>{/* comentário em JSX */}</div>;\n```\n"
    assert "{/* comentário em JSX */}" in _parse(text)[0].content


def test_heading_with_reference_links_and_escaped_generics_is_readable() -> None:
    heading = (
        "##### fastify.onCloseHookHandler\\<[RawServer][RawServerGeneric]\\>"
        "(instance: [FastifyInstance][FastifyInstance]): Promise\\<unknown\\> | void"
    )
    chunks = _parse(f"# TypeScript\n\n{heading}\n\nCorpo.")
    assert chunks[0].heading_path == (
        "TypeScript › fastify.onCloseHookHandler<RawServer>"
        "(instance: FastifyInstance): Promise<unknown> | void"
    )


def test_real_html_tag_in_heading_is_still_dropped() -> None:
    chunks = _parse('# Guia\n\n## Instalação <a name="install"></a>\n\nCorpo.')
    assert chunks[0].heading_path == "Guia › Instalação"
