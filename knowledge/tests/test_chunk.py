from bula_knowledge.ingest.chunk import full_heading, parse_document, slugify


def test_title_comes_from_front_matter() -> None:
    parsed = parse_document("---\ntitle: Guia\n---\n\n# Outro\n\nTexto.", fallback_title="x")
    assert parsed.title == "Guia"


def test_title_falls_back_to_first_heading_then_to_file_name() -> None:
    assert parse_document("# Hooks\n\nTexto.", fallback_title="x").title == "Hooks"
    assert parse_document("Só texto.", fallback_title="Arquivo").title == "Arquivo"


def test_heading_path_follows_the_document_hierarchy() -> None:
    text = "# A\n\nintro\n\n## B\n\ncorpo b\n\n### C\n\ncorpo c\n\n## D\n\ncorpo d"
    paths = [chunk.heading_path for chunk in parse_document(text, fallback_title="x").chunks]
    assert paths == ["A", "A › B", "A › B › C", "A › D"]


def test_heading_inside_code_fence_is_not_a_section() -> None:
    text = "# Uso\n\n```bash\n# isto é comentário\nnpm install\n```\n"
    chunks = parse_document(text, fallback_title="x").chunks
    assert len(chunks) == 1
    assert "# isto é comentário" in chunks[0].content


def test_code_fence_is_never_split_from_its_closing() -> None:
    code = "```ts\n" + "\n".join(f"const v{i} = {i};" for i in range(40)) + "\n```"
    text = "# T\n\n" + ("parágrafo. " * 60) + "\n\n" + code
    chunks = parse_document(text, fallback_title="x", max_chars=900).chunks
    with_code = [chunk for chunk in chunks if "```ts" in chunk.content]
    assert len(with_code) == 1
    assert with_code[0].content.count("```") == 2


def test_oversized_block_is_split_by_lines_within_the_limit() -> None:
    text = "# T\n\n" + "\n".join(f"linha {i} " + "x" * 50 for i in range(100))
    chunks = parse_document(text, fallback_title="x", max_chars=500).chunks
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 500 for chunk in chunks)


def test_mdx_imports_and_component_tags_are_dropped_but_text_is_kept() -> None:
    text = (
        'import { Callout } from "ui";\n\n# Erros\n\n<Callout type="warn">\nAtenção.\n</Callout>\n'
    )
    chunks = parse_document(text, fallback_title="x").chunks
    assert [chunk.content for chunk in chunks] == ["Atenção."]


def test_anchor_matches_github_and_repeated_headings_get_a_suffix() -> None:
    text = "# API\n\n## The `retryLimit` option\n\na\n\n## Notes\n\nb\n\n## Notes\n\nc"
    anchors = [chunk.anchor for chunk in parse_document(text, fallback_title="x").chunks]
    assert anchors == ["the-retrylimit-option", "notes", "notes-1"]
    assert slugify("What's new?") == "whats-new"


def test_heading_without_body_produces_no_chunk() -> None:
    assert parse_document("# Só título\n", fallback_title="x").chunks == []


def test_decorative_html_h1_is_dropped() -> None:
    text = '<h1 align="center">Fastify</h1>\n\n## Hooks\n\nTexto.'
    parsed = parse_document(text, fallback_title="x")
    assert parsed.title == "Hooks"
    assert [chunk.content for chunk in parsed.chunks] == ["Texto."]


def test_full_heading_does_not_repeat_the_title() -> None:
    assert full_heading("Hooks", "Hooks › onRequest") == "Hooks › onRequest"
    assert full_heading("Hooks", "Hooks") == "Hooks"
    assert full_heading("Guia", "Instalação") == "Guia › Instalação"
