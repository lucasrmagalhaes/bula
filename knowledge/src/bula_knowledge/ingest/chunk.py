"""Divide Markdown e MDX em trechos que respeitam a estrutura do documento.

Regras:
- cada trecho pertence a uma seção e leva o caminho dos títulos, que entra no
  texto vetorizado e na citação;
- bloco de código cercado nunca é cortado ao meio, salvo se sozinho passar do
  limite;
- seção longa é quebrada em parágrafos, sem sobreposição: o título repetido em
  cada trecho já devolve o contexto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MDX_STATEMENT = re.compile(r"^(import|export)\s.+$")
_JSX_TAG_LINE = re.compile(r"^\s*</?[A-Z][\w.]*(\s[^>]*)?/?>\s*$")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Link por referência, `[texto][alvo]`: comum em título de referência de API.
_MD_REFERENCE_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# Tag HTML de verdade. `\<` escapado é texto (um tipo genérico, por exemplo) e fica.
_HTML_TAG = re.compile(r"(?<!\\)<[^>]+>")
_ESCAPED = re.compile(r"\\([\\`*_{}\[\]()<>#+\-.!|])")
# Comentário de MDX, `{/* ... */}`. Pode ocupar várias linhas e esconder seções inteiras.
_MDX_COMMENT = re.compile(r"\{/\*.*?\*/\}")
# Cabeçalho decorativo em HTML, comum no topo de README e de docs: não é conteúdo.
_HTML_H1_LINE = re.compile(r"^\s*<h1[^>]*>.*</h1>\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkDraft:
    heading_path: str
    anchor: str | None
    content: str


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    chunks: list[ChunkDraft]


def parse_document(text: str, *, fallback_title: str, max_chars: int = 1600) -> ParsedDocument:
    meta, body = _split_front_matter(text)
    body = _strip_mdx(body)
    sections = _split_sections(body)
    title = meta.get("title") or _first_h1(sections) or fallback_title
    chunks: list[ChunkDraft] = []
    for path, anchor, lines in sections:
        heading_path = " › ".join(path) if path else title
        for piece in _pack(_blocks(lines), max_chars):
            chunks.append(ChunkDraft(heading_path=heading_path, anchor=anchor, content=piece))
    return ParsedDocument(title=title, chunks=chunks)


def full_heading(title: str, heading_path: str) -> str:
    """Título do documento mais o caminho da seção, sem repetir o título quando ele é o H1."""
    if heading_path == title or heading_path.startswith(f"{title} › "):
        return heading_path
    return f"{title} › {heading_path}"


def slugify(heading: str) -> str:
    """Âncora no formato do GitHub: minúsculas, sem pontuação, espaços viram hífens."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and not key.startswith((" ", "\t")):
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[match.end() :]


def _strip_mdx(body: str) -> str:
    """Remove o que é sintaxe de MDX e não conteúdo: `import`/`export`, linhas que só abrem
    ou fecham um componente JSX e comentários, inclusive os que escondem seções inteiras.
    Dentro de bloco de código nada é mexido."""
    kept: list[str] = []
    in_fence = False
    in_comment = False
    for raw_line in body.splitlines():
        line = raw_line
        if in_comment:
            end = line.find("*/}")
            if end < 0:
                continue
            in_comment = False
            line = line[end + 3 :]
            if not line.strip():
                continue
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            line = _MDX_COMMENT.sub("", line)
            start = line.find("{/*")
            if start >= 0:
                in_comment = True
                line = line[:start]
            if (
                (line != raw_line and not line.strip())
                or _MDX_STATEMENT.match(line)
                or _JSX_TAG_LINE.match(line)
                or _HTML_H1_LINE.match(line)
            ):
                continue
        kept.append(line)
    return "\n".join(kept)


def _clean_heading(raw: str) -> str:
    text = _MD_LINK.sub(r"\1", raw)
    text = _MD_REFERENCE_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = _ESCAPED.sub(r"\1", text)
    return text.replace("`", "").replace("*", "").strip()


def _split_sections(body: str) -> list[tuple[list[str], str | None, list[str]]]:
    """Devolve (caminho de títulos, âncora, linhas) de cada seção, na ordem do documento."""
    sections: list[tuple[list[str], str | None, list[str]]] = []
    stack: list[tuple[int, str]] = []
    anchor: str | None = None
    seen: dict[str, int] = {}
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in current):
            sections.append(([title for _, title in stack], anchor, list(current)))
        current.clear()

    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading is None:
            current.append(line)
            continue
        flush()
        level = len(heading.group(1))
        title = _clean_heading(heading.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        base = slugify(title)
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
    flush()
    return sections


def _first_h1(sections: list[tuple[list[str], str | None, list[str]]]) -> str | None:
    for path, _, _ in sections:
        if path:
            return path[0]
    return None


def _blocks(lines: list[str]) -> list[str]:
    """Parágrafos separados por linha em branco; um bloco de código é um bloco só."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(blocks: list[str], max_chars: int) -> list[str]:
    """Junta blocos vizinhos até o limite; só corta por linha o bloco que sozinho passa dele."""
    pieces: list[str] = []
    current = ""
    for block in blocks:
        for part in _split_oversized(block, max_chars):
            if current and len(current) + len(part) + 2 > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def _split_oversized(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in block.splitlines():
        line = line[:max_chars]
        if current and size + len(line) + 1 > max_chars:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts
