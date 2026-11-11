"""Split Documents into Chunks that stay on one page, so every citation names an exact page."""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document, Element

SECTION_RE = re.compile(
    r"^\s*("
    r"(?:part\s+[ivx]+\b.*)|(?:item\s+\d+[a-c]?\b.*)|(?:note\s+\d+\b.*)"
    r"|(?:(?:consolidated\s+)?(?:balance sheets?|statements? of (?:operations|income|cash flows?|"
    r"comprehensive income|financial position|stockholders.? equity|shareholders.? equity)))"
    r"|management.s discussion and analysis.*|risk factors"
    r")\s*$",
    re.IGNORECASE,
)


def _section_of(text: str) -> str | None:
    """Return the last heading in a text element. Long lines are prose, not headings."""
    heading = None
    for line in text.splitlines():
        if len(line) <= 90 and SECTION_RE.match(line):
            heading = line.strip()
    return heading


def _split_prose(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a sentence end, then whitespace, so a figure is never cut in half.
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut < size // 2:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = end - overlap
        # Begin the overlap on a word boundary.
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start if next_start > start else end
    return [piece for piece in pieces if piece]


def _split_table(text: str, size: int) -> list[str]:
    """Keep tables whole when they fit; otherwise split by row and repeat the header on each part."""
    if len(text) <= size:
        return [text]
    lines = text.splitlines()
    is_markdown = len(lines) > 1 and set(lines[1]) <= set("|-: ")
    header = lines[:2] if is_markdown else []
    rows = lines[2:] if is_markdown else lines
    parts: list[str] = []
    current: list[str] = []
    length = sum(len(line) + 1 for line in header)
    for row in rows:
        if current and length + len(row) + 1 > size:
            parts.append("\n".join(header + current))
            current, length = [], sum(len(line) + 1 for line in header)
        current.append(row)
        length += len(row) + 1
    if current:
        parts.append("\n".join(header + current))
    return parts


def _make_chunk(doc: Document, index: int, text: str, element: Element, section: str) -> Chunk:
    digest = hashlib.sha1(f"{doc.source}:{index}:{text}".encode()).hexdigest()[:16]
    return Chunk(
        id=digest,
        text=text,
        source=doc.source,
        page=element.page,
        kind=element.kind,
        section=section,
        ocr=element.ocr,
    )


def chunk_document(doc: Document, size: int = 1200, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    pending: list[Element] = []

    def flush() -> None:
        if not pending:
            return
        merged = "\n".join(element.text for element in pending)
        if merged.strip() == section:
            # A bare heading is already carried as metadata, and alone it would crowd out real hits.
            pending.clear()
            return
        for piece in _split_prose(merged, size, overlap):
            chunks.append(_make_chunk(doc, len(chunks), piece, pending[0], section))
        pending.clear()

    for element in doc.elements:
        if element.kind == "table":
            flush()
            for piece in _split_table(element.text, size):
                chunks.append(_make_chunk(doc, len(chunks), piece, element, section))
            continue
        heading = _section_of(element.text)
        if heading:
            flush()
            section = heading
        if pending and (
            pending[0].page != element.page
            or sum(len(item.text) + 1 for item in pending) + len(element.text) > size
        ):
            flush()
        pending.append(element)
    flush()
    return chunks
