"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Element:
    """A unit of parsed content: a run of prose or one whole table."""

    kind: str  # text | table
    text: str
    page: int
    ocr: bool = False


@dataclass
class Document:
    source: str
    elements: list[Element] = field(default_factory=list)
    # Pages that looked scanned but could not be read. Reported, never dropped silently.
    skipped_pages: list[int] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    kind: str = "text"
    section: str = ""
    ocr: bool = False

    def embedding_text(self) -> str:
        """Text to embed: the section title gives a bare table row its meaning."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    hits: list[Hit]  # everything the model was shown, in citation order
    citations: list[int]  # 1-based indices into hits that the answer cites
    unverified_numbers: list[str]  # numbers in the answer that no excerpt contains
    invalid_citations: list[int] = field(default_factory=list)
