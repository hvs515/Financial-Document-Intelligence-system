"""Wires ingestion, retrieval, and the LLM together. Models load only when first needed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import chunk_document
from .config import Settings
from .embedding import Embedder, SentenceTransformerEmbedder
from .ingest import iter_paths, load_document
from .llm import LLM, build_llm
from .models import Answer, Hit
from .qa import DEFAULT_FIELDS, answer_question, extract_field
from .stores import VectorStore, open_store


@dataclass
class IngestReport:
    source: str
    chunks: int
    tables: int
    ocr_chunks: int
    skipped_pages: list[int]


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._embedder = embedder
        self._llm = llm
        self._store: VectorStore | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.settings.embed_model)
        return self._embedder

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = open_store(self.settings, self.embedder.dim)
        return self._store

    def ingest(self, paths: list[Path]) -> list[IngestReport]:
        reports = []
        for path in iter_paths(paths):
            document = load_document(path, self.settings)
            chunks = chunk_document(
                document, self.settings.chunk_chars, self.settings.chunk_overlap
            )
            vectors = (
                self.embedder.embed_documents([chunk.embedding_text() for chunk in chunks])
                if chunks
                else np.empty((0, self.embedder.dim), dtype=np.float32)
            )
            self.store.replace_source(document.source, chunks, vectors)
            reports.append(
                IngestReport(
                    source=document.source,
                    chunks=len(chunks),
                    tables=sum(chunk.kind == "table" for chunk in chunks),
                    ocr_chunks=sum(chunk.ocr for chunk in chunks),
                    skipped_pages=document.skipped_pages,
                )
            )
        return reports

    def search(self, query: str, k: int | None = None, source: str | None = None) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, k or self.settings.top_k, source)

    def ask(self, question: str, k: int | None = None, source: str | None = None) -> Answer:
        hits = self.search(question, k, source)
        return answer_question(self.llm, question, hits, self.settings.max_new_tokens)

    def extract(self, source: str, fields: list[str] | None = None) -> list[dict]:
        return [
            extract_field(self.llm, name, self.search(name, source=source))
            for name in fields or DEFAULT_FIELDS
        ]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
