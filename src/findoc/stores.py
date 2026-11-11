"""Vector stores. FAISS and Qdrant share one contract, and re-ingesting a source replaces it."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings
from .models import Chunk, Hit


class VectorStore(Protocol):
    def replace_source(self, source: str, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int, source: str | None = None) -> list[Hit]: ...

    def sources(self) -> dict[str, int]: ...

    def close(self) -> None: ...


class FaissStore:
    def __init__(self, directory: Path, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self._index_path = directory / "index.faiss"
        self._chunks_path = directory / "chunks.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[int, Chunk] = {}
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            if self._index.d != dim:
                raise ValueError(
                    f"Index in {directory} has dimension {self._index.d}, but the embedder "
                    f"produces {dim}. Use a new data directory or re-ingest."
                )
            with self._chunks_path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    self._chunks[record["vector_id"]] = Chunk(**record["chunk"])
        else:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def replace_source(self, source: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        stale = [vid for vid, chunk in self._chunks.items() if chunk.source == source]
        if stale:
            self._index.remove_ids(np.array(stale, dtype=np.int64))
            for vid in stale:
                del self._chunks[vid]
        if chunks:
            next_id = max(self._chunks, default=-1) + 1
            ids = np.arange(next_id, next_id + len(chunks), dtype=np.int64)
            self._index.add_with_ids(vectors, ids)
            self._chunks.update(zip(ids.tolist(), chunks))
        self._save()

    def _save(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        with self._chunks_path.open("w", encoding="utf-8") as handle:
            for vid, chunk in self._chunks.items():
                handle.write(json.dumps({"vector_id": vid, "chunk": chunk.to_dict()}) + "\n")

    def search(self, vector: np.ndarray, k: int, source: str | None = None) -> list[Hit]:
        total = self._index.ntotal
        if total == 0:
            return []
        # A flat index has no pre-filter, so a source filter scores everything and filters after.
        fetch = total if source else min(k, total)
        scores, ids = self._index.search(vector.reshape(1, -1), fetch)
        hits = []
        for score, vid in zip(scores[0], ids[0]):
            chunk = self._chunks.get(int(vid))
            if chunk and (source is None or chunk.source == source):
                hits.append(Hit(chunk, float(score)))
        return hits[:k]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self._chunks.values():
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return counts

    def close(self) -> None:
        pass


class QdrantStore:
    def __init__(self, dim: int, collection: str, url: str = "", path: Path | None = None) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("The Qdrant store needs: pip install 'findoc[qdrant]'") from exc

        self._models = models
        self._client = QdrantClient(url=url) if url else QdrantClient(path=str(path))
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def _source_filter(self, source: str):
        models = self._models
        return models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )

    def replace_source(self, source: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        models = self._models
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=self._source_filter(source)),
        )
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.id)),
                vector=vector.tolist(),
                payload=chunk.to_dict(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(self._collection, points=points)

    def search(self, vector: np.ndarray, k: int, source: str | None = None) -> list[Hit]:
        response = self._client.query_points(
            self._collection,
            query=vector.tolist(),
            limit=k,
            query_filter=self._source_filter(source) if source else None,
        )
        return [Hit(Chunk(**point.payload), float(point.score)) for point in response.points]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection, limit=512, offset=offset, with_payload=["source"]
            )
            for point in points:
                counts[point.payload["source"]] = counts.get(point.payload["source"], 0) + 1
            if offset is None:
                return counts

    def close(self) -> None:
        self._client.close()


def open_store(settings: Settings, dim: int) -> VectorStore:
    if settings.store == "faiss":
        return FaissStore(settings.data_dir / "faiss", dim)
    if settings.store == "qdrant":
        return QdrantStore(
            dim,
            settings.qdrant_collection,
            url=settings.qdrant_url,
            path=settings.data_dir / "qdrant",
        )
    raise ValueError(f"Unknown store: {settings.store}")
