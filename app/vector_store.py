"""Per-collection storage of chunks + their embeddings, with disk persistence.

Each collection is a self-contained folder under `data_dir/<collection_id>/`:
  - chunks.json      chunk text + source/index metadata
  - vectors.npy       embedding matrix (float32, one row per chunk)
  - embedder.bin       serialized fitted embedder (e.g. TF-IDF vocabulary)

Because everything is keyed by `collection_id` and created on demand, using
a brand-new document set never requires touching code — just a new id.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

from app.chunker import Chunk
from app.embeddings import Embedder, get_embedder


class CollectionNotFoundError(FileNotFoundError):
    pass


class VectorStore:
    def __init__(self, collection_id: str, data_dir: str, embedder: Embedder | None = None):
        self.collection_id = collection_id
        self.data_dir = data_dir
        self.embedder = embedder or get_embedder("tfidf")
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @property
    def _dir(self) -> str:
        return os.path.join(self.data_dir, self.collection_id)

    # ---------------------------------------------------------------- build
    def build(self, chunks: List[Chunk]) -> None:
        """(Re)fit the embedder and vectors from scratch for this collection."""
        self.chunks = chunks
        if not chunks:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            return
        texts = [c.text for c in chunks]
        self.vectors = self.embedder.fit(texts)

    def add_documents(self, new_chunks: List[Chunk]) -> None:
        """Append new chunks to an existing collection and refit.

        TF-IDF's vocabulary depends on the full corpus, so adding documents
        refits over `existing + new` chunks. This keeps retrieval quality
        consistent; for very large, frequently-updated collections a dense
        embedder (which doesn't need a global refit) would scale better —
        see app/embeddings.py.
        """
        combined = self.chunks + new_chunks
        # Re-index chunk.index within each source consistently.
        self.build(combined)

    # -------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vec = self.embedder.embed_query(query)
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        norms[norms == 0] = 1e-10
        scores = (self.vectors @ query_vec) / norms
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    # ----------------------------------------------------------- persist
    def save(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        with open(os.path.join(self._dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)
        np.save(os.path.join(self._dir, "vectors.npy"), self.vectors)
        with open(os.path.join(self._dir, "embedder.bin"), "wb") as f:
            f.write(self.embedder.to_bytes())

    @classmethod
    def load(cls, collection_id: str, data_dir: str) -> "VectorStore":
        directory = os.path.join(data_dir, collection_id)
        chunks_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(chunks_path):
            raise CollectionNotFoundError(f"Collection '{collection_id}' does not exist.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        chunks = [Chunk(**c) for c in raw_chunks]

        vectors = np.load(os.path.join(directory, "vectors.npy"))

        from app.embeddings import TfidfEmbedder

        with open(os.path.join(directory, "embedder.bin"), "rb") as f:
            embedder = TfidfEmbedder.from_bytes(f.read())

        store = cls(collection_id, data_dir, embedder=embedder)
        store.chunks = chunks
        store.vectors = vectors
        return store

    @staticmethod
    def exists(collection_id: str, data_dir: str) -> bool:
        return os.path.exists(os.path.join(data_dir, collection_id, "chunks.json"))

    @staticmethod
    def list_collections(data_dir: str) -> List[str]:
        if not os.path.isdir(data_dir):
            return []
        return sorted(
            name
            for name in os.listdir(data_dir)
            if os.path.exists(os.path.join(data_dir, name, "chunks.json"))
        )

    @staticmethod
    def delete(collection_id: str, data_dir: str) -> None:
        import shutil

        directory = os.path.join(data_dir, collection_id)
        if os.path.exists(directory):
            shutil.rmtree(directory)
