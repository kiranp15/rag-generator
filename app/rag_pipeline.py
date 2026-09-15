"""Orchestrates the end-to-end flow: ingest documents into a collection,
and answer questions grounded in a collection.

This is the module both the FastAPI app and the CLI call into, so the two
entry points never duplicate pipeline logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.chunker import chunk_text
from app.config import settings
from app.document_loader import load_document
from app.generator import GroundedAnswer, generate_answer
from app.retriever import retrieve
from app.vector_store import CollectionNotFoundError, VectorStore


@dataclass
class IngestResult:
    collection_id: str
    files_ingested: List[str]
    chunks_added: int
    total_chunks: int


class RagPipeline:
    """Holds an in-memory cache of loaded VectorStores, backed by disk."""

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir or settings.data_dir
        self._cache: Dict[str, VectorStore] = {}

    # ------------------------------------------------------------ helpers
    def _get_or_create_store(self, collection_id: str) -> VectorStore:
        if collection_id in self._cache:
            return self._cache[collection_id]
        if VectorStore.exists(collection_id, self.data_dir):
            store = VectorStore.load(collection_id, self.data_dir)
        else:
            store = VectorStore(collection_id, self.data_dir)
        self._cache[collection_id] = store
        return store

    # ------------------------------------------------------------- public
    def create_collection(self, collection_id: str) -> None:
        if VectorStore.exists(collection_id, self.data_dir):
            raise ValueError(f"Collection '{collection_id}' already exists.")
        store = VectorStore(collection_id, self.data_dir)
        store.save()
        self._cache[collection_id] = store

    def list_collections(self) -> List[str]:
        return VectorStore.list_collections(self.data_dir)

    def delete_collection(self, collection_id: str) -> None:
        self._cache.pop(collection_id, None)
        VectorStore.delete(collection_id, self.data_dir)

    def ingest(self, collection_id: str, files: Dict[str, bytes]) -> IngestResult:
        """files: mapping of filename -> raw bytes."""
        store = self._get_or_create_store(collection_id)

        new_chunks = []
        ingested_names = []
        for filename, raw in files.items():
            text = load_document(filename, raw)
            if not text:
                continue
            file_chunks = chunk_text(
                text,
                source=filename,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            new_chunks.extend(file_chunks)
            ingested_names.append(filename)

        if store.chunks:
            store.add_documents(new_chunks)
        else:
            store.build(new_chunks)
        store.save()

        return IngestResult(
            collection_id=collection_id,
            files_ingested=ingested_names,
            chunks_added=len(new_chunks),
            total_chunks=len(store.chunks),
        )

    def query(self, collection_id: str, question: str, top_k: int | None = None) -> GroundedAnswer:
        if not VectorStore.exists(collection_id, self.data_dir) and collection_id not in self._cache:
            raise CollectionNotFoundError(f"Collection '{collection_id}' does not exist.")
        store = self._get_or_create_store(collection_id)
        chunks = retrieve(store, question, top_k=top_k or settings.top_k)
        return generate_answer(question, chunks)
