"""Thin retrieval layer on top of a VectorStore.

Kept separate from VectorStore so retrieval-time concerns (top_k, minimum
score thresholds, result formatting) can evolve independently of storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


def retrieve(store: VectorStore, question: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievedChunk]:
    results = store.search(question, top_k=top_k)
    return [
        RetrievedChunk(text=chunk.text, source=chunk.source, chunk_index=chunk.index, score=score)
        for chunk, score in results
        if score >= min_score
    ]
