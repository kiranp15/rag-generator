from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    collection_id: str = Field(..., description="Unique, URL-safe id for this document set.")


class IngestResponse(BaseModel):
    collection_id: str
    files_ingested: List[str]
    chunks_added: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    used_sources: List[str]
    chunks: List[SourceChunk]


class CollectionListResponse(BaseModel):
    collections: List[str]
