"""FastAPI entrypoint for the RAG Generator.

Endpoints let a client create a collection (document set), upload files into
it at runtime, and ask grounded questions against it — all without any code
change per document set.
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.document_loader import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError
from app.rag_pipeline import RagPipeline
from app.schemas import (
    CollectionListResponse,
    CreateCollectionRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.vector_store import CollectionNotFoundError

app = FastAPI(
    title="RAG Generator",
    description="Runtime-configurable RAG service: create a document collection, "
    "ingest files into it, and ask grounded questions — no code changes per document set.",
    version="1.0.0",
)

pipeline = RagPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/collections", response_model=CollectionListResponse)
def list_collections():
    return CollectionListResponse(collections=pipeline.list_collections())


@app.post("/collections", status_code=201)
def create_collection(req: CreateCollectionRequest):
    try:
        pipeline.create_collection(req.collection_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"collection_id": req.collection_id, "status": "created"}


@app.delete("/collections/{collection_id}")
def delete_collection(collection_id: str):
    pipeline.delete_collection(collection_id)
    return {"collection_id": collection_id, "status": "deleted"}


@app.post("/collections/{collection_id}/documents", response_model=IngestResponse)
async def ingest_documents(collection_id: str, files: list[UploadFile] = File(...)):
    file_bytes = {}
    for f in files:
        content = await f.read()
        file_bytes[f.filename] = content

    try:
        result = pipeline.ingest(collection_id, file_bytes)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(
        collection_id=result.collection_id,
        files_ingested=result.files_ingested,
        chunks_added=result.chunks_added,
        total_chunks=result.total_chunks,
    )


@app.post("/collections/{collection_id}/query", response_model=QueryResponse)
def query_collection(collection_id: str, req: QueryRequest):
    try:
        result = pipeline.query(collection_id, req.question, top_k=req.top_k)
    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # e.g. missing ANTHROPIC_API_KEY
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result.answer,
        used_sources=result.used_sources,
        chunks=[
            SourceChunk(
                source=c.source, chunk_index=c.chunk_index, score=c.score, text=c.text
            )
            for c in result.raw_chunks
        ],
    )


@app.get("/supported-file-types")
def supported_file_types():
    return {"extensions": list(SUPPORTED_EXTENSIONS)}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
