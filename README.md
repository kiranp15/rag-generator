# RAG Generator

A runtime-configurable Retrieval-Augmented Generation (RAG) service. Point it
at any set of documents — no code changes required — and it builds a
searchable index and answers questions grounded in that content, with
citations back to the source chunks.

## Why this design

- **Runtime document sets, not hardcoded ones.** Documents are organized into
  *collections*. A collection is created via an API call (or CLI command),
  documents are uploaded/ingested into it at runtime, and it's immediately
  queryable. Multiple collections can exist side by side (e.g. "hr-policies",
  "product-docs-v2"), each fully isolated.
- **Grounded, not hallucinated.** The generator is instructed to answer
  strictly from the retrieved chunks, to cite which source/chunk each claim
  came from, and to say it doesn't know rather than invent an answer when the
  retrieved context is insufficient.
- **Pluggable pieces.** Loading, chunking, embedding, storage, and generation
  are each isolated behind small interfaces (see `app/`) so any one of them
  (e.g. swapping the embedder for OpenAI/Cohere, or the store for a real
  vector DB like Chroma/FAISS/pgvector) can be replaced without touching the
  rest of the pipeline.
- **Works offline out of the box.** The default embedder is a TF-IDF /
  cosine-similarity retriever (scikit-learn), so ingestion and retrieval work
  with zero external API calls and zero GPU/model downloads. Only the final
  answer-generation step calls an LLM (Anthropic's Claude by default). This
  keeps the core retrieval logic fast, deterministic, and easy to grade/test.

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │                 FastAPI app                 │
                 │   /collections   /documents   /query         │
                 └───────────────┬───────────────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                       │                           │
┌───────▼────────┐   ┌──────────▼──────────┐   ┌────────────▼───────────┐
│ document_loader│   │       chunker        │   │      vector_store       │
│ pdf/docx/txt/md│──▶│  paragraph-aware,    │──▶│  per-collection, saved   │
│      /csv      │   │  overlapping windows │   │  to disk as .npz + json │
└─────────────────┘   └──────────────────────┘   └────────────┬───────────┘
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │      retriever       │
                                                     │ TF-IDF cosine top-k  │
                                                     └──────────┬──────────┘
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │      generator       │
                                                     │ Claude, grounded     │
                                                     │ answer + citations   │
                                                     └──────────────────────┘
```

Each collection is persisted under `data/<collection_id>/` as:
- `chunks.json` — chunk text + metadata (source file, chunk index)
- `vectors.npz` — TF-IDF matrix for that collection
- `vectorizer.pkl` — the fitted TF-IDF vectorizer for that collection

This means collections survive process restarts, and a new document set is
just a new folder — no code change needed.

## Setup

```bash
cd rag-generator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then, e.g.:

```bash
# 1. Create a collection
curl -X POST localhost:8000/collections -H "Content-Type: application/json" \
  -d '{"collection_id": "handbook"}'

# 2. Ingest documents into it (any mix of pdf/docx/txt/md/csv)
curl -X POST localhost:8000/collections/handbook/documents \
  -F "files=@sample_docs/employee_handbook.md" \
  -F "files=@sample_docs/expense_policy.txt"

# 3. Ask a grounded question
curl -X POST localhost:8000/collections/handbook/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do new employees get?"}'
```

Interactive API docs are auto-served at `localhost:8000/docs`.

## Running via CLI (no server needed)

```bash
python cli.py ingest --collection handbook --path sample_docs
python cli.py ask --collection handbook --question "What is the expense reimbursement limit?"
python cli.py list
```

## Running the tests

The core pipeline (loading, chunking, retrieval) has no external
dependencies and is covered by a small offline test suite:

```bash
python -m tests.run_tests
```

(Generation is mocked in tests since it requires a live Anthropic API key —
see `tests/test_pipeline.py`.)

## Project layout

```
rag-generator/
├── app/
│   ├── config.py          # env-driven settings
│   ├── document_loader.py # pdf / docx / txt / md / csv -> raw text
│   ├── chunker.py         # text -> overlapping chunks
│   ├── embeddings.py      # pluggable embedder interface + TF-IDF impl
│   ├── vector_store.py    # per-collection persistence + similarity search
│   ├── retriever.py       # top-k retrieval over a collection
│   ├── generator.py       # grounded answer generation via Claude
│   ├── rag_pipeline.py    # orchestrates ingest() and query()
│   ├── schemas.py         # request/response models
│   └── main.py            # FastAPI endpoints
├── cli.py                 # thin CLI wrapper around rag_pipeline
├── sample_docs/           # example documents for demo/testing
├── tests/                 # dependency-free unit tests
├── data/                  # runtime storage (gitignored, created at runtime)
├── requirements.txt
└── .env.example
```

## Extending it

- **Swap the embedder**: implement `Embedder` in `app/embeddings.py` (e.g.
  wrap `sentence-transformers` or an OpenAI embeddings call) and point
  `EMBEDDER_BACKEND` in `.env` at it.
- **Swap the vector store**: implement `VectorStore` in
  `app/vector_store.py` against FAISS/Chroma/pgvector; the rest of the
  pipeline only calls `.add()`, `.search()`, `.save()`, `.load()`.
- **Swap the LLM**: `app/generator.py` isolates the single call that builds
  the grounded prompt and calls the model; swap in a different provider's
  SDK there.

## Known limitations / trade-offs

- TF-IDF retrieval is lexical, not semantic — it won't catch pure paraphrase
  matches the way a dense embedding model would. It was chosen so the
  pipeline is testable fully offline with no model downloads or API keys;
  swapping in a dense embedder is a one-file change (see above).
- Single-process, local-disk persistence. Fine for the scope of this
  assessment; a production deployment would move `vector_store.py` to a real
  vector database for concurrency and scale.
- No auth on the API endpoints — add an auth dependency in `main.py` before
  exposing this beyond localhost.
