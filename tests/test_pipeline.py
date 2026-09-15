"""End-to-end pipeline tests.

Generation (the only step that needs a live LLM API call) is monkeypatched
so the full ingest -> retrieve -> generate flow can be verified offline.
"""
import shutil
import tempfile

import app.rag_pipeline as rag_pipeline_module
from app.rag_pipeline import RagPipeline


def _fake_generate_answer(question, chunks):
    from app.generator import GroundedAnswer

    return GroundedAnswer(
        answer=f"[mocked answer for: {question}] based on {len(chunks)} chunk(s)",
        used_sources=sorted({c.source for c in chunks}),
        raw_chunks=chunks,
    )


def test_ingest_then_query_end_to_end(monkeypatch):
    monkeypatch.setattr(rag_pipeline_module, "generate_answer", _fake_generate_answer)

    tmp = tempfile.mkdtemp()
    try:
        pipeline = RagPipeline(data_dir=tmp)
        pipeline.create_collection("handbook")

        files = {
            "hr.txt": b"New employees accrue fifteen vacation days per year. "
            b"Remote work is allowed up to three days per week.",
            "finance.txt": b"Expenses over twenty five dollars require a receipt. "
            b"Reimbursement happens within ten business days.",
        }
        result = pipeline.ingest("handbook", files)
        assert result.total_chunks >= 2
        assert set(result.files_ingested) == {"hr.txt", "finance.txt"}

        answer = pipeline.query("handbook", "how many vacation days do new employees get?")
        assert "mocked answer" in answer.answer
        assert answer.used_sources  # grounded in at least one source
        assert all(c.source in ("hr.txt", "finance.txt") for c in answer.raw_chunks)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_different_collections_stay_isolated(monkeypatch):
    monkeypatch.setattr(rag_pipeline_module, "generate_answer", _fake_generate_answer)

    tmp = tempfile.mkdtemp()
    try:
        pipeline = RagPipeline(data_dir=tmp)
        pipeline.create_collection("set-a")
        pipeline.create_collection("set-b")

        pipeline.ingest("set-a", {"a.txt": b"The secret code for project Alpha is 1234."})
        pipeline.ingest("set-b", {"b.txt": b"The weather in Paris is usually mild in spring."})

        answer_a = pipeline.query("set-a", "what is the secret code?")
        answer_b = pipeline.query("set-b", "what is the secret code?")

        assert any("a.txt" in c.source for c in answer_a.raw_chunks)
        # set-b has no relevant content re: secret codes, so it shouldn't
        # surface set-a's chunk (collections are fully isolated on disk).
        assert all(c.source != "a.txt" for c in answer_b.raw_chunks)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_query_nonexistent_collection_raises():
    from app.vector_store import CollectionNotFoundError

    tmp = tempfile.mkdtemp()
    try:
        pipeline = RagPipeline(data_dir=tmp)
        try:
            pipeline.query("nope", "anything")
            assert False, "expected CollectionNotFoundError"
        except CollectionNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_relevant_chunks_returns_graceful_message():
    """Without mocking: real generate_answer() short-circuits to a fixed
    message when there are zero retrieved chunks, regardless of API key."""
    tmp = tempfile.mkdtemp()
    try:
        pipeline = RagPipeline(data_dir=tmp)
        pipeline.create_collection("empty-topic")
        pipeline.ingest("empty-topic", {"a.txt": b"unrelated filler content about gardening"})

        # Force zero results by asking for top_k=0
        answer = pipeline.query("empty-topic", "irrelevant question", top_k=0)
        assert "couldn't find anything relevant" in answer.answer
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- minimal monkeypatch shim so these tests run under tests/run_tests.py
# (which has no pytest fixtures) as well as under real pytest if available.
class _MonkeyPatch:
    def __init__(self):
        self._restores = []

    def setattr(self, obj, name, value):
        self._restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._restores):
            setattr(obj, name, old)


def _run_with_monkeypatch(fn):
    mp = _MonkeyPatch()
    try:
        fn(mp)
    finally:
        mp.undo()


if __name__ == "__main__":
    _run_with_monkeypatch(test_ingest_then_query_end_to_end)
    _run_with_monkeypatch(test_different_collections_stay_isolated)
    test_query_nonexistent_collection_raises()
    test_no_relevant_chunks_returns_graceful_message()
    print("ok")
