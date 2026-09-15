import shutil
import tempfile

from app.chunker import Chunk
from app.vector_store import CollectionNotFoundError, VectorStore


def _sample_chunks():
    return [
        Chunk(text="employees get fifteen vacation days per year", index=0, source="hr.md"),
        Chunk(text="expense reports are reimbursed within ten business days", index=1, source="finance.txt"),
        Chunk(text="remote work is allowed up to three days per week", index=2, source="hr.md"),
    ]


def test_build_and_search_finds_relevant_chunk():
    store = VectorStore("test-collection", tempfile.mkdtemp())
    store.build(_sample_chunks())
    results = store.search("how many vacation days do employees get", top_k=2)
    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert "vacation" in top_chunk.text
    assert top_score > 0


def test_search_on_empty_collection_returns_empty():
    store = VectorStore("empty-collection", tempfile.mkdtemp())
    assert store.search("anything", top_k=5) == []


def test_save_and_load_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        store = VectorStore("roundtrip", tmp)
        store.build(_sample_chunks())
        store.save()

        loaded = VectorStore.load("roundtrip", tmp)
        assert len(loaded.chunks) == len(store.chunks)

        results = loaded.search("remote work days per week", top_k=1)
        assert results and "remote work" in results[0][0].text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_missing_collection_raises():
    try:
        VectorStore.load("does-not-exist", tempfile.mkdtemp())
        assert False, "expected CollectionNotFoundError"
    except CollectionNotFoundError:
        pass


def test_add_documents_grows_collection():
    tmp = tempfile.mkdtemp()
    store = VectorStore("growing", tmp)
    store.build(_sample_chunks()[:1])
    assert len(store.chunks) == 1
    store.add_documents(_sample_chunks()[1:])
    assert len(store.chunks) == 3


def test_list_and_delete_collections():
    tmp = tempfile.mkdtemp()
    store = VectorStore("to-delete", tmp)
    store.build(_sample_chunks())
    store.save()
    assert "to-delete" in VectorStore.list_collections(tmp)

    VectorStore.delete("to-delete", tmp)
    assert "to-delete" not in VectorStore.list_collections(tmp)


if __name__ == "__main__":
    test_build_and_search_finds_relevant_chunk()
    test_search_on_empty_collection_returns_empty()
    test_save_and_load_roundtrip()
    test_load_missing_collection_raises()
    test_add_documents_grows_collection()
    test_list_and_delete_collections()
    print("ok")
