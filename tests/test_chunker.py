from app.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", source="a.txt") == []
    assert chunk_text("   \n\n  ", source="a.txt") == []


def test_short_text_becomes_single_chunk():
    chunks = chunk_text("Hello world.", source="a.txt", chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].source == "a.txt"
    assert chunks[0].index == 0


def test_long_text_splits_into_multiple_chunks_respecting_size():
    paragraphs = [f"Paragraph number {i}. " * 10 for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, source="b.txt", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        # Allow slack: a single oversized paragraph is hard-split at exactly
        # chunk_size, but packed chunks may exceed it slightly due to overlap.
        assert len(c.text) <= 400
    # indices are sequential starting at 0
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_single_huge_paragraph_is_hard_split():
    text = "word " * 1000  # one giant paragraph, no blank lines
    chunks = chunk_text(text, source="c.txt", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


if __name__ == "__main__":
    test_empty_text_returns_no_chunks()
    test_short_text_becomes_single_chunk()
    test_long_text_splits_into_multiple_chunks_respecting_size()
    test_single_huge_paragraph_is_hard_split()
    print("ok")
