import io

from app.document_loader import UnsupportedFileTypeError, load_document


def test_loads_plain_text():
    text = load_document("notes.txt", b"hello world")
    assert text == "hello world"


def test_loads_markdown_as_text():
    text = load_document("readme.md", b"# Title\n\nSome content.")
    assert "Title" in text
    assert "Some content." in text


def test_loads_csv_as_readable_rows():
    csv_bytes = b"name,age\nAlice,30\nBob,25"
    text = load_document("people.csv", csv_bytes)
    assert "name, age" in text
    assert "Alice, 30" in text


def test_unsupported_extension_raises():
    try:
        load_document("archive.zip", b"garbage")
        assert False, "expected UnsupportedFileTypeError"
    except UnsupportedFileTypeError:
        pass


def test_loads_docx():
    import docx

    buf = io.BytesIO()
    doc = docx.Document()
    doc.add_paragraph("This is a docx paragraph.")
    doc.save(buf)
    raw = buf.getvalue()

    text = load_document("test.docx", raw)
    assert "This is a docx paragraph." in text


if __name__ == "__main__":
    test_loads_plain_text()
    test_loads_markdown_as_text()
    test_loads_csv_as_readable_rows()
    test_unsupported_extension_raises()
    test_loads_docx()
    print("ok")
