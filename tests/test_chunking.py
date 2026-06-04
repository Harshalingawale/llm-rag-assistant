from rag.chunking import chunk_text


def test_chunk_basic():
    text = " ".join(str(i) for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_empty():
    assert chunk_text("") == []
