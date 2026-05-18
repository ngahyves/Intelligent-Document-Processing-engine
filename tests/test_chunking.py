# tests/test_chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def test_chunking():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    text = "Bonjour Yves. " * 200
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    print("Chunks:", len(chunks))
