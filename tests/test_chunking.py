# tests/test_chunking
from rag.chunker import DocumentChunker

def test_split_text_logic():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a long sentence that should be split into smaller pieces for RAG."
    chunks = chunker.split_text(text)
    
    assert len(chunks) > 1
    assert isinstance(chunks, list)
    assert len(chunks[0]) <= 50