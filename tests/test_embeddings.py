# tests/test_embeddings.py

from rag.embeddings_utils import EmbeddingModel

def test_embeddings():
    embedder = EmbeddingModel()
    vec = embedder.embed("Bonjour Yves")

    assert isinstance(vec, list)
    assert len(vec) == 384
    print("Embedding OK, dim:", len(vec))
