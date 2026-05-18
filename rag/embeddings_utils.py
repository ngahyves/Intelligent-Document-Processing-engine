# rag/embeddings_utils.py

from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """
        Embedding for one text (384 dimensions)
        """
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding for list of many texts
        """
        return self.model.encode(texts).tolist()

