# rag/embeddings_utils.py

from sentence_transformers import SentenceTransformer
from src.config.logging_config import get_logger

logger=get_logger('Embedding')

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"loading embedding model : {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list):
        """
        Transform text's list in vector.
        """
        logger.info(f"Generate embeddings for {len(chunks)} chunks.")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        return embeddings

    def embed_query(self, query: str):
        """
        Transform a user's question in vector.
        """
        return self.model.encode([query])[0]