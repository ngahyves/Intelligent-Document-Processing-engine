# rag/insert_chunks.py

import uuid
from rag.chroma_client import get_or_create_collection
from rag.embeddings_utils import EmbeddingModel

def insert_chunk(content: str, metadata: dict = None):
    collection = get_or_create_collection("document_chunks")
    embedder = EmbeddingModel()

    embedding = embedder.embed(content)

    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[content],
        metadatas=[metadata or {}],
        embeddings=[embedding]
    )

    print("Chunk inserted successfully!")
