# rag/chroma_client.py

from pathlib import Path
import chromadb
from chromadb.config import Settings

def get_chroma_client():
    # __file__ = rag/chroma_client.py
    # parent = rag/, parent.parent = root of the project
    root_dir = Path(__file__).resolve().parent.parent
    persist_dir = root_dir / "vectorstore"
    
    # Create if it doesn't exist
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_dir)  # ChromaDB is waiting an str
        )
    )
    return client


def get_or_create_collection(name="document_chunks"):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )