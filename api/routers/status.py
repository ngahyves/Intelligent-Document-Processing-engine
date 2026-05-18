# src/api/routers/status.py

from fastapi import APIRouter
from rag.chroma_client import get_or_create_collection
from rag.embeddings_utils import EmbeddingModel

router = APIRouter()

@router.get("/status")
def status():
    try:
        _ = get_or_create_collection("doc_chunks")
        embedder = EmbeddingModel()
        test_vec = embedder.embed("ping")
        return {"db": "ok", "embeddings": "ok", "status": "ready"}
    except Exception as e:
        return {"status": "error", "details": str(e)}
