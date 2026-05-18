# src/api/routers/rag.py

from fastapi import APIRouter
from rag.query_engine import RAGEngine

router = APIRouter()

@router.post("/ask")
async def ask_question(question: str):
    rag = RAGEngine()
    result = rag.query(question)
    return result
