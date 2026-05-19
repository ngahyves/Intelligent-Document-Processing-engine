#api/schemas

from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """
    Schema for the user's question.
    """
    question: str = Field(..., example="What is the total amount due on this invoice?")

class QueryResponse(BaseModel):
    """
    Schema for the RAG output.
    """
    answer: str
    sources: List[str]
    latency_ms: float
    # status to indicate if Llama 3 found the answer or no
    status: str = "success"

class DocumentInfo(BaseModel):
    """
    Schema for document metadata after classification.
    """
    filename: str
    document_type: str
    confidence_score: float

class UploadResponse(BaseModel):
    """
    Schema returned after a successful document ingestion.
    """
    message: str
    document_info: DocumentInfo
    status: str = "processed"