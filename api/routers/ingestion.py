# src/api/routers/ingestion.py

from fastapi import APIRouter, UploadFile
from rag.ingestion_pipeline import IngestionPipeline

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile):
    pipeline = IngestionPipeline()
    result = pipeline.process_document(file.file)
    return {"status": "ok", "result": result}
