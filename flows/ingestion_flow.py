#flows/ingestion_flow

from prefect import task, flow
from src.ocr.extractor import OCREngine
from rag.embedder import TextEmbedder
from rag.vector_store import VectorStoreManager
from pathlib import Path
from src.config.logging_config import get_logger

logger=get_logger(__name__)

# 1. Defining different tasks
@task(retries=2, retry_delay_seconds=10)
def process_single_doc(file_path, ocr, embedder, v_store):
    logger.info(f"Processing: {file_path.name}")
    
    # OCR
    text = ocr.extract_text(str(file_path))
    
    # RAG Ingestion
    vectors = embedder.embed_chunks([text])
    v_store.add_documents([text], vectors)
    
    return f"Success: {file_path.name}"

# 2.Working flow
@flow(name="IDP Batch Ingestion")
def document_ingestion_flow(folder_path: str):
    # Initialization
    ocr = OCREngine()
    embedder = TextEmbedder()
    v_store = VectorStoreManager()
    
    # List all the files in the folder
    files = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.tif"))
    
    logger.info(f"Found {len(files)} documents to ingest.")
    
    # Execution of tasks
    results = []
    for f in files:
        res = process_single_doc(f, ocr, embedder, v_store)
        results.append(res)
        
    return results

if __name__ == "__main__":
    # local test
    document_ingestion_flow(folder_path="data/samples")