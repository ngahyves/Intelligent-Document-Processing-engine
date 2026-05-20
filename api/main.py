from fastapi import FastAPI, UploadFile, File
import time

# Internal imports
from prometheus_fastapi_instrumentator import Instrumentator
from src.config import settings
from src.config.logging_config import get_logger
from src.vision.classifier import DocumentClassifier
from src.ocr.extractor import OCREngine
from rag.embedder import TextEmbedder
from rag.vector_store import VectorStoreManager
from src.llm.rag_chain import RAGEngine
from api.middleware import LoggingMiddleware
from src.llm.langsmith import tracer 

logger=get_logger(__name__)

app = FastAPI(title="IntelliDoc-Stream API")
app.add_middleware(LoggingMiddleware)
Instrumentator().instrument(app).expose(app)

# --- STEP 1: LOAD EVERYTHING AT STARTUP ---
# loading models here so they stay in RAM and don't reload on every click.
classifier = DocumentClassifier(str(settings.ONNX_MODEL_PATH))
ocr_engine = OCREngine()
embedder = TextEmbedder()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store)

#Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/upload")
async def process_document(file: UploadFile = File(...)):
    #Measuring the time request of each part in our API
    t0 = time.perf_counter()
    content = await file.read()
    
    # 1. Vision
    t1 = time.perf_counter()
    doc_type = classifier.predict(content)
    v_time = (time.perf_counter() - t1) * 1000

    # 2. OCR
    t2 = time.perf_counter()
    raw_text = ocr_engine.extract_text(content) 
    ocr_time = (time.perf_counter() - t2) * 1000
    
    # 3. RAG Ingestion
    t3 = time.perf_counter()
    vectors = embedder.embed_chunks([raw_text])
    vector_store.add_documents([raw_text], vectors)
    rag_time = (time.perf_counter() - t3) * 1000

    total_time = (time.perf_counter() - t0) * 1000
    
    logger.info(f"BREAKDOWN: Vision: {v_time:.0f}ms | OCR: {ocr_time:.0f}ms | RAG: {rag_time:.0f}ms")
    
    return {
        "type": doc_type,
        "detail": f"Total processing: {total_time/1000:.2f}s (OCR: {ocr_time/1000:.2f}s)"
    }
@app.post("/ask")
async def ask_llm(question: str):
    # Calling RAG with meta data passing for langsmith
    result = rag_engine.get_answer(
        question, 
        embedder, 
        metadata=tracer.get_run_metadata("invoice") #tag the pass
    )
    return {"answer": result["answer"]}