from fastapi import FastAPI, UploadFile, File
import time

# Internal imports
from src.config import settings
from src.config.logging_config import get_logger
from src.vision.classifier import DocumentClassifier
from src.ocr.extractor import OCREngine
from rag.embedder import TextEmbedder
from rag.vector_store import VectorStoreManager
from src.llm.rag_chain import RAGEngine

app = FastAPI()

# --- STEP 1: LOAD EVERYTHING AT STARTUP ---
# loading models here so they stay in RAM and don't reload on every click.
classifier = DocumentClassifier(str(settings.ONNX_MODEL_PATH))
ocr_engine = OCREngine()
embedder = TextEmbedder()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store)

@app.post("/upload")  #EfficientNet model via ONNX and EasyOCR for text extraction.
async def process_document(file: UploadFile = File(...)):
    # 1. Receive file
    content = await file.read()
    
    # 2. Classify (Vision)
    doc_type = classifier.predict(content)
    
    # 3. Extract & Store (OCR + RAG)
    raw_text = ocr_engine.extract_text(content) 
    vectors = embedder.embed_chunks([raw_text])
    vector_store.add_documents([raw_text], vectors)
    
    return {"filename": file.filename, "type": doc_type, "status": "indexed"}

@app.post("/ask") #(RAG)
async def ask_llm(question: str):
    # 4. Search & Generate (Llama 3)
    result = rag_engine.get_answer(question, embedder)
    return {"answer": result["answer"]}