# src/rag/ingestion_pipeline.py
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.vision.classifier import predict_document_type
from src.rag.chroma_client import get_chroma_client
from src.rag.embeddings_utils import EmbeddingModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(self):
        self.ocr = PaddleOCR(lang='en', use_angle_cls=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        self.collection = get_chroma_client("doc_chunks")
        self.embedder = EmbeddingModel()

    def _get_doc_hash(self, file_path: Path) -> str:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _doc_already_exists(self, doc_hash: str) -> bool:
        result = self.collection.get(where={"doc_hash": doc_hash}, limit=1)
        return len(result['ids']) > 0

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        logger.info(f"Processing {file_path}...")
        doc_hash = self._get_doc_hash(file_path)

        if self._doc_already_exists(doc_hash):
            logger.info(f"Skipping {file_path} (already indexed)")
            return {"status": "skipped", "reason": "already indexed", "hash": doc_hash}

        # 1. Vision
        try:
            doc_type = predict_document_type(file_path)
        except Exception as e:
            logger.warning(f"Classification failed for {file_path}: {e}")
            doc_type = "unknown"

        # 2. OCR
        result = self.ocr.ocr(str(file_path), cls=True)
        if not result or not result[0]:
            logger.error(f"OCR failed or empty for {file_path}")
            return {"status": "failed", "reason": "OCR empty", "file": str(file_path)}

        full_text = "\n".join([line[1][0] for line in result[0]])

        if not full_text.strip():
            logger.warning(f"OCR returned empty text for {file_path}")
            return {"status": "failed", "reason": "OCR empty text", "file": str(file_path)}

        # 3. Chunking
        chunks = self.splitter.split_text(full_text)
        logger.info(f"Created {len(chunks)} chunks for {file_path}")

        # 4. Embeddings (batch)
        embeddings = self.embedder.embed_batch(chunks)

        # 5. Upsert Chroma
        ids = [f"{doc_hash}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": str(file_path), "doc_type": doc_type, "doc_hash": doc_hash, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

        logger.info(f"Successfully indexed {file_path}: {len(chunks)} chunks")
        return {
            "status": "success",
            "hash": doc_hash,
            "doc_type": doc_type,
            "num_chunks": len(chunks),
            "file": str(file_path)
        }