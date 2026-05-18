from pathlib import Path
from prefect import flow, task, get_run_logger

from src.vision.classifier import predict_document_type
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.embeddings_utils import EmbeddingModel
from rag.chroma_client import get_or_create_collection
from rag.query_engine import RAGEngine


# -----------------------------
# TASKS
# -----------------------------

@task
def task_classify(file_path: str) -> str:
    logger = get_run_logger()
    doc_type = predict_document_type(Path(file_path))
    logger.info(f"📄 Type détecté : {doc_type}")
    return doc_type


@task
def task_ocr(file_path: str) -> str:
    logger = get_run_logger()
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    result = ocr.ocr(str(file_path), cls=True)
    text = "\n".join([line[1][0] for line in result[0]])
    logger.info(f"📝 OCR extrait {len(text)} caractères")
    return text


@task
def task_chunk(text: str) -> list[str]:
    logger = get_run_logger()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    logger.info(f"✂️ {len(chunks)} chunks générés")
    return chunks


@task
def task_embed(chunks: list[str]) -> list[list[float]]:
    logger = get_run_logger()
    embedder = EmbeddingModel()
    embeddings = embedder.embed_batch(chunks)
    logger.info(f"🧠 Embeddings générés : {len(embeddings)} vecteurs")
    return embeddings


@task
def task_upsert(chunks, embeddings, file_path, doc_type):
    logger = get_run_logger()
    collection = get_or_create_collection("doc_chunks")

    ids = [f"{Path(file_path).stem}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": file_path, "doc_type": doc_type, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings
    )

    logger.info("📦 Chunks insérés dans Chroma")
    return True


@task
def task_rag(question: str):
    rag = RAGEngine()
    return rag.query(question)


# -----------------------------
# FLOW PRINCIPAL
# -----------------------------

@flow(name="IDP-Engine-Full")
def idp_flow(file_path: str, question: str | None = None):

    doc_type = task_classify(file_path)
    text = task_ocr(file_path)
    chunks = task_chunk(text)
    embeddings = task_embed(chunks)
    task_upsert(chunks, embeddings, file_path, doc_type)

    if question:
        return task_rag(question)

    return {"status": "ingested", "doc_type": doc_type}


if __name__ == "__main__":
    idp_flow(
        file_path="samples/test_invoice.png",
        question="Quel est le montant total ?"
    )
