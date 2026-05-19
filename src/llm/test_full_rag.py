#src/llm/test_full_rag
import os
import sys
sys.path.append(os.getcwd())

from rag.embedder import TextEmbedder
from rag.vector_store import VectorStoreManager
from src.llm.rag_chain import RAGEngine
from src.config.logging_config import get_logger

logger=get_logger(__name__)

def test_end_to_end_rag():
    logger.info("---STARTING FULL RAG TEST ---")

    # 1. Setup Mock Data
    embedder = TextEmbedder()
    v_store = VectorStoreManager()
    
    text_data = [
        "The total amount for invoice #999 is $5,400.20.",
        "The billing address is 123 ML Street, Toronto, ON.",
        "Payment is due within 15 days of the invoice date."
    ]
    
    # 2. Ingest into FAISS
    vectors = embedder.embed_chunks(text_data)
    v_store.add_documents(text_data, vectors)

    # 3. Initialize RAG
    rag = RAGEngine(vector_store=v_store)

    # 4. Ask a question
    question = "What is the total price and the address?"
    result = rag.get_answer(question, embedder)

    print("\n" + "="*30)
    print(f"QUESTION: {question}")
    print(f"LLM ANSWER: {result['answer']}")
    print("="*30)

if __name__ == "__main__":
    test_end_to_end_rag()