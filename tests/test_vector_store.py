#tests/test_vector_store
#This script validates that the Embedder and the
#FAISS Index work together correctly to find information based on meaning.
import numpy as np
import sys
import os

# Ensure the project root is in the python path
sys.path.append(os.getcwd())

from rag.embedder import TextEmbedder
from rag.vector_store import VectorStoreManager
from src.config.logging_config import get_logger

logger=get_logger('test vector storing')

def test_faiss_integration():
    """
    End-to-end test for the RAG ingestion pipeline:
    Text -> Embeddings -> FAISS Storage -> Semantic Query.
    """
    logger.info("---STARTING FAISS INTEGRATION TEST ---")

    try:
        # 1. Initialize Components
        # MiniLM generates vectors of size 384
        embedder = TextEmbedder()
        v_store = VectorStoreManager(dimension=384)

        # 2. Define Mock Document Data (Invoice Example)
        test_chunks = [
            "The total amount due for this invoice is $1,250.00.",
            "Payment must be received by June 30, 2024.",
            "Service provider: TechCorp Solutions Inc., Toronto, Canada."
        ]
        
        # 3. Generate Embeddings
        # Converting raw text into mathematical vectors
        embeddings = embedder.embed_chunks(test_chunks)
        
        # 4. Store in FAISS
        v_store.add_documents(chunks=test_chunks, embeddings=embeddings)

        # 5. Execute Semantic Search (The 'R' in RAG)
        user_query = "What is the cost and when should I pay?"
        logger.info(f"User Query: {user_query}")
        
        query_vector = embedder.embed_query(user_query)
        search_results = v_store.search(query_vector, n_results=2)

        # 6. Validate Results
        if search_results['documents'][0]:
            top_result = search_results['documents'][0][0]
            logger.info(f"TOP MATCH FOUND: {top_result}")
            
            # Verify that the most relevant chunk was retrieved
            assert "1,250" in top_result or "June 30" in top_result
            logger.info("SUCCESS: FAISS correctly retrieved the relevant context!")
        else:
            logger.error("FAILURE: No documents were retrieved.")
            raise ValueError("Empty search results")

    except Exception as e:
        logger.error(f"CRITICAL TEST ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_faiss_integration()