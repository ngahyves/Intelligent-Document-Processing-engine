import numpy as np
from rag.embedder import TextEmbedder
from rag.vector_store import VectorStoreManager
from src.config import logger

def test_complete_vector_pipeline():
    logger.info("--- Starting VECTOR STORE test ---")

    # 1. Initializing tools
    embedder = TextEmbedder()
    v_store = VectorStoreManager(collection_name="test_collection")

    # 2. Custom data
    test_chunks = [
        "The total amount of the invoice is 1,250 euros.",
        "The supplier is TechCorp, located in Paris.",
        "The payment due date is June 30, 2024."
    ]
    
    # Generate unique Ids and metadata
    test_ids = [f"id_{i}" for i in range(len(test_chunks))]
    test_metadatas = [{"type": "finance"}, {"type": "contact"}, {"type": "date"}]

    # 3. Transform chunks in embeddings
    embeddings = embedder.embed_chunks(test_chunks)
    
    # Adding in chromadb
    v_store.add_documents(
        chunks=test_chunks,
        embeddings=embeddings.tolist(), # chromadb is waiting a list
        metadatas=test_metadatas,
        ids=test_ids
    )

    # 4. TEST of Retrieval
    question = "How much does the company have to pay? ?"
    logger.info(f"Question test : {question}")
    
    # vectorizing the question
    query_vector = embedder.embed_query(question)
    
    # Query in data base
    results = v_store.search(query_vector.tolist(), n_results=1)

    # 5. Verifying results
    if results['documents']:
        found_doc = results['documents'][0][0]
        found_meta = results['metadatas'][0][0]
        logger.info(f"DOCUMENT FOUND : {found_doc}")
        logger.info(f"METADATA : {found_meta}")
        
        # Simple assertion to validate
        assert "1250" in found_doc
        logger.info("TEST PASSED : semantic research is ok!")
    else:
        logger.error(" TEST FAILED : no document found.")

if __name__ == "__main__":
    test_complete_vector_pipeline()