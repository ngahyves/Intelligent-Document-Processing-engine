#rag/vector_store
#!pip install faiss-cpu --quiet

import faiss
import numpy as np
from src.config.logging_config import get_logger

logger=get_logger('vector Storing')

class VectorStoreManager:
    """
    Manages the FAISS index for high-performance semantic search.
    FAISS is used here to avoid dependency conflicts and ensure scalability.
    """
    def __init__(self, dimension: int = 384):
        """
        Initialize the FAISS index.
        :param dimension: The size of the embedding vectors (384 for all-MiniLM-L6-v2).
        """
        # IndexFlatL2 uses Euclidean distance for similarity search
        self.index = faiss.IndexFlatL2(dimension)
        
        # Dictionary to map numerical indices to actual text content
        self.doc_map = {} 
        self.current_id = 0
        
        logger.info(f"FAISS Vector Store initialized with dimension: {dimension}")

    def add_documents(self, chunks: list, embeddings: np.ndarray):
        """
        Adds text chunks and their corresponding embeddings to the index.
        :param chunks: List of strings (text segments).
        :param embeddings: Numpy array of vectors (float32).
        """
        try:
            # FAISS requires float32 precision for efficient C++ processing
            vectors = np.array(embeddings).astype('float32')
            
            # Add vectors to the underlying C++ index
            self.index.add(vectors)
            
            # Map the new vectors to their text content
            for chunk in chunks:
                self.doc_map[self.current_id] = chunk
                self.current_id += 1
                
            logger.info(f"Successfully added {len(chunks)} chunks to FAISS index.")
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {str(e)}")
            raise

    def search(self, query_embedding: np.ndarray, n_results: int = 3):
        """
        Performs a semantic search to find the most relevant document chunks.
        :param query_embedding: Vector representation of the user's question.
        :param n_results: Number of top results to return.
        :return: A dictionary containing the retrieved documents.
        """
        logger.info(f"Performing semantic search for top-{n_results} results.")
        
        # Prepare the query vector for FAISS
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search the index for distances (D) and indices (I)
        distances, indices = self.index.search(query_vector, n_results)
        
        # Retrieve the text content using the doc_map
        retrieved_docs = []
        for idx in indices[0]:
            if idx != -1:  # -1 means no match was found
                retrieved_docs.append(self.doc_map[idx])
        
        # Return format consistent with standard vector DB outputs
        return {'documents': [retrieved_docs], 'distances': distances[0]}