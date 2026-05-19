#rag/vector_store

import chromadb
from src.config import settings
from src.config.logging_config import get_logger

logger=get_logger('vectostore')

class VectorStoreManager:
    def __init__(self, collection_name="document_collection"):
        """
        Initialize ChromaDB in persistant mode  (saving on disk).
        """
        self.db_path = str(settings.VECTOR_DB_PATH)
        logger.info(f"Initialization of ChromaDB : {settings.VECTOR_DB_PATH}")
        
        # 1. client creation in persistent mode
        self.client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_PATH))
        
        # 2. Gathering collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: list, embeddings: list, metadatas: list, ids: list):
        """
        Adding chunks and vector in database.
        """
        try:
            logger.info(f"Adding {len(chunks)} documents in ChromaDB.")
            self.collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(), # Chroma is expecting lists
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Documents added succesfully.")
        except Exception as e:
            logger.error(f"Error when adding to ChromaDB : {e}")

    def query_similar_docs(self, query_embedding: list, n_results=3):
        """
        Check the most closed N chunks to a vector .
        """
        logger.info(f"Research of {n_results} similar documents .")
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results
    

import chromadb
from src.config import settings, logger

class VectorStoreManager:
    def __init__(self, collection_name="idp_documents"):
        # On utilise directement le chemin de ton fichier settings
        self.db_path = str(settings.VECTOR_DB_PATH)
        
        logger.info(f"Initialisation de ChromaDB au chemin : {self.db_path}")

        # Pas besoin de l'objet Settings ici, le PersistentClient gère tout
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Création ou récupération de la collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks, embeddings, metadatas, ids):
        try:
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ {len(chunks)} morceaux ajoutés à ChromaDB.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout : {e}")

    def search(self, query_embedding, n_results=3):
        logger.info(f"Recherche sémantique : top-{n_results} résultats.")
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )