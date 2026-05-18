# src/rag/query_engine.py

from rag.chroma_client import get_or_create_collection
from rag.embeddings_utils import EmbeddingModel
from langchain_groq import ChatGroq
from src.config.settings import GROQ_API_KEY


SYSTEM_PROMPT = """
Tu es un assistant spécialisé en extraction d'information à partir de documents.
Tu dois répondre STRICTEMENT en utilisant le contexte fourni.
Si l'information n'est pas dans le contexte, tu réponds : "Je ne trouve pas cette information dans les documents."

Règles :
- Ne jamais inventer d'informations.
- Ne jamais halluciner des montants, dates ou noms.
- Ne jamais utiliser de connaissances externes.
- Répondre de manière concise et précise.
- Si plusieurs documents contiennent des réponses différentes, le signaler.
"""


def build_prompt(context: str, question: str) -> str:
    return f"""
{SYSTEM_PROMPT}

Contexte :
{context}

Question :
{question}

Réponse :
"""


class RAGEngine:
    def __init__(self, collection_name="doc_chunks", k=4):
        self.collection = get_or_create_collection(collection_name)
        self.embedder = EmbeddingModel()
        self.k = k

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-70b-8192",
            temperature=0.2
        )

    def query(self, question: str) -> dict:
        # 1. Embedding de la question
        q_embedding = self.embedder.embed(question)

        # 2. Recherche Chroma
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=self.k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # 3. Construire le contexte
        context = "\n\n".join(documents)

        # 4. Construire le prompt RAG
        prompt = build_prompt(context, question)

        # 5. Appel LLM
        answer = self.llm.invoke(prompt).content

        # 6. Retour complet
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(documents, metadatas)
            ]
        }
