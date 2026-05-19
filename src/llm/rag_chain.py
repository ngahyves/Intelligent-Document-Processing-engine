#src/llm/rag_chain.py
#langchain RAG pipeline
# pip install langchain-groq langchain-core langchain
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config.settings import settings
from src.config.logging_config import get_logger
from src.llm.prompts import PromptFactory

logger=get_logger(__name__)

class RAGEngine:
    def __init__(self, vector_store):
        """
        Initializes the RAG chain using Groq LPU for fast inference.
        """
        self.vector_store = vector_store

        
        # 1. Initialize LLM (Llama 3 8B is perfect for speed/accuracy)
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=settings.GROQ_API_KEY, 
            model_name="llama3-8b-8192"
        )
        
        # 2. Setup Prompt
        self.prompt = PromptFactory.get_rag_prompt()
        
        logger.info("RAG Engine initialized with Llama 3 via Groq.")

    def format_docs(self, docs):
        """Helper to merge retrieved chunks into one string."""
        return "\n\n".join(docs)

    def get_answer(self, question: str, embedder):
        """
        The full RAG cycle.
        """
        try:
            logger.info(f"Answering question: {question}")
            
            # A. RETRIEVAL: Get vectors from FAISS
            query_vector = embedder.embed_query(question)
            search_results = self.vector_store.search(query_vector, n_results=3)
            context = self.format_docs(search_results['documents'][0])
            
            # B. AUGMENTATION & GENERATION:
            # We use LangChain Expression Language (LCEL) to build the chain
            chain = (
                self.prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return {
                "answer": response,
                "sources": search_results['documents'][0]
            }
            
        except Exception as e:
            logger.error(f"Error in RAG Engine: {str(e)}")
            return {"answer": "I encountered an error processing your request.", "sources": []}