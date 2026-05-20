from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import settings
from src.config.logging_config import get_logger

logger=get_logger(__name__)

class RAGEngine:
    """
    Handles the Retrieval-Augmented Generation (RAG) logic.
    Connects the Vector Store context with the Llama 3 model on Groq.
    """
    def __init__(self, vector_store):
        # 1. Initialize the LLM (Llama 3.1)
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        # 2. Store the FAISS vector store reference
        self.vector_store = vector_store

        # 3. Define the System Prompt (Grounding)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question using ONLY the context provided below.\n\nContext: {context}"),
            ("human", "{question}")
        ])

        logger.info("RAG Engine successfully initialized with Llama 3.1")

    def get_answer(self, question: str, embedder, metadata: dict = None):
        """
        The full RAG process: Retrieve -> Augment -> Generate.
        :param metadata: Information passed to LangSmith for tracking.
        """
        try:
            # A. RETRIEVAL: Find relevant text in FAISS
            query_vector = embedder.embed_query(question)
            search_results = self.vector_store.search(query_vector, n_results=3)
            context_text = "\n\n".join(search_results['documents'][0])
            
            # B. CHAIN: Link Prompt -> LLM -> Output Parser
            # We use the '|' (pipe) operator for LangChain syntax
            chain = self.prompt_template | self.llm | StrOutputParser()
            
            # C. EXECUTION: Send to Groq and track with LangSmith
            # The 'config' dictionary is how LangChain talks to LangSmith
            langchain_config = {"metadata": metadata} if metadata else {}
            
            answer = chain.invoke(
                {"context": context_text, "question": question},
                config=langchain_config
            )
            
            return {
                "answer": answer,
                "sources": search_results['documents'][0]
            }

        except Exception as e:
            logger.error(f"Error during RAG inference: {str(e)}")
            raise e