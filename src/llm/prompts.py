#src/llm/prompts
#pip install langchain-groq
from langchain_core.prompts import ChatPromptTemplate

class PromptFactory:
    """
    Centralized store for LLM prompts to ensure consistency and prevent hallucinations.
    """
    
    RAG_SYSTEM_PROMPT = """
    You are an expert Document Analysis Assistant. 
    Use the following pieces of retrieved context to answer the user's question.
    
    CONTEXT:
    {context}
    
    INSTRUCTIONS:
    1. If you don't know the answer based on the context, state that you don't know. 
    2. Do NOT use outside knowledge. 
    3. Keep the answer concise and professional.
    4. Mention the document source if available in the metadata.
    """

    @classmethod
    def get_rag_prompt(cls):
        return ChatPromptTemplate.from_messages([
            ("system", cls.RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])