#src/llm.grok_client

from langchain_groq import ChatGroq
from src.config.settings import settings  
from src.config.logging_config import get_logger

logger=get_logger(__name__)

class GroqClient:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0):
        try:
            logger.info(f"Connecting to Groq using model: {model_name}")
            
            # We access the key through our centralized settings object
            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY, 
                model_name=model_name,
                temperature=temperature
            )
            logger.info("Groq LLM connection established.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Groq: {e}")
            raise

    def get_model(self):
        return self.llm