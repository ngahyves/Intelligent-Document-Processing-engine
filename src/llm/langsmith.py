import os
from src.config import settings
from src.config.logging_config import get_logger

logger=get_logger(__name__)

class LangSmithTracer:
    """
    Handles the observability layer.
    Ensures every LLM call is traced for debugging and audit purposes.
    """
    def __init__(self):
        # Variables telling LangChain to send data to LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "idp-engine-production")
        
        if settings.LANGCHAIN_API_KEY:
            logger.info(f"LangSmith Tracing enabled on project: {os.environ['LANGCHAIN_PROJECT']}")
        else:
            logger.warning("LangSmith API Key missing. Tracing is disabled.")

    def get_run_metadata(self, doc_type: str):
        """Returns metadata to tag the traces in LangSmith dashboard."""
        return {
            "environment": settings.ENV,
            "document_type": doc_type,
            "model": "llama-3.1-8b-instant"
        }

# Global tracer instance
tracer = LangSmithTracer()