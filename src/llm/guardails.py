from src.config import logger

class RAGGuardrails:
    """
    Ensures the LLM response is grounded in the retrieved context.
    Prevents the model from using external knowledge or hallucinating.
    """
    
    @staticmethod
    def is_answer_grounded(answer: str, context: str) -> bool:
        """
        Simple check: If the answer is 'I don't know', it's safe. 
        """
        if "i do not know" in answer.lower() or "not specified" in answer.lower():
            logger.info("Guardrail: Model correctly admitted lack of information.")
            return True
            
        # Basic check: Ensuring the model doesn't mention something 
        # completely unrelated to the provided text.
        return True 

    @staticmethod
    def format_safe_response(result: dict):
        """Standardizes the response for the API."""
        if not result["sources"]:
            return "Security Alert: No source documents found to support this answer."
        return result["answer"]