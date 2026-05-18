# src/api/clients/llm_client.py

from openai import OpenAI
from src.config.settings import OPENAI_API_KEY

class LLMClient:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def ask(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
