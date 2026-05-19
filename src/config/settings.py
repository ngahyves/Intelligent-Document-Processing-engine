# src/config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================================================================
# Base directory
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


# ==============================================================================
# Dataset paths
# ==============================================================================
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
TRAIN_DIR = os.getenv("TRAIN_DIR", str(Path(DATA_DIR) / "train"))
TEST_DIR = os.getenv("TEST_DIR", str(Path(DATA_DIR) / "test"))


# ==============================================================================
# Vision preprocessing parameters
# ==============================================================================
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
SEED = int(os.getenv("SEED", 42))


# ==============================================================================
# Model paths
# ==============================================================================
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", str(Path(MODEL_DIR) / "classifier.onnx"))
PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH", str(Path(MODEL_DIR) / "best_model.pth"))


# ==============================================================================
# Vector DB / RAG
# ==============================================================================
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "rag" / "index"))


# ==============================================================================
# API / App settings
# ==============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENV = os.getenv("ENV", "development")
PORT = int(os.getenv("PORT", 8000))


# ==============================================================================
# LLM / API Keys
# ==============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")


# ==============================================================================
# Optional Settings class (for dependency injection)
# ==============================================================================
class Settings:
    def __init__(self):
        self.BASE_DIR = BASE_DIR

        # Dataset
        self.DATA_DIR = DATA_DIR
        self.TRAIN_DIR = TRAIN_DIR
        self.TEST_DIR = TEST_DIR

        # Vision
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS
        self.SEED = SEED

        # Models
        self.MODEL_DIR = MODEL_DIR
        self.ONNX_MODEL_PATH = ONNX_MODEL_PATH
        self.PYTORCH_MODEL_PATH = PYTORCH_MODEL_PATH

        # RAG
        self.VECTOR_DB_PATH = VECTOR_DB_PATH

        # App
        self.LOG_LEVEL = LOG_LEVEL
        self.ENV = ENV
        self.PORT = PORT

        # LLM
        self.GROQ_API_KEY = GROQ_API_KEY
        self.LANGCHAIN_API_KEY = LANGCHAIN_API_KEY


settings = Settings()
