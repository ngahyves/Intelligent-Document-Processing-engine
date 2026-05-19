# rag/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.logging_config import get_logger

logger=get_logger('chunking')

class DocumentChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        chunk_size: number of character by chunks.
        chunk_overlap: to avoid chunking and lose the sens of the sentence
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split_text(self, text: str):
        logger.info("chunking's start.")
        chunks = self.splitter.split_text(text)
        logger.info(f"Text splitted in {len(chunks)} chunks.")
        return chunks