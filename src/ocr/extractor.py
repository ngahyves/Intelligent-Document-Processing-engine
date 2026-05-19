# src/ocr/extractor.py
import easyocr
import numpy as np
from src.config.logging_config import get_logger

logger = get_logger("extraction")

class OCREngine:
    def __init__(self, languages=["en", "fr"], gpu=True):
        logger.info(f"Initialize EasyOCR for languages : {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, image_path: str):
        """
        extract text from image and return chain of characters
        """
        try:
            logger.info(f"starting ocr extraction for : {image_path}")
            results = self.reader.readtext(image_path)

            extracted_lines = [res[1] for res in results]
            full_text = " ".join(extracted_lines)

            logger.info(f"number of lines detected : {len(extracted_lines)}")
            return full_text

        except Exception as e:
            logger.error(f"Error in OCR process : {e}")
            raise e

if __name__ == "__main__":
    engine = OCREngine()
    text = engine.extract_text("data/samples/2078153623.tif")
    print(f"Texte extrait : {text[:200]}...")
