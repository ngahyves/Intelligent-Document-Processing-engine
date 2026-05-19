#src/ocr/extractor.py
#pip install easyocr
import easyocr
import numpy as np
from src.config.load_config import get_logger

#Calling the logger
logger= get_logger("extraction")

class OCREngine:
    def __init__(self, languages=["en", "fr"], gpu=True):

        logger.info(f"Initialize EasyOCR for languages : {languages}")
        # Weights are initialised one time
        self.reader = easyocr.Reader(languages, gpu=gpu)

        def extract_text(self, image_path: str):
            """
            tract text from image and return chain of characters
            """
            try:
                logger.info(f"arting ocr extraction for : {image_path}")
                results = self.reader.readtext(image_path)
            
                # EasyOCR return tuples : (bbox, text, confidence)
                extracted_lines = [res[1] for res in results]
                full_text = " ".join(extracted_lines)
            
                logger.info(f"number of lines detected : {len(extracted_lines)}")
                return full_text
            
            except Exception as e:
                logger.error(f"Error in OCR process : {e}")
            raise e
#Testing our extractor
if __name__ == "__main__":
    engine = OCREngine()
    text = engine.extract_text("data/samples/2078153623.tif")
    print(f"Texte extrait : {text[:200]}...")
