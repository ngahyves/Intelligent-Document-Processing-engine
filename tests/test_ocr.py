#tests/tests_ocr
from src.ocr.extractor import OCREngine
from src.ocr.post_process import clean_text
from src.config.logging_config import get_logger

logger=get_logger("test_ocr")

# 1. Initialisation
ocr = OCREngine(languages=['en', 'fr'])

# 2. Extraction
raw_text = ocr.extract_text("data/samples/ton_image.jpg")

# 3. Nettoyage
final_text = clean_text(raw_text)

logger.info(f"Résultat final : {final_text}")