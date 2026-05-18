# tests/test_ocr.py

from pathlib import Path
from paddleocr import PaddleOCR

def test_ocr():
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    sample = Path("samples/test_invoice.png")

    result = ocr.ocr(str(sample), cls=True)
    text = "\n".join([line[1][0] for line in result[0]])

    assert isinstance(text, str)
    assert len(text) > 0
    print("OCR text sample:", text[:200])
