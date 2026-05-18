# tests/test_classifier.py

from pathlib import Path
from src.vision.classifier import predict_document_type

def test_classifier():
    sample = Path("samples/test_invoice.png")
    result = predict_document_type(sample)

    assert isinstance(result, str)
    assert len(result) > 0
    print("Document type:", result)
