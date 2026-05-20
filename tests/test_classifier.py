# tests/test_classifier.py
import pytest
import os
from src.vision.classifier import DocumentClassifier
from src.config import settings

# verify if the file exists
MODEL_EXISTS = os.path.exists(str(settings.ONNX_MODEL_PATH))

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model file not found. Skipping test in CI environment.")
def test_classifier_init():
    """
    Test the classifier initialization.
    This test only runs if the .onnx file is present (local dev).
    It is skipped on GitHub Actions to avoid failure due to missing large artifacts.
    """
    classifier = DocumentClassifier(str(settings.ONNX_MODEL_PATH))
    assert classifier.session is not None
    assert "invoice" in classifier.classes