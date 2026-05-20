# tests/test_classifier.py
from src.vision.classifier import DocumentClassifier
from src.config.settings import settings

def test_classifier_init():
    # tests if the model is well loaded
    classifier = DocumentClassifier(str(settings.ONNX_MODEL_PATH))
    assert classifier.session is not None
