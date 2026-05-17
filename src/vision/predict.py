# src/vision/predict.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.config.settings import PYTORCH_MODEL_PATH, IMG_SIZE
from src.config.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================
# Build model
# ============================================================
def load_model(num_classes: int):
    logger.info("Loading EfficientNet-B0 for inference...")

    # same weights setting as training
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location="cpu"))
    model.eval()

    logger.info("Model loaded successfully.")
    return model


# ============================================================
# Preprocessing for inference (same as test transforms)
# ============================================================
def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# Predict function
# ============================================================
def predict_image(image_path: str, classes: list):
    logger.info(f"Running inference on: {image_path}")

    transform = get_inference_transform()

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # shape: [1, 3, H, W]

    # Load model
    model = load_model(num_classes=len(classes))

    # Inference
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    predicted_class = classes[pred_idx.item()]
    confidence = conf.item()

    logger.info(f"Prediction: {predicted_class} ({confidence:.4f})")

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            classes[i]: probs[0][i].item() for i in range(len(classes))
        }
    }
