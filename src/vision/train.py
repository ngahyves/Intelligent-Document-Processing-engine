# src/vision/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.vision.preprocess import DataProcess
from src.config.logging_config import get_logger
from src.config.settings import PYTORCH_MODEL_PATH, SEED

import mlflow
import mlflow.pytorch

import random
import numpy as np

logger = get_logger(__name__)


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


# ============================================================
# Build model (EfficientNet-B0)
# ============================================================
def build_model(num_classes: int):
    logger.info("Loading EfficientNet-B0 pretrained model...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    logger.info(f"Model head replaced for {num_classes} classes.")
    return model


# ============================================================
# Training loop
# ============================================================
def train_model(model, train_loader, test_loader, device, classes, epochs=10):
    # Ensure a run is active
    if mlflow.active_run() is None:
        mlflow.start_run(run_name="vision_classifier_internal")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # History for graphs
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    model.to(device)
    best_acc = 0.0
    num_classes = len(classes)

    # MLflow params
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("lr", 1e-4)
    mlflow.log_param("scheduler", "CosineAnnealingLR")
    mlflow.log_param("model", "EfficientNet-B0")
    mlflow.log_param("num_classes", num_classes)

    for epoch in range(epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{epochs} =====")

        # -------------------------------------------------------
        # TRAINING PHASE
        # -------------------------------------------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        logger.info(f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

        # -------------------------------------------------------
        # EVALUATION PHASE
        # -------------------------------------------------------
        model.eval()
        all_preds = []
        all_labels = []
        test_running_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_running_loss / len(test_loader)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        test_loss_history.append(test_loss)
        test_acc_history.append(acc)

        logger.info(
            f"Test  -> Loss: {test_loss:.4f} | Acc: {acc:.4f} | "
            f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}"
        )

        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_accuracy", acc, step=epoch)
        mlflow.log_metric("precision_macro", precision, step=epoch)
        mlflow.log_metric("recall_macro", recall, step=epoch)
        mlflow.log_metric("f1_macro", f1, step=epoch)

        scheduler.step()

        # -------------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------------
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), PYTORCH_MODEL_PATH)
            logger.info(f"New best model saved at {PYTORCH_MODEL_PATH}")
            mlflow.pytorch.log_model(model, artifact_path="best_model")

    logger.info(f"\nTraining complete. Best accuracy: {best_acc:.4f}")
    mlflow.log_metric("best_accuracy", best_acc)

    return {
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "test_loss": test_loss_history,
        "test_acc": test_acc_history,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("Starting training script...")

    set_seed(SEED)

    dp = DataProcess()
    train_loader, test_loader, classes = dp.get_loaders()

    model = build_model(num_classes=len(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    mlflow.set_experiment("idp_engine")

    with mlflow.start_run(run_name="vision_classifier"):
        mlflow.set_tag("task", "vision")
        mlflow.set_tag("model", "efficientnet_b0")

        history = train_model(model, train_loader, test_loader, device, classes, epochs=10)

        # Summary
        final_test_acc = history["test_acc"][-1]
        logger.info(f"Final test accuracy: {final_test_acc:.4f}")