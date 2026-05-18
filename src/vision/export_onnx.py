import torch
import torch.onnx
import os

def export_to_onnx(model, device, onnx_path="models/classifier.onnx"):
    """Export a PyTorch model to ONNX format using a dummy input."""

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17
    )

    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    from src.vision.train import build_model
    from src.config.settings import PYTORCH_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    state = torch.load(PYTORCH_MODEL_PATH, map_location=device)

    # Detect number of classes from checkpoint
    num_classes = state["classifier.1.weight"].shape[0]
    print(f"Detected num_classes = {num_classes}")

    # Build model with correct head
    model = build_model(num_classes=num_classes)

    # Load weights
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    export_to_onnx(model, device)
