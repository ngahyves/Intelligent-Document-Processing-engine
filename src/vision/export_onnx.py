import torch
import torch.onnx

def export_to_onnx(model, device, onnx_path="classifier.onnx"):
    """Export a PyTorch model to ONNX format using a dummy input."""
    
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
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = build_model(num_classes=2)  
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    export_to_onnx(model, device)
