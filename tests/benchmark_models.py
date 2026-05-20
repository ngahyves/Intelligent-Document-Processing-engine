import time
import torch
from torchvision import models
import onnxruntime as ort
import numpy as np
from pathlib import Path

# 1. Seetting paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
ONNX_PATH = str(MODEL_DIR / "classifier.onnx")
PTH_PATH = str(MODEL_DIR / "classifier.pth")

# 2. Recreate pytorch architecture
model_pth = models.efficientnet_b0()
num_ftrs = model_pth.classifier[1].in_features
# Prediction of 5 classes as our data set
model_pth.classifier[1] = torch.nn.Linear(num_ftrs, 5) 

# Loading weights
state_dict = torch.load(PTH_PATH, map_location="cpu")
model_pth.load_state_dict(state_dict)
model_pth.eval()

# 3. Loading onnx model
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name

# 4. Create false image: Dummy Input
dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
dummy_input_torch = torch.from_numpy(dummy_input_np)

print(f"Benchmarking models in {MODEL_DIR}...")

# --- Test PyTorch ---
start = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        _ = model_pth(dummy_input_torch)
pt_time = (time.perf_counter() - start) / 20

# --- Test ONNX ---
start = time.perf_counter()
for _ in range(20):
    _ = session.run(None, {input_name: dummy_input_np})
ort_time = (time.perf_counter() - start) / 20

# --- Results ---
print("-" * 30)
print(f"PyTorch Latency: {pt_time*1000:.2f} ms")
print(f"ONNX Latency:    {ort_time*1000:.2f} ms")
print(f"Speedup:      {pt_time / ort_time:.2f}x")
print("-" * 30)