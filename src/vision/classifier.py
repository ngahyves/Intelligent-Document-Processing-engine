import onnxruntime as ort
import numpy as np
from PIL import Image
import io

class DocumentClassifier:
    def __init__(self, model_path):
        # Load the model once
        self.session = ort.InferenceSession(model_path)
        self.classes = ["letter", "form", "email", "invoice", "report"]

    def predict(self, image_bytes):
        # 1. Simple Preprocessing: Resize and Normalize
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # 2. Run Inference
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_array})
        
        # 3. Get the class with highest score
        idx = np.argmax(outputs[0])
        return self.classes[idx]