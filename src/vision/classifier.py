import onnxruntime as ort
import numpy as np
from PIL import Image
import io

class DocumentClassifier:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.classes = ["letter", "form", "email", "invoice", "report"]

    def predict(self, image_bytes):
        # Preprocessing
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_array})
        
        # confidence score (Softmax)
        exp_out = np.exp(outputs[0][0])
        probs = exp_out / np.sum(exp_out)
        
        idx = np.argmax(probs)
        # return values
        return self.classes[idx], float(probs[idx])