import cv2
import numpy as np
from typing import Tuple

class LicenseClassifier:
    """Classify if image is a driving license or not"""
    
    def __init__(self, model_path: str, img_size=(224, 224)):
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            self.img_size = img_size
            print(f"[OK] Classifier model loaded from {model_path}")
        except Exception as e:
            print(f"[WARN] Classifier not available: {e}")
            self.model = None
    
    def classify(self, image_path: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Classify image as license or not_license
        Returns: (label, confidence)
        """
        if self.model is None:
            return "license", 1.0  # Assume license if no model
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")
        
        img_resized = cv2.resize(img, self.img_size)
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        prediction = self.model.predict(img_resized, verbose=0)[0][0]
        
        # Assuming: 0=license, 1=not_license
        # Confidence for license is (1 - prediction)
        confidence_license = float(1 - prediction)
        confidence_not_license = float(prediction)
        
        if confidence_license >= threshold:
            return "license", confidence_license
        else:
            # If not meeting license threshold, or it's clearly not_license
            label = "not_license"
            confidence = confidence_not_license
            return label, confidence
