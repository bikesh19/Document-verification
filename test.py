import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------
# Parameters
# -------------------------
MODEL_PATH = "model.h5"  # or "model.keras"
IMG_PATH = r"03.jpg"       # put test.jpg in the same folder as this script
IMG_SIZE = (224, 224)

# -------------------------
# Load model
# -------------------------
model = load_model(MODEL_PATH)

# -------------------------
# Preprocess image
# -------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError(f"Image not found: {IMG_PATH}")

img = cv2.resize(img, IMG_SIZE)
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# -------------------------
# Predict
# -------------------------
prediction = model.predict(img)[0][0]
print(f"Prediction probability: {prediction:.4f}")

# -------------------------
# Class mapping (from training)
# -------------------------
# Adjust according to your training class indices:
# Example: {'license': 0, 'no_license': 1}
license_index = 0  # license class index
if license_index == 0:
    predicted_class = "Driving License" if prediction < 0.5 else "Not a Driving License"
else:
    predicted_class = "Driving License" if prediction > 0.5 else "Not a Driving License"

print(f"Predicted class: {predicted_class}")
