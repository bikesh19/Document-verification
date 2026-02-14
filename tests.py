import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------
# Parameters
# -------------------------
MODEL_PATH = "model.h5"
DATASET_DIR = r"dataset\license"
IMG_SIZE = (224, 224)

START = 0
END = 150  # inclusive

# -------------------------
# Load model
# -------------------------
model = load_model(MODEL_PATH)

correct = 0
total = 0

# -------------------------
# Test loop
# -------------------------
for i in range(START, END + 1):
    img_name = f"license_{i:03d}.jpg"
    img_path = os.path.join(DATASET_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Missing image: {img_name}")
        continue

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    predicted_class = "Not a Driving License" if prediction > 0.5 else "Driving License"

    print(f"{img_name} → {prediction:.4f} → {predicted_class}")

    # Since these are NO_LICENSE images, correct prediction is "Not a Driving License"
    if predicted_class == "Not a Driving License":
        correct += 1

    total += 1

# -------------------------
# Results
# -------------------------
accuracy = (correct / total) * 100 if total > 0 else 0
print("\n=========================")
print(f"Tested images : {total}")
print(f"Correct       : {correct}")
print(f"Accuracy      : {accuracy:.2f}%")
print("=========================")
