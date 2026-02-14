import os
import cv2
import numpy as np
import easyocr
from tensorflow.keras.models import load_model

# -------------------------
# Parameters
# -------------------------
MODEL_PATH = "model.h5"        # Path to your trained classifier
IMG_PATH = r"01.jpg"           # Image to test
IMG_SIZE = (224, 224)

# -------------------------
# Load classifier model
# -------------------------
model = load_model(MODEL_PATH)

# -------------------------
# Initialize OCR reader
# -------------------------
reader = easyocr.Reader(['en'])  # English

# -------------------------
# 1️⃣ Classification function
# -------------------------
def classify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_resized)[0][0]

    # Assuming during training: 0=license, 1=not_license
    if prediction > 0.5:
        return "not_license", prediction
    else:
        return "license", prediction

# -------------------------
# 2️⃣ OCR function
# -------------------------
def extract_text(image_path):
    results = reader.readtext(image_path)
    extracted_text = []
    for bbox, text, confidence in results:
        extracted_text.append(text)
    return extracted_text

# -------------------------
# 3️⃣ Full pipeline
# -------------------------
def process_image(image_path):
    # Step 1: classify
    label, confidence = classify_image(image_path)
    print(f"\nClassification: {label}")
    print(f"Confidence: {confidence:.4f}")

    # Step 2: run OCR only if it's a license
    if label == "license":
        print("\nRunning OCR...")
        texts = extract_text(image_path)

        print("\nExtracted Text:")
        for t in texts:
            print("-", t)

        # Step 3: simple verification
        full_text = " ".join(texts).lower()
        if "driving" in full_text and "license" in full_text:
            print("\n✅ Driving License Verified")
        else:
            print("\n⚠️ Text extracted but keywords not found")

    else:
        print("\n❌ Not a license. OCR skipped.")
# -------------------------
# 4️⃣ Run pipeline
# -------------------------
process_image(IMG_PATH)



