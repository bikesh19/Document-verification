import re
import cv2
import numpy as np
import easyocr
from tensorflow.keras.models import load_model

# -------------------------
# Parameters
# -------------------------
MODEL_PATH = "model.h5"
# IMG_PATH = r"/dataset/license/01.jpg"
IMG_PATH = r"C:\Users\Bikesh Sah\Desktop\k\dataset\license\license_001.jpg"

IMG_SIZE = (224, 224)

# -------------------------
# Load classifier model
# -------------------------
model = load_model(MODEL_PATH)
reader = easyocr.Reader(['en'])

# -------------------------
# Classification function
# -------------------------
def classify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_resized)[0][0]

    if prediction > 0.5:
        return "not_license", prediction
    else:
        return "license", prediction

# -------------------------
# OCR function
# -------------------------
def extract_text(image_path):
    results = reader.readtext(image_path)
    extracted_text = [text for bbox, text, conf in results]
    return extracted_text

# -------------------------
# Extract structured fields
# -------------------------
def parse_license_fields(texts):
    data = {
        "License Number": "",
        "Name": "",
        "Father/Husband Name": "",
        "DOB": "",
        "Issue Date": "",
        "Expiry Date": "",
        "Address": "",
        "Category": "",
        "Phone Number": ""
    }

    for i, line in enumerate(texts):
        line = line.strip()

        # License Number
        if "D.LNo" in line:
            parts = line.split(":")
            if len(parts) > 1:
                data["License Number"] = parts[-1].strip()

        # Name
        if "Name:" in line and "FIH" not in line:
            name_part = line.split("Name:")[-1].strip()
            # check if next line continues name
            if i + 1 < len(texts) and texts[i+1].isupper():
                name_part += " " + texts[i+1]
            data["Name"] = name_part

        # Father/Husband Name
        if "FIH Name" in line:
            if i + 1 < len(texts):
                data["Father/Husband Name"] = texts[i+1].strip()

        # DOB
        if "D.O.B" in line:
            data["DOB"] = line.split(":")[-1].strip()

        # Issue Date
        if "D.O.I" in line or "D.OL" in line:
            data["Issue Date"] = line.split()[-1]

        # Expiry Date
        if "D.O.E" in line:
            data["Expiry Date"] = line.split()[-1]

        # Category
        if "Category" in line:
            data["Category"] = line.split(":")[-1].strip()

        # Phone
        if "Phone" in line:
            numbers = ''.join(filter(str.isdigit, line))
            data["Phone Number"] = numbers

        # Address
        if "Address" in line:
            address = ""
            j = i + 1
            while j < len(texts) and "D.O.B" not in texts[j]:
                address += texts[j] + " "
                j += 1
            data["Address"] = address.strip()

    return data

# -------------------------
# Full pipeline
# -------------------------
def process_image(image_path):
    label, confidence = classify_image(image_path)
    print(f"Classification: {label} | Confidence: {confidence:.4f}")

    if label == "license":
        texts = extract_text(image_path)
        structured_data = parse_license_fields(texts)
        print("\nStructured Data Extracted:")
        for k, v in structured_data.items():
            print(f"{k}: {v}")
        return structured_data
    else:
        print("Not a license. Skipping OCR.")
        return None

# -------------------------
# Run
# -------------------------
data = process_image(IMG_PATH)
