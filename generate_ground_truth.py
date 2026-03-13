import os
import csv
from kyc_verifier.verifier import NepalKYCVerifier

print("========== SCRIPT STARTED ==========")

# Show current working directory
print("Current working directory:", os.getcwd())

# Model path
model_path = "models/model.h5"
print("Checking model path:", model_path)

if not os.path.exists(model_path):
    print("ERROR: Model file not found!")
    exit()

print("Model file found.")

# Initialize verifier
print("\nInitializing NepalKYCVerifier...")
try:
    verifier = NepalKYCVerifier(classifier_model_path=model_path)
    print("Verifier initialized successfully.")
except Exception as e:
    print("ERROR initializing verifier:", e)
    exit()

# Image folder
image_folder = "test_images"
output_csv = "ground_truth.csv"

print("\nChecking image folder:", image_folder)

if not os.path.exists(image_folder):
    print("ERROR: Folder not found:", image_folder)
    exit()

# List image files
files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("Images detected:", files)

if len(files) == 0:
    print("WARNING: No images found in test_images folder.")

# Create CSV
print("\nCreating CSV file:", output_csv)

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # CSV header
    writer.writerow(["image", "name", "dob", "dl_number", "citizenship", "valid"])

    # Process each image
    for file in files:
        print("\n-----------------------------------")
        print("Processing image:", file)

        path = os.path.join(image_folder, file)
        print("Image path:", path)

        try:
            result = verifier.verify_license(path, verbose=False)

            print("Verification result:", result)

            data = result.get("extracted_data", {})

            name = data.get("name", "")
            dob = data.get("date_of_birth", "")
            dl = data.get("dl_number", "")
            citizen = data.get("citizenship_number", "")

            print("Extracted Data:")
            print("Name:", name)
            print("DOB:", dob)
            print("DL Number:", dl)
            print("Citizenship:", citizen)

            writer.writerow([
                file,
                name,
                dob,
                dl,
                citizen,
                1  # Default valid
            ])

            print("Row written to CSV.")

        except Exception as e:
            print("ERROR processing", file)
            print("Exception:", e)

print("\n========== PROCESS COMPLETED ==========")
print("CSV file generated:", output_csv)
print("Please open the CSV file and manually verify the values.")
