import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from kyc_verifier.verifier import NepalKYCVerifier 
from Levenshtein import ratio 

# 1. Initialize the system
verifier = NepalKYCVerifier(classifier_model_path="models/model.h5")
ground_truth = pd.read_csv("ground_truth.csv")

total_fields = 0
correct_fields = 0
all_cra_scores = []
y_true = []
y_pred = []

print("Starting Evaluation...")

for index, row in ground_truth.iterrows():
    image_path = "test_images/" + row["image"]
    
    # 2. Run verification
    result = verifier.verify_license(image_path, verbose=False)
    
    # Match the keys from your verifier.py output
    predicted_data = result.get("extracted_data", {})
    # Convert 'VERIFIED' status to 1, others to 0 for sklearn metrics
    predicted_valid = 1 if result.get("verification_status") == "VERIFIED" else 0

    # ---- FIELD ACCURACY (FEA) ----
    # Mapping CSV columns to Parser keys
    field_map = {
        "name": "name", 
        "dob": "date_of_birth", 
        "dl_number": "dl_number", 
        "citizenship": "citizenship_number"
    }

    image_gt_text = ""
    image_pred_text = ""

    for csv_col, parser_key in field_map.items():
        total_fields += 1
        gt_val = str(row[csv_col]).strip().upper()
        pred_val = str(predicted_data.get(parser_key, "")).strip().upper()

        if gt_val == pred_val:
            correct_fields += 1
        
        image_gt_text += gt_val
        image_pred_text += pred_val

    # ---- CHARACTER ACCURACY (CRA) ----
    # Using Levenshtein ratio is the industry standard for OCR CRA
    if image_gt_text:
        score = ratio(image_gt_text, image_pred_text)
        all_cra_scores.append(score)

    # ---- VALIDATION METRICS ----
    y_true.append(row["valid"])
    y_pred.append(predicted_valid)

# Final Calculations
CRA = (sum(all_cra_scores) / len(all_cra_scores)) * 100 if all_cra_scores else 0
FEA = (correct_fields / total_fields) * 100

# Sklearn Metrics
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "="*30)
print("     FINAL PROJECT REPORT")
print("="*30)
print(f"OCR PERFORMANCE:")
print(f"  Character Accuracy (CRA): {round(CRA, 2)}%")
print(f"  Field Accuracy (FEA):     {round(FEA, 2)}%")
print(f"\nVALIDATION METRICS:")
print(f"  Precision: {round(precision, 3)}")
print(f"  Recall:    {round(recall, 3)}")
print(f"  F1 Score:  {round(f1, 3)}")
print(f"  Accuracy:  {round(accuracy, 3)}")
print("="*30)