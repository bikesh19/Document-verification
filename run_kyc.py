import os
import json
from kyc_verifier import NepalKYCVerifier

def main():
    # Configuration
    IMAGE_PATH = r"test_images\test.jpg"
    CLASSIFIER_MODEL_PATH = "models/model.h5"  # Optional, set to None if not available
    
    # Initialize verifier
    # Ensure relative imports or current directory is in path
    verifier = NepalKYCVerifier(
        classifier_model_path=CLASSIFIER_MODEL_PATH if os.path.exists(CLASSIFIER_MODEL_PATH) else None
    )
    
    # Verify license
    result = verifier.verify_license(IMAGE_PATH, verbose=True)
    
    # Save result
    output_file = "kyc_verification_result.json"
    verifier.save_result(result, output_file)
    
    # Print JSON result
    print("\nJSON OUTPUT:")
    print("=" * 70)
    print(json.dumps(result['extracted_data'], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
