import os
import json
from typing import Dict, Optional

from .classifier import LicenseClassifier
from .ocr import OCREngine
from .parser import NepalLicenseParser

class NepalKYCVerifier:
    """Complete KYC verification pipeline"""
    
    def __init__(self, classifier_model_path: Optional[str] = None):
        """
        Initialize the KYC verifier
        
        Args:
            classifier_model_path: Path to trained classifier model (optional)
        """
        print("\n" + "=" * 70)
        print("INITIALIZING NEPAL KYC VERIFICATION SYSTEM")
        print("=" * 70 + "\n")
        
        # Initialize components
        if classifier_model_path and os.path.exists(classifier_model_path):
            self.classifier = LicenseClassifier(classifier_model_path)
        else:
            print("[WARN] No classifier model provided - skipping classification")
            self.classifier = None
        
        self.ocr_engine = OCREngine()
        self.parser = NepalLicenseParser()
        
        print("\n[OK] System ready!\n")
    
    def verify_license(self, image_path: str, verbose: bool = True) -> Dict:
        """
        Complete verification pipeline
        
        Args:
            image_path: Path to license image
            verbose: Print detailed output
            
        Returns:
            Dictionary with verification results
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"PROCESSING: {os.path.basename(image_path)}")
            print("=" * 70 + "\n")
        
        result = {
            'image_path': image_path,
            'classification': None,
            'extracted_data': None,
            'validation': None,
            'verification_status': 'PENDING'
        }
        
        # Step 1: Classification
        if self.classifier:
            if verbose:
                print("STEP 1: Classification (Threshold: 80%)")
                print("-" * 70)
            
            # Use 80% threshold as requested
            label, confidence = self.classifier.classify(image_path, threshold=0.8)
            result['classification'] = {
                'label': label,
                'confidence': float(confidence)
            }
            
            if verbose:
                print(f"Result: {label}")
                print(f"Confidence: {confidence:.2%}\n")
            
            if label != "license":
                result['verification_status'] = 'REJECTED'
                result['error'] = "Not a driving license or low confidence"
                if verbose:
                    print(f"[FAIL] VERIFICATION FAILED: {result['error']} ({confidence:.2%})")
                return result
        
        # Step 2: OCR Extraction
        if verbose:
            print("STEP 2: OCR Text Extraction")
            print("-" * 70)
        
        try:
            ocr_texts = self.ocr_engine.extract_text(image_path)
            
            if verbose:
                print(f"Extracted {len(ocr_texts)} text elements")
                print("\nExtracted Texts:")
                for i, text in enumerate(ocr_texts[:10], 1):  # Show first 10
                    print(f"  {i}. {text}")
                if len(ocr_texts) > 10:
                    print(f"  ... and {len(ocr_texts) - 10} more")
                print()
        
        except Exception as e:
            result['verification_status'] = 'ERROR'
            result['error'] = f"OCR extraction failed: {str(e)}"
            if verbose:
                print(f"[FAIL] OCR ERROR: {e}")
            return result
        
        # Step 3: Field Parsing
        if verbose:
            print("STEP 3: Field Extraction & Parsing")
            print("-" * 70)
        
        try:
            extracted_data = self.parser.parse(ocr_texts)
            result['extracted_data'] = extracted_data
            
            if verbose:
                print(self.parser.get_formatted_output())
        
        except Exception as e:
            result['verification_status'] = 'ERROR'
            result['error'] = f"Parsing failed: {str(e)}"
            if verbose:
                print(f"[FAIL] PARSING ERROR: {e}")
            return result
        
        # Step 4: Validation
        validation = self.parser.validate_dates()
        result['validation'] = validation
        
        # Determine final status
        if 'EXPIRED' in str(validation):
            result['verification_status'] = 'EXPIRED'
        elif extracted_data.get('dl_number') and extracted_data.get('name'):
            result['verification_status'] = 'VERIFIED'
        else:
            result['verification_status'] = 'INCOMPLETE'
        
        if verbose:
            print("\nFINAL VERIFICATION STATUS")
            print("=" * 70)
            status = result['verification_status']
            if status == 'VERIFIED':
                print("[OK] LICENSE VERIFIED SUCCESSFULLY")
            elif status == 'EXPIRED':
                print("[WARN] LICENSE EXPIRED")
            elif status == 'INCOMPLETE':
                print("[WARN] INCOMPLETE DATA EXTRACTION")
            else:
                print(f"[FAIL] {status}")
            print("=" * 70 + "\n")
        
        return result
    
    def save_result(self, result: Dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
