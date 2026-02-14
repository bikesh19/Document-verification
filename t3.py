# import os
# import cv2
# import numpy as np
# import easyocr
# from tensorflow.keras.models import load_model

# # -------------------------
# # Parameters
# # -------------------------
# MODEL_PATH = "model.h5"        # Path to your trained classifier
# IMG_PATH = r"./dataset/license/license_009.jpg"           # Image to test
# IMG_SIZE = (224, 224)

# # -------------------------
# # Load classifier model
# # -------------------------
# model = load_model(MODEL_PATH)

# # -------------------------
# # Initialize OCR reader
# # -------------------------
# reader = easyocr.Reader(['en'])  # English

# # -------------------------
# # 1️⃣ Classification function
# # -------------------------
# def classify_image(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Image not found: {image_path}")
#     img_resized = cv2.resize(img, IMG_SIZE)
#     img_resized = img_resized.astype(np.float32) / 255.0
#     img_resized = np.expand_dims(img_resized, axis=0)

#     prediction = model.predict(img_resized)[0][0]

#     # Assuming during training: 0=license, 1=not_license
#     if prediction > 0.5:
#         return "not_license", prediction
#     else:
#         return "license", prediction

# # -------------------------
# # 2️⃣ OCR function
# # -------------------------
# def extract_text(image_path):
#     results = reader.readtext(image_path)
#     extracted_text = []
#     for bbox, text, confidence in results:
#         extracted_text.append(text)
#     return extracted_text

# # -------------------------
# # 3️⃣ Full pipeline
# # -------------------------
# def process_image(image_path):
#     # Step 1: classify
#     label, confidence = classify_image(image_path)
#     print(f"\nClassification: {label}")
#     print(f"Confidence: {confidence:.4f}")

#     # Step 2: run OCR only if it's a license
#     if label == "license":
#         print("\nRunning OCR...")
#         texts = extract_text(image_path)

#         print("\nExtracted Text:")
#         for t in texts:
#             print("-", t)

#         # Step 3: simple verification
#         full_text = " ".join(texts).lower()
#         if "driving" in full_text and "license" in full_text:
#             print("\n✅ Driving License Verified")
#         else:
#             print("\n⚠️ Text extracted but keywords not found")

#     else:
#         print("\n❌ Not a license. OCR skipped.")
# # -------------------------
# # 4️⃣ Run pipeline
# # -------------------------
# process_image(IMG_PATH)



"""
Complete Nepal Driving License KYC Verification System
Integrates: Classification → OCR → Field Extraction → Validation
"""

import os
import cv2
import numpy as np
import easyocr
import re
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# ========================================
# 1. CLASSIFICATION (Your existing code)
# ========================================
class LicenseClassifier:
    """Classify if image is a driving license or not"""
    
    def __init__(self, model_path: str, img_size=(224, 224)):
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            self.img_size = img_size
            print(f"✓ Classifier model loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Classifier not available: {e}")
            self.model = None
    
    def classify(self, image_path: str) -> Tuple[str, float]:
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
        if prediction > 0.5:
            return "not_license", float(prediction)
        else:
            return "license", float(1 - prediction)


# ========================================
# 2. OCR EXTRACTION
# ========================================
class OCREngine:
    """Extract text from license images using EasyOCR"""
    
    def __init__(self, languages=['en']):
        print("Initializing OCR engine...")
        self.reader = easyocr.Reader(languages, verbose=False)
        print("✓ OCR engine ready")
    
    def extract_text(self, image_path: str) -> List[str]:
        """
        Extract all text from image
        Returns: List of extracted text strings
        """
        results = self.reader.readtext(image_path)
        extracted_texts = [text for (bbox, text, confidence) in results]
        return extracted_texts


# ========================================
# 3. FIELD PARSER FOR NEPAL LICENSE
# ========================================
class NepalLicenseParser:
    """Parse Nepal driving license fields from OCR text"""
    
    def __init__(self):
        self.extracted_data = {}
    
    def parse(self, ocr_texts: List[str]) -> Dict[str, Optional[str]]:
        """
        Parse OCR texts into structured fields
        
        Returns:
            Dictionary with all extracted fields
        """
        # Combine all text
        full_text = " ".join(ocr_texts)
        
        # Extract each field
        result = {
            'dl_number': self._extract_dl_number(full_text),
            'name': self._extract_name(ocr_texts),
            'date_of_birth': self._extract_dob(full_text),
            'blood_group': self._extract_blood_group(full_text),
            'address': self._extract_address(ocr_texts),
            'license_office': self._extract_license_office(ocr_texts),
            'father_husband_name': self._extract_fh_name(ocr_texts),
            'citizenship_number': self._extract_citizenship(full_text),
            'category': self._extract_category(full_text),
            'date_of_issue': self._extract_doi(ocr_texts, full_text),
            'date_of_expiry': self._extract_doe(full_text),
            'passport_number': self._extract_passport(full_text),
            'contact_number': self._extract_contact(full_text),
            'raw_ocr_text': full_text
        }
        
        self.extracted_data = result
        return result
    
    def _extract_dl_number(self, text: str) -> Optional[str]:
        """Extract DL Number: 99-26-72642298"""
        patterns = [
            r'DL\.?No\.?:?\s*([0-9\-]+)',
            r'DLNo\.?:?\s*([0-9\-]+)',
            r'(\d{2}-\d{2}-\d{8})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_name(self, texts: List[str]) -> Optional[str]:
        """Extract Name from text list"""
        for text in texts:
            if 'Name:' in text:
                parts = re.split(r'[Nn]ame:\s*', text)
                if len(parts) > 1:
                    name = re.sub(r'[,:;].*$', '', parts[1])
                    return name.strip()
        return None
    
    def _extract_address(self, texts: List[str]) -> Optional[str]:
        """Extract address (may span multiple lines)"""
        address_parts = []
        for i, text in enumerate(texts):
            if 'Address:' in text:
                parts = re.split(r'[Aa]ddress:\s*', text)
                if len(parts) > 1:
                    address_parts.append(parts[1].strip())
                
                # Get next few lines
                for j in range(i+1, min(i+4, len(texts))):
                    next_text = texts[j].strip()
                    if ':' in next_text or next_text.startswith('D.O'):
                        break
                    if next_text and not next_text.startswith('License'):
                        address_parts.append(next_text)
        
        if address_parts:
            address = ', '.join(address_parts)
            address = re.sub(r';+', ',', address)
            address = re.sub(r',+', ', ', address)
            return address.strip(', ')
        return None
    
    def _extract_dob(self, text: str) -> Optional[str]:
        """Extract Date of Birth"""
        patterns = [
            r'D\.?O\.?B\.?:?\s*(\d{2}-\d{2}-\d{4})',
            r'DOB:?\s*(\d{2}-\d{2}-\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_blood_group(self, text: str) -> Optional[str]:
        """Extract Blood Group"""
        patterns = [
            r'B\.?G\.?:?\s*([ABO][+-])',
            r'8\.6:\s*([ABO][+-])',
            r'\b([ABO][+-])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                bg = match.group(1)
                if bg in ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']:
                    return bg
        return None
    
    def _extract_license_office(self, texts: List[str]) -> Optional[str]:
        """Extract License Office"""
        for text in texts:
            if 'License Office' in text:
                parts = re.split(r'[Ll]icense [Oo]ffice[;:]\s*', text)
                if len(parts) > 1:
                    return parts[1].strip()
        return None
    
    def _extract_fh_name(self, texts: List[str]) -> Optional[str]:
        """Extract Father/Husband Name"""
        for text in texts:
            if 'FIH Name:' in text or 'F/H Name:' in text or 'FH Name:' in text:
                parts = re.split(r'F[/I]?H Name:\s*', text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    return parts[1].strip()
        return None
    
    def _extract_citizenship(self, text: str) -> Optional[str]:
        """Extract Citizenship Number"""
        patterns = [
            r'Citizenship No\.?:+\s*([\d\-]+)',
            r'(\d{2}-\d{2}-\d{2}-\d{5})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                citizenship = match.group(1).strip()
                if re.match(r'\d{2}-\d{2}-\d{2}-\d{5}', citizenship):
                    return citizenship
        return None
    
    def _extract_category(self, text: str) -> Optional[str]:
        """Extract License Category"""
        patterns = [
            r'Category:\s*([A-Z]+)\s*(?:DOE|D\.O\.E)',
            r'Category:\s*([A-Z\s]{1,4})\s+[A-Z][a-z]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                category = re.sub(r'\s+', '', match.group(1).strip())
                if re.match(r'^[ABCDE]{1,5}$', category):
                    return category
        
        # Fallback
        match = re.search(r'Category:\s*([A-Z\s]{1,5})', text)
        if match:
            category = ''.join([c for c in match.group(1) if c in 'ABCDE'])
            if category:
                return category
        return None
    
    def _extract_doi(self, texts: List[str], full_text: str) -> Optional[str]:
        """Extract Date of Issue"""
        patterns = [
            r'D\.?O\.?I\.?:?\s*(\d{2}-\d{2}-\d{4})',
            r'DOl:?\s*(\d{2}-\d{2}-\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Check for standalone date after DOI/DOl
        for i, text in enumerate(texts):
            if 'DOl:' in text or 'DOI:' in text:
                if i + 1 < len(texts):
                    next_text = texts[i + 1].strip()
                    if re.match(r'\d{2}-\d{2}-\d{4}', next_text):
                        return next_text
        return None
    
    def _extract_doe(self, text: str) -> Optional[str]:
        """Extract Date of Expiry"""
        patterns = [
            r'D\.?O\.?E\.?:?\s*(\d{2}-\d{2}-\d{4})',
            r'DOE:?\s*(\d{2}-\d{2}-\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_passport(self, text: str) -> Optional[str]:
        """Extract Passport Number"""
        patterns = [
            r'Passport No\.?:+\s*([A-Z0-9]{6,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                passport = match.group(1).strip()
                # Filter false positives
                if passport not in ['0', 'O', 'No', 'NO', 'Contact', 'CONTACT']:
                    if re.match(r'^[A-Z]', passport):
                        return passport
        return None
    
    def _extract_contact(self, text: str) -> Optional[str]:
        """Extract Contact Number"""
        patterns = [
            r'Contact No\.?:+\s*(\d{9,10})',
            r'Contact\s+No\.?:+\s+(\d{9,10})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def validate_dates(self) -> Dict[str, str]:
        """Validate extracted dates"""
        validation = {}
        
        # Validate DOB and calculate age
        dob = self.extracted_data.get('date_of_birth')
        if dob:
            try:
                dob_date = datetime.strptime(dob, '%d-%m-%Y')
                age = (datetime.now() - dob_date).days // 365
                validation['age'] = f"{age} years old"
                if age < 18:
                    validation['age_warning'] = "⚠️ Under 18 years old"
                else:
                    validation['age_status'] = "✓ Legal age"
            except:
                validation['dob_error'] = "Invalid DOB format"
        
        # Validate expiry
        doe = self.extracted_data.get('date_of_expiry')
        if doe:
            try:
                expiry_date = datetime.strptime(doe, '%d-%m-%Y')
                if expiry_date < datetime.now():
                    validation['expiry_status'] = "❌ License EXPIRED"
                else:
                    days_left = (expiry_date - datetime.now()).days
                    validation['expiry_status'] = f"✓ Valid ({days_left} days remaining)"
            except:
                validation['expiry_error'] = "Invalid expiry date"
        
        return validation
    
    def get_formatted_output(self) -> str:
        """Get formatted text output"""
        if not self.extracted_data:
            return "No data extracted"
        
        output = []
        output.append("=" * 70)
        output.append("NEPAL DRIVING LICENSE - EXTRACTED DATA")
        output.append("=" * 70)
        output.append("")
        
        fields = [
            ("DL Number", "dl_number"),
            ("Name", "name"),
            ("Date of Birth", "date_of_birth"),
            ("Blood Group", "blood_group"),
            ("Address", "address"),
            ("License Office", "license_office"),
            ("Father/Husband Name", "father_husband_name"),
            ("Citizenship No", "citizenship_number"),
            ("Category", "category"),
            ("Date of Issue", "date_of_issue"),
            ("Date of Expiry", "date_of_expiry"),
            ("Passport Number", "passport_number"),
            ("Contact Number", "contact_number"),
        ]
        
        for label, key in fields:
            value = self.extracted_data.get(key)
            output.append(f"{label:25}: {value if value else '[Not found]'}")
        
        output.append("")
        output.append("=" * 70)
        
        # Add validation
        validation = self.validate_dates()
        if validation:
            output.append("\nVALIDATION:")
            output.append("-" * 70)
            for key, value in validation.items():
                output.append(f"{key:25}: {value}")
            output.append("")
        
        return "\n".join(output)


# ========================================
# 4. MAIN KYC VERIFICATION PIPELINE
# ========================================
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
            print("⚠ No classifier model provided - skipping classification")
            self.classifier = None
        
        self.ocr_engine = OCREngine()
        self.parser = NepalLicenseParser()
        
        print("\n✓ System ready!\n")
    
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
                print("STEP 1: Classification")
                print("-" * 70)
            
            label, confidence = self.classifier.classify(image_path)
            result['classification'] = {
                'label': label,
                'confidence': float(confidence)
            }
            
            if verbose:
                print(f"Result: {label}")
                print(f"Confidence: {confidence:.2%}\n")
            
            if label != "license":
                result['verification_status'] = 'REJECTED'
                result['error'] = "Not a driving license"
                if verbose:
                    print("❌ VERIFICATION FAILED: Not a valid license")
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
                print(f"❌ OCR ERROR: {e}")
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
                print(f"❌ PARSING ERROR: {e}")
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
                print("✅ LICENSE VERIFIED SUCCESSFULLY")
            elif status == 'EXPIRED':
                print("⚠️ LICENSE EXPIRED")
            elif status == 'INCOMPLETE':
                print("⚠️ INCOMPLETE DATA EXTRACTION")
            else:
                print(f"❌ {status}")
            print("=" * 70 + "\n")
        
        return result
    
    def save_result(self, result: Dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


# ========================================
# 5. EXAMPLE USAGE
# ========================================
if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "./dataset/license/license_132.jpg"
    CLASSIFIER_MODEL_PATH = "model.h5"  # Optional, set to None if not available
    
    # Initialize verifier
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