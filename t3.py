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
import sys
import cv2
import numpy as np
import easyocr
import re
import json
import tempfile
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# Fix Unicode output on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ========================================
# 0. IMAGE PREPROCESSOR
# ========================================
class ImagePreprocessor:
    """Preprocess license images: auto-crop, deskew, enhance for OCR"""

    @staticmethod
    def order_points(pts):
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        """Apply perspective transform to get a top-down view of the card"""
        rect = ImagePreprocessor.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def auto_crop_card(image):
        """
        Detect and crop the license card from a photo.
        Uses contour detection to find the largest rectangular shape.
        Falls back to the original image if no card is found.
        """
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        # Dilate to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("  ⚠ No contours found, using original image")
            return orig

        # Sort by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                # Found a quadrilateral — likely the card
                area = cv2.contourArea(approx)
                img_area = image.shape[0] * image.shape[1]

                # Card should be at least 10% of the image
                if area > img_area * 0.1:
                    print("  ✓ Card detected, applying perspective transform")
                    pts = approx.reshape(4, 2).astype("float32")
                    return ImagePreprocessor.four_point_transform(orig, pts)

        print("  ⚠ No card rectangle found, using original image")
        return orig

    @staticmethod
    def deskew(image):
        """Correct small rotations by detecting text line angle"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold to get text regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = -(angle - 90)
        else:
            angle = -angle

        # Only correct small angles (< 15 degrees)
        if abs(angle) > 15 or abs(angle) < 0.5:
            return image

        print(f"  ✓ Deskewing by {angle:.1f}°")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def enhance_for_ocr(image):
        """Enhance image contrast for better OCR (light touch)"""
        # Resize if too small
        h, w = image.shape[:2]
        if w < 800:
            scale = 800 / w
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        # Light denoise
        denoised = cv2.bilateralFilter(image, 5, 50, 50)

        # Gentle contrast boost using CLAHE on L channel
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    @staticmethod
    def preprocess(image_path: str) -> str:
        """
        Full preprocessing pipeline.
        Returns path to the preprocessed temp image file.
        """
        print("\nPREPROCESSING:")
        print("-" * 70)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        # Step 1: Auto-crop the card
        cropped = ImagePreprocessor.auto_crop_card(img)

        # Step 2: Deskew
        deskewed = ImagePreprocessor.deskew(cropped)

        # Step 3: Enhance for OCR
        enhanced = ImagePreprocessor.enhance_for_ocr(deskewed)

        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        cv2.imwrite(temp_path, enhanced)
        print(f"  ✓ Preprocessed image saved ({enhanced.shape[1]}x{enhanced.shape[0]})")

        return temp_path

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
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> List[str]:
        """
        Extract all text from image.
        If preprocess=True, tries both preprocessed and original image,
        and picks whichever extracts more text.
        Returns: List of extracted text strings
        """
        # Always get original OCR results
        original_results = self.reader.readtext(image_path)
        original_texts = [text for (bbox, text, confidence) in original_results]

        if not preprocess:
            return original_texts

        # Try preprocessed version
        temp_path = None
        preprocessed_texts = []
        try:
            temp_path = ImagePreprocessor.preprocess(image_path)
            preprocessed_results = self.reader.readtext(temp_path)
            preprocessed_texts = [text for (bbox, text, confidence) in preprocessed_results]
        except Exception as e:
            print(f"  ⚠ Preprocessing failed ({e}), using original image")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        # Pick the version with more extracted text
        orig_len = sum(len(t) for t in original_texts)
        prep_len = sum(len(t) for t in preprocessed_texts)

        if prep_len > orig_len:
            print(f"  ✓ Using preprocessed OCR ({len(preprocessed_texts)} elements vs {len(original_texts)} original)")
            return preprocessed_texts
        else:
            print(f"  ✓ Using original OCR ({len(original_texts)} elements, better than preprocessed)")
            return original_texts


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
            r'D\.?L\.?No\.?:*\s*([0-9\-]+)',
            r'DLNo\.?:*\s*([0-9\-]+)',
            r'(\d{2}-\d{2}-\d{6,8})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_name(self, texts: List[str]) -> Optional[str]:
        """Extract Name from text list"""
        full_text = " ".join(texts)

        # Try from full text first (handles split across OCR elements)
        match = re.search(r'\bName:*\s+([A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)*)', full_text)
        if match:
            name = match.group(1).strip()
            # Remove trailing field labels (B.G, Address, DOB, etc.)
            name = re.sub(r'\s+(?:B\.?G\.?|Address|D\.?O\.?B|FIH|F/H|Category).*$', '', name, flags=re.IGNORECASE)
            if name:
                return name.strip()

        # Fallback: per-element search
        for i, text in enumerate(texts):
            if 'Name' in text and 'FIH' not in text and 'F/H' not in text:
                parts = re.split(r'[Nn]ame:*\s*', text)
                if len(parts) > 1 and parts[1].strip():
                    name = parts[1].strip()
                    return name
                # Name might be in next element
                elif i + 1 < len(texts):
                    next_t = texts[i + 1].strip()
                    if next_t and next_t[0].isupper() and ':' not in next_t:
                        return next_t
        return None
    
    def _extract_address(self, texts: List[str]) -> Optional[str]:
        """Extract address (may span multiple lines)"""
        # Try from full text first
        full_text = " ".join(texts)
        # Handle Address followed by space or colon, stopping at common next-field markers
        match = re.search(r'Address[:\s]*(.+?)(?=\s*(?:D\.?O\.?B|License Office|FIH|F/H|FM|Category|$))', full_text, re.IGNORECASE)
        if match:
            address = match.group(1).strip()
            # Replace common OCR misreads in separators
            address = re.sub(r'[;:]+', ',', address)
            address = re.sub(r',+', ', ', address)
            address = address.strip(', ')
            if address:
                return address

        # Fallback: per-element search
        address_parts = []
        for i, text in enumerate(texts):
            if re.search(r'Address:?', text, re.IGNORECASE):
                parts = re.split(r'[Aa]ddress[:\s]*', text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    address_parts.append(parts[1].strip())

                # Get next few lines
                for j in range(i+1, min(i+4, len(texts))):
                    next_text = texts[j].strip()
                    if re.search(r'(D\.?O\.?B|License|FIH|F/H|FM|Category)', next_text, re.IGNORECASE):
                        break
                    if next_text:
                        address_parts.append(next_text)

        if address_parts:
            address = ', '.join(address_parts)
            address = re.sub(r'[;:]+', ',', address)
            address = re.sub(r',+', ', ', address)
            return address.strip(', ')
        return None
    
    def _extract_dob(self, text: str) -> Optional[str]:
        """Extract Date of Birth"""
        patterns = [
            r'D\.?O\.?B\.?[:\s]*(\d{1,2}[-+.\s]*\d{1,2}[-+.\s]*\d{4})',
            r'DOB[:\s]*(\d{1,2}[-+.\s]*\d{1,2}[-+.\s]*\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                # Normalize delimiters (-, +, ., space) to '-'
                val = re.sub(r'[-+.\s]+', '-', val)
                return val
        return None
    
    def _extract_blood_group(self, text: str) -> Optional[str]:
        """Extract Blood Group"""
        # Search for B.G or 8.6 (OCR misread) followed by optional colons/spaces and then a BG pattern
        # Handles A+, B+, O+, AB+, etc.
        match = re.search(r'(?:B\.?G\.?|8\.?6)[:\s]*((?:AB|[ABO0])[+-])', text, re.IGNORECASE)
        if match:
            bg = match.group(1).upper().replace('0', 'O')
            if bg in ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']:
                return bg
        
        # Standalone search if label missing
        match = re.search(r'\b((?:AB|[ABO0])[+-])\b', text, re.IGNORECASE)
        if match:
            bg = match.group(1).upper().replace('0', 'O')
            if bg in ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']:
                return bg
        return None
    
    def _extract_license_office(self, texts: List[str]) -> Optional[str]:
        """Extract License Office"""
        full_text = " ".join(texts)
        # Search for License Office followed by optional punctuation and then the name
        match = re.search(r'License\s*Office[:;\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: per-element
        for text in texts:
            if 'License Office' in text:
                parts = re.split(r'[Ll]icense\s*[Oo]ffice[:;\s]*', text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    return parts[1].strip()
        return None
    
    def _extract_fh_name(self, texts: List[str]) -> Optional[str]:
        """Extract Father/Husband Name"""
        full_text = " ".join(texts)
        # Handle variations: FIH Name, F/H Name, FM Name, etc.
        # Allow any characters between F and H/M context
        match = re.search(r'F[/\s\.IM]*?[HM]\s*Name[:\s;]*([A-Z][A-Za-z.\s]+?)(?=\s*(?:Citizenship|Category|D\.?O|Passport|Phone|L47|$))', full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove any leading punctuation often read by OCR (e.g. ; Ganesh)
            name = re.sub(r'^[^A-Z]+', '', name)
            return name

        # Per-element fallback
        for i, text in enumerate(texts):
            if re.search(r'F[/\s\.IM]*?[HM]\s*Name', text, re.IGNORECASE):
                parts = re.split(r'F[/\s\.IM]*?[HM]\s*Name[:\s;]*', text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    name = parts[1].strip()
                    name = re.sub(r'^[^A-Z]+', '', name)
                    name = re.sub(r'\s+(?:Cit|Category|D\.?O|Passport|Phone|L47).*$', '', name, flags=re.IGNORECASE)
                    return name
                if i + 1 < len(texts):
                    next_t = texts[i + 1].strip()
                    if next_t and not re.search(r'(Cit|Category|D\.?O|Passport|Phone|L47)', next_t, re.IGNORECASE):
                        next_t = re.sub(r'^[^A-Z]+', '', next_t)
                        return next_t
        return None
    
    def _extract_citizenship(self, text: str) -> Optional[str]:
        """Extract Citizenship Number"""
        patterns = [
            r'C[li]t[a-z]*s?h?i?p?\s*No\.?:*\s*([\d\-/]+)',
            r'(\d{2}-\d{2}-\d{2}-\d{5})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                citizenship = match.group(1).strip()
                if citizenship and len(citizenship) >= 3:
                    return citizenship
        return None
    
    def _extract_category(self, text: str) -> Optional[str]:
        """Extract License Category (Nepal: A, B, C, D, E, F, G, H, I, J, K)"""
        valid_categories = set('ABCDEFGHIJK')
        # Match Category: followed by valid letters (e.g. A, B or AB)
        match = re.search(r'Categ(?:ory|any):*\s*([A-K\s,]+)', text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            # Extract only valid category letters
            categories = [c.upper() for c in raw if c.upper() in valid_categories]
            
            # Stop before next fields (D.O.I / D.O.E / Passport / etc)
            remaining_text = text[match.end():]
            # If the next word looks like O.I or O.E misread
            if re.match(r'^\s*[\.0oO]?\s*[0oO][IlEe]\b', remaining_text, re.IGNORECASE):
                # Only remove 'D' if there are OTHER categories, or if it's a clear misread
                if len(categories) > 1 and categories[-1] == 'D':
                    categories = categories[:-1]
                # If only 'D' found, we keep it as it's likely both the Category and start of DOI
            
            if categories:
                return ''.join(dict.fromkeys(categories))
        return None
    
    def _extract_doi(self, texts: List[str], full_text: str) -> Optional[str]:
        """Extract Date of Issue"""
        # Very robust patterns to handle merged fields and typos
        patterns = [
            # Standard D.O.I / D.Ol
            r'D\.?O\.?[Il1]\.?[:\s]*(\d{1,2}[-+.\s]*\d{1,2}[-+.\s]*\d{4})',
            # Merged with Category or misread as .OI / .Ol
            r'[\.0oO]\s*[0oO][Il1]\.?[:\s]*(\d{1,2}[-+.\s]*\d{1,2}[-+.\s]*\d{4})',
            # Standalone DOI label
            r'DOI[:\s]*(\d{1,2}[-+.\s]*\d{1,2}[-+.\s]*\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                return re.sub(r'[-+.\s]+', '-', val)

        # Per-element fallback
        for i, text in enumerate(texts):
            if re.search(r'D\.?[0oO][Il1]\b', text, re.IGNORECASE):
                text_to_search = " ".join(texts[i:i+2])
                date_match = re.search(r'(\d{1,2}[-+.\s]*\d{1,2}[-+.\s]*\d{4})', text_to_search)
                if date_match:
                    return re.sub(r'[-+.\s]+', '-', date_match.group(1))
        return None
    
    def _extract_doe(self, text: str) -> Optional[str]:
        """Extract Date of Expiry"""
        patterns = [
            r'D\.?O\.?E\.?:*\s*(\d{1,2}-\d{1,2}-\d{4})',
            r'DOE:*\s*(\d{1,2}-\d{1,2}-\d{4})',
            r'D\.?OE\.?:*\s*(\d{1,2}-\d{1,2}-\d{4})',
            r'D\.?O\.?E\.?:*\s*(\d{1,2}/\d{1,2}/\d{4})',
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
        """Extract Contact/Phone Number"""
        # Standard: number after label
        patterns = [
            r'(?:Contact|Phone)\s*No\.?:*\s*(\d{9,10})',
            r'(?:Contact|Phone)\s*:*\s*(\d{9,10})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Reversed: number before label (OCR sometimes reads number first)
        match = re.search(r'(\d{9,10})\s*(?:Contact|Phone)\s*No\.?:*', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Last resort: find any 10-digit Nepal mobile number (starts with 98)
        match = re.search(r'\b(9[78]\d{8})\b', text)
        if match:
            return match.group(1)
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
    IMAGE_PATH = r"./dataset/license/license_132.jpg"
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