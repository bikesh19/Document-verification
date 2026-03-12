import os
import easyocr
from typing import List
from .preprocessor import ImagePreprocessor

class OCREngine:
    """Extract text from license images using EasyOCR"""
    
    def __init__(self, languages=['en']):
        print("Initializing OCR engine...")
        self.reader = easyocr.Reader(languages, verbose=False)
        print("[OK] OCR engine ready")
    
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
            print(f"  [WARN] Preprocessing failed ({e}), using original image")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        # Pick the version with more extracted text
        orig_len = sum(len(t) for t in original_texts)
        prep_len = sum(len(t) for t in preprocessed_texts)

        if prep_len > orig_len:
            print(f"  [OK] Using preprocessed OCR ({len(preprocessed_texts)} elements vs {len(original_texts)} original)")
            return preprocessed_texts
        else:
            print(f"  [OK] Using original OCR ({len(original_texts)} elements, better than preprocessed)")
            return original_texts
