import re
from datetime import datetime
from typing import Dict, Optional, List

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
        # Relaxing [A-Z] requirement to handle any case
        match = re.search(r'\bName:*\s+([A-Za-z\s.]+)', full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove trailing field labels (B.G, Address, DOB, etc.)
            name = re.sub(r'\s+(?:B\.?G\.?|Address|D\.?O\.?B|FIH|F/H|Category).*$', '', name, flags=re.IGNORECASE)
            if name:
                return name.strip()

        # Fallback: per-element search
        for i, text in enumerate(texts):
            # Case-insensitive check for 'Name' but NOT 'F/H Name'
            if re.search(r'\bName\b', text, re.IGNORECASE) and not re.search(r'F[/\s\.IM]*?[HM]\s*Name', text, re.IGNORECASE):
                parts = re.split(r'Name:*\s*', text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    name = parts[1].strip()
                    return name
                # Name might be in next element
                elif i + 1 < len(texts):
                    next_t = texts[i + 1].strip()
                    if next_t and ':' not in next_t:
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
            if re.search(r'License\s*Office', text, re.IGNORECASE):
                parts = re.split(r'License\s*Office[:;\s]*', text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    return parts[1].strip()
        return None
    
    def _extract_fh_name(self, texts: List[str]) -> Optional[str]:
        """Extract Father/Husband Name"""
        full_text = " ".join(texts)
        # Handle variations: FIH Name, F/H Name, FM Name, etc.
        # Relaxing [A-Z] requirement
        match = re.search(r'F[/\s\.IM]*?[HM]\s*Name[:\s;]*([A-Za-z\.\s]+?)(?=\s*(?:Citizenship|Category|D\.?O|Passport|Phone|L47|$))', full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove any leading punctuation often read by OCR (e.g. ; Ganesh)
            name = re.sub(r'^[^A-Za-z]+', '', name)
            return name

        # Per-element fallback
        for i, text in enumerate(texts):
            if re.search(r'F[/\s\.IM]*?[HM]\s*Name', text, re.IGNORECASE):
                parts = re.split(r'F[/\s\.IM]*?[HM]\s*Name[:\s;]*', text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    name = parts[1].strip()
                    name = re.sub(r'^[^A-Za-z]+', '', name)
                    name = re.sub(r'\s+(?:Cit|Category|D\.?O|Passport|Phone|L47).*$', '', name, flags=re.IGNORECASE)
                    return name
                if i + 1 < len(texts):
                    next_t = texts[i + 1].strip()
                    if next_t and not re.search(r'(Cit|Category|D\.?O|Passport|Phone|L47)', next_t, re.IGNORECASE):
                        next_t = re.sub(r'^[^A-Za-z]+', '', next_t)
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
                    validation['age_warning'] = "[WARN] Under 18 years old"
                else:
                    validation['age_status'] = "[OK] Legal age"
            except:
                validation['dob_error'] = "Invalid DOB format"
        
        # Validate expiry
        doe = self.extracted_data.get('date_of_expiry')
        if doe:
            try:
                expiry_date = datetime.strptime(doe, '%d-%m-%Y')
                if expiry_date < datetime.now():
                    validation['expiry_status'] = "[FAIL] License EXPIRED"
                else:
                    days_left = (expiry_date - datetime.now()).days
                    validation['expiry_status'] = f"[OK] Valid ({days_left} days remaining)"
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
