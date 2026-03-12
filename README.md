# Nepal Driving License KYC Verification System

A modular and professional system for verifying Nepal Driving Licenses using CNN-based classification and EasyOCR-based data extraction.

## 🚀 Features

- **Automated Classification**: Identifies if an uploaded image is a valid driving license (threshold: 80% confidence).
- **Advanced OCR Pipeline**: Auto-crops, deskews, and enhances images for high-accuracy text extraction.
- **Smart Data Parsing**: Extracts DL Number, Name, DOB, Citizenship Number, Category, and more using refined regex patterns.
- **KYC Verification Page**: A modern Next.js dashboard with:
  - Real-time verification progress steps.
  - User detail comparison vs document data.
  - Expiry date validation and warnings.

## 📂 Project Structure

```text
Document-verification/
├── kyc/                # Next.js Frontend Application
├── kyc_verifier/       # Modular Python Backend Package
│   ├── classifier.py   # CNN Classification Logic
│   ├── ocr.py          # OCR Extraction Engine
│   ├── parser.py       # Nepal License Regex Parser
│   └── verifier.py     # Main Verification Pipeline
├── models/             # Pre-trained ML Models
│   └── model.h5        # License Classification Model
├── test_images/        # Sample images for testing
├── verify_api.py       # Flask API to serve the verification system
├── run_kyc.py          # CLI Runner for local verification
└── requirements.txt    # Python dependencies
```

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend API
```bash
python verify_api.py
```

### 3. Start the Frontend (Next.js)
```bash
cd kyc
npm install
npm run dev
```

## 🧪 Testing

To run a quick verification on a sample image locally:
```bash
python run_kyc.py
```
Check `run_kyc.py` to change the sample image or model path.

## ⚖️ License
This project is developed for KYC verification purposes in Nepal.
