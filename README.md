# Document Verification System

This project classifies images into "license" or "not_license" and extracts structured data from driving licenses using OCR.

## Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, if using other OCR backends, but `easyocr` handles most dependencies automatically)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd Document-verification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Training/Building the Model (Optional)
If you need to retrain the model with your own dataset, use:
```bash
python build.py
```
*Note: Ensure your dataset is organized in the `dataset/` directory with `license/` and `not_license/` subfolders.*

### 2. Running Verification
To process an image and extract license information:
```bash
python t1.py
```
You can modify the `IMG_PATH` variable in `t1.py` to point to the image you want to verify.

## Project Structure
- `t1.py`: Main script for classification and OCR extraction.
- `build.py`: Script to train the CNN classifier (`model.h5`).
- `model.h5`: Pre-trained Keras model for license detection.
- `dataset/`: Folder containing training images.
