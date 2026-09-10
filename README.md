# Intelligent Document ID Processing System

This project builds an end-to-end document processing pipeline for identity cards such as PAN and Aadhaar. It classifies an input document, extracts key fields with OCR, validates the extracted details, and returns structured JSON output.

## Overview

The system is designed around a three-stage pipeline:

1. Classification
   - Detects whether the document is a PAN, Aadhaar, or unknown.
   - Uses a transfer-learning image classifier built on a pretrained ResNet/ViT backbone.
   - Falls back to keyword-based classification if no trained weights are available.

2. Field extraction
   - Runs OCR on the image using EasyOCR.
   - Maps detected text to document-specific fields using template-based extraction rules.
   - Supports PAN and Aadhaar layouts.

3. Post-processing and validation
   - Normalizes dates and IDs.
   - Cleans OCR confusions such as O -> 0 and I -> 1.
   - Validates PAN format and Aadhaar checksum.
   - Produces confidence scores for extracted values.

## Project structure

```text
id_doc_system_actual/
├── README.md
├── requirements.txt
├── samples/
│   ├── aadhaar_sample.jpg
│   ├── aadar2.png
│   ├── aadar3.jpg
│   ├── liscence.jpg
│   ├── pan1.jpg
│   └── pan_sample.jpg
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── config.py
│   ├── extractor.py
│   ├── pipeline.py
│   ├── postprocess.py
│   ├── preprocess.py
│   └── train_classifier.py
└── venv/
```

### Source files

- `src/classifier.py`  
  Handles document classification and fallback keyword detection.

- `src/extractor.py`  
  Runs OCR and extracts fields like PAN number, Aadhaar number, name, DOB, gender, and address.

- `src/postprocess.py`  
  Validates extracted values, normalizes them, and calculates confidence scores.

- `src/preprocess.py`  
  Loads image/PDF files and applies denoising, skew correction, contrast normalization, and sharpening.

- `src/pipeline.py`  
  Main end-to-end processing entry point.

- `src/train_classifier.py`  
  Training script for the document classifier.

- `src/config.py`  
  Central configuration for labels, field schemas, regex, and confidence weights.

## Features

- Accepts JPEG, PNG, BMP, TIFF, and PDF inputs.
- Handles PAN and Aadhaar document classification.
- Extracts structured fields such as:
  - PAN: `name`, `father_name`, `dob`, `pan_number`
  - Aadhaar: `name`, `dob`, `gender`, `aadhaar_number`, `address`
- Runs validation for:
  - PAN format
  - Aadhaar numeric format
  - Aadhaar Verhoeff checksum
  - date normalization
- Produces JSON output with document type, field values, and confidence data.

## Environment setup instructions and Requirements
1. **Create and activate a virtual environment** (recommended, keeps dependencies isolated):
   ```bash
   python -m venv venv
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # macOS / Linux:
   source venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

Main packages include:

- PyTorch
- TorchVision
- Transformers
- EasyOCR
- OpenCV
- NumPy
- Pillow
- pdf2image
- python-dateutil
- scikit-image

## Quick start

Run the pipeline on a sample image:

```bash
python -m src.pipeline --input samples/pan_sample.jpg
```

Or for Aadhaar:

```bash
python -m src.pipeline --input samples/aadhaar_sample.jpg
```

Or for saving the output in output folder:

```bash
python -m src.pipeline --input samples/aadhaar_sample.jpg > output/aadhaar_output.json
```

### Example output

```json
{
  "document_type": "PAN",
  "fields": {
    "name": "ABC XYZ",
    "dob": "1990-01-01",
    "pan_number": "ABCDE1234F"
  },
  "confidence": {
    "name": 0.82,
    "dob": 0.91,
    "pan_number": 0.96
  },
  "classification_confidence": 0.94
}
```

## Training the classifier

The repository includes a training script for the classification model. To train a classifier from labeled document folders:

```bash
python -m src.train_classifier --data_dir <path_to_dataset> --epochs 15 --backbone resnet18 --out_path models/classifier.pt
```

Expected dataset structure:

```text
<dataset>/
├── train/
│   ├── AADHAAR/
│   ├── PAN/
│   └── UNKNOWN/
└── val/
    ├── AADHAAR/
    ├── PAN/
    └── UNKNOWN/
```

## Notes

- The default classifier checkpoint path is `models/classifier.pt`.
- If no trained weights exist yet, the system still runs using keyword-based classification as a fallback.
- This project uses a practical baseline implementation: EasyOCR + template-based field matching, not a fully trained end-to-end document understanding model.
- The code includes a `LayoutModelFieldExtractor` integration point for future migration to a stronger transformer-based model such as Donut or LayoutLMv3.

## License

This project does not currently include a license file. Please confirm with the project owner before commercial or public reuse.

## Typical use cases

- Automated extraction from scanned identity documents
- Document-type identification for downstream workflows
- OCR cleanup and validation for government ID cards
- Foundation for building a larger document intelligence pipeline
