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
## Model selection and justification

**Classification — ResNet18 / ViT-B16 (transfer learning)**
A pretrained ImageNet backbone was chosen over training a CNN from scratch because the classification task only has 3 classes (AADHAAR, PAN, UNKNOWN) and realistically small amounts of labeled data are available. Transfer learning lets the model reuse low-level features (edges, color blocks, layout structure) learned from millions of ImageNet images, so it converges faster and generalizes better with a small dataset than a from-scratch CNN would. ResNet18 is the default because it is lightweight enough to train and run on CPU in reasonable time; ViT-B16 is offered as a swap-in option (`--backbone vit_b_16`) for cases where more training data and compute are available, since transformers tend to outperform CNNs at larger data scales.
*Why not a rule-based classifier alone?* Keyword matching on OCR text (the current fallback, used automatically when no `.pt` weights exist) is fast and needs no training, but it is brittle: it depends entirely on OCR reading the header text correctly, which fails under glare, skew, or poor lighting. The CNN/ViT path is more robust because it can key off visual layout and color cues even when individual characters are unreadable.

**Field extraction — EasyOCR (CRAFT detector + CRNN recognizer)**
EasyOCR was chosen as the baseline OCR engine because it is a genuinely deep-learning-based pipeline (not a rule-based/template OCR like classic Tesseract page-segmentation), it ships pretrained and needs no custom training to get first results, and it returns both text and per-box confidence scores, which the post-processing stage relies on for its confidence formula. Detection (CRAFT) and recognition (CRNN) are both neural, so "field detection and OCR" are already handled as one learned pass over the image, per the assignment's requirement — the remaining step (mapping raw OCR boxes to named fields) uses lightweight regex/position rules rather than a separate model, because PAN and Aadhaar layouts are fixed enough that this is reliable without extra training data.
*Upgrade path considered:* Donut (OCR-free image→JSON transformer) and LayoutLMv3 (joint text+layout+image transformer) were evaluated as stronger alternatives — both are documented as drop-in replacements via the `FieldExtractor` interface in `extractor.py` (see `LayoutModelFieldExtractor`) — but were not used as the default because they require fine-tuning on a labeled PAN/Aadhaar dataset to be reliable, which was out of scope for this baseline.

**Post-processing — regex + Verhoeff checksum (not a model)**
Format validation and the Aadhaar checksum are deterministic algorithms, not learned models, because PAN and Aadhaar formats are officially fixed and documented (UIDAI publishes the Verhoeff checksum spec). Using a rule-based check here is more reliable and auditable than training a classifier to "detect valid numbers," and it gives an exact pass/fail signal that feeds directly into the confidence score.

## Assumptions, limitations, and known failure cases

**Assumptions**
- Input documents are near-frontal photos or scans (not extreme perspective angles or partial crops).
- PAN and Aadhaar cards follow the standard, current layout conventions (e.g. Aadhaar name appears directly above the DOB/YOB line; PAN name appears directly above "Father's Name").
- English text only — Aadhaar's regional-language name line (printed above the English name on real cards) is not parsed.
- One document per image; multi-page PDFs are processed page-by-page but not cross-referenced.

**Known limitations**
- No trained classifier checkpoint ships with this repo — classification runs on the keyword fallback until `train_classifier.py` is run on a labeled dataset, so classification accuracy is limited by OCR quality until then.
- Field extraction uses fixed anchor rules (regex + relative position), not a learned key-value model, so it can silently produce wrong field assignments if a real card's layout differs from the assumed template (e.g. reordered fields, an unusual print layout, or a redesigned card).
- Address extraction is not implemented for Aadhaar in the current baseline (listed as optional in the spec).
- Severe image degradation is a hard limit: no OCR or classifier can recover text destroyed by heavy blur, extreme low light, or very low resolution. `preprocess.quality_report()` detects and rejects these cases early rather than returning a wrong answer.

**Known failure cases**
- *Heavy blur / motion blur*: OCR confidence drops sharply below a blur-score threshold; the pipeline now returns `"error": "image_quality_too_low"` instead of guessing.
- *Glare or reflection across the header text*: can break the keyword-fallback classifier specifically, since it depends on reading exact phrases like "INCOME TAX DEPARTMENT."
- *Non-standard or damaged cards*: if the name/DOB anchor lines aren't found (e.g. "Father's Name" label missing or misread), the corresponding field is simply omitted from the output rather than guessed.
- *Extreme skew (>45°) or upside-down photos*: the current deskew logic (Hough-line based) is tuned for moderate tilts and may not fully correct very large rotation angles.
- *Regional-language-only text regions*: EasyOCR is configured for English (`languages=["en"]`); Hindi/regional script on the card is not reliably extracted.

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
