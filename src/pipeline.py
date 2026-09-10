from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from . import config, preprocess
from .classifier import DocumentClassifier
from .extractor import EasyOCRFieldExtractor
from .postprocess import postprocess_fields


def process_document(path: str, classifier: DocumentClassifier, extractor: EasyOCRFieldExtractor) -> Dict[str, Any]:
    pages = preprocess.load_images(path)
    # Simplified: process the first page only. Multi-page PDFs could loop
    # over `pages` and merge/pick the most confident classification.
    image = preprocess.preprocess_pipeline(pages[0])

    # Run OCR once via the extractor's reader so it can double as the
    # keyword-fallback hint for the classifier (avoids running OCR twice).
    raw_boxes = extractor._run_ocr(image)  # noqa: SLF001 - internal reuse by design
    full_text_hint = " ".join(b.text for b in raw_boxes)

    pil_image = preprocess.to_pil(image)
    classification = classifier.classify(pil_image, ocr_text_hint=full_text_hint)

    if classification.label == "UNKNOWN":
        return {
            "document_type": "UNKNOWN",
            "fields": {},
            "confidence": {},
            "classification_confidence": classification.confidence,
        }

    extraction = extractor.extract(image, classification.label)
    validated = postprocess_fields(classification.label, extraction.fields)

    output = {
        "document_type": classification.label,
        "fields": {name: vf.value for name, vf in validated.items()},
        "confidence": {name: vf.confidence for name, vf in validated.items()},
        "classification_confidence": round(classification.confidence, 4),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description="Intelligent Document ID Processing System")
    parser.add_argument("--input", required=True, help="Path to a JPEG/PNG/PDF ID document")
    parser.add_argument("--weights", default="models/classifier.pt", help="Path to trained classifier weights")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR/classification if available")
    args = parser.parse_args()

    classifier = DocumentClassifier(weights_path=args.weights)
    extractor = EasyOCRFieldExtractor(gpu=args.gpu)

    result = process_document(args.input, classifier, extractor)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
