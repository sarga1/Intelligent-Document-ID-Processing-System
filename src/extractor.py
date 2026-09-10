"""
Stage 2: Model-Based Key Field Extraction.
detects every text region with its bounding box and confidence in one pass;
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config


@dataclass
class TextBox:
    text: str
    confidence: float
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]

    @property
    def y_center(self) -> float:
        ys = [pt[1] for pt in self.bbox]
        return sum(ys) / len(ys)

    @property
    def x_left(self) -> float:
        return min(pt[0] for pt in self.bbox)


@dataclass
class ExtractedField:
    value: str
    confidence: float


@dataclass
class ExtractionResult:
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    raw_text_boxes: List[TextBox] = field(default_factory=list)


class FieldExtractor(ABC):
    """Common interface so extraction backends are interchangeable."""

    @abstractmethod
    def extract(self, image: np.ndarray, document_type: str) -> ExtractionResult:
        ...


# ---------------------------------------------------------------------------
# Baseline: EasyOCR (deep-learning OCR) + anchor-based field assignment
# ---------------------------------------------------------------------------
class EasyOCRFieldExtractor(FieldExtractor):
    """
    mapping raw text boxes to semantic field
    names -- uses regex + document-template position rules rather than a
    separate detector, since PAN/Aadhaar layouts are fixed enough that this
    is reliable.
    """

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        import easyocr  # imported lazily so the module loads even without GPU/easyocr installed

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def _run_ocr(self, image: np.ndarray) -> List[TextBox]:
        results = self.reader.readtext(image)  # [(bbox, text, conf), ...]
        return [TextBox(text=t.strip(), confidence=float(c), bbox=b) for b, t, c in results if t.strip()]

    def extract(self, image: np.ndarray, document_type: str) -> ExtractionResult:
        boxes = self._run_ocr(image)
        boxes.sort(key=lambda b: (round(b.y_center / 15), b.x_left))  # reading order

        if document_type == "PAN":
            fields = self._extract_pan(boxes)
        elif document_type == "AADHAAR":
            fields = self._extract_aadhaar(boxes)
        else:
            fields = {}

        return ExtractionResult(fields=fields, raw_text_boxes=boxes)

    # -- Template-specific anchor logic -----------------------------------
    def _extract_pan(self, boxes: List[TextBox]) -> Dict[str, ExtractedField]:
        fields: Dict[str, ExtractedField] = {}
        full_lines = [b.text for b in boxes]

        # PAN number: unique 10-char alphanumeric pattern, unlikely to
        # collide with anything else on the card.
        for b in boxes:
            m = re.search(config.PAN_REGEX, b.text.upper().replace(" ", ""))
            if m:
                fields["pan_number"] = ExtractedField(value=m.group(0), confidence=b.confidence)
                break

        # DOB: first line matching a date pattern.
        for b in boxes:
            if re.search(config.DATE_REGEX, b.text):
                fields["dob"] = ExtractedField(value=b.text, confidence=b.confidence)
                break

        # Name: PAN cards print "Name" as the first all-caps line following
        # the header, and "Father's Name" as the line after it. We take the
        # line preceding a line containing "father" as the name; if that
        # anchor isn't found, fall back to the first long all-caps line.
        father_idx = next(
            (i for i, t in enumerate(full_lines) if "father" in t.lower()), None
        )
        if father_idx is not None and father_idx > 0:
            b = boxes[father_idx - 1]
            fields["name"] = ExtractedField(value=b.text.strip(), confidence=b.confidence)
        else:
            candidate = next(
                (b for b in boxes if b.text.isupper() and len(b.text.split()) >= 2 and b.text.isalpha() is False
                 and re.fullmatch(r"[A-Z ]{4,}", b.text)),
                None,
            )
            if candidate:
                fields["name"] = ExtractedField(value=candidate.text.strip(), confidence=candidate.confidence)

        return fields

    def _extract_aadhaar(self, boxes: List[TextBox]) -> Dict[str, ExtractedField]:
        fields: Dict[str, ExtractedField] = {}

        # Aadhaar number: 12 digits, often space-grouped as 4-4-4.
        for b in boxes:
            digits_only = re.sub(r"\D", "", b.text)
            if len(digits_only) == 12:
                formatted = f"{digits_only[0:4]} {digits_only[4:8]} {digits_only[8:12]}"
                fields["aadhaar_number"] = ExtractedField(value=formatted, confidence=b.confidence)
                break

        # DOB / Year of Birth
        dob_idx = None
        for i, b in enumerate(boxes):
            if re.search(config.DATE_REGEX, b.text):
                fields["dob"] = ExtractedField(value=b.text, confidence=b.confidence)
                dob_idx = i
                break
            yob_match = re.search(config.YOB_REGEX, b.text.lower())
            if yob_match:
                fields["dob"] = ExtractedField(value=yob_match.group(1), confidence=b.confidence)
                dob_idx = i
                break

        # Gender
        for b in boxes:
            low = b.text.lower()
            if "male" in low or "female" in low:
                fields["gender"] = ExtractedField(
                    value="Female" if "female" in low else "Male", confidence=b.confidence
                )
                break

        # Name: by UIDAI convention, the name line sits directly above the
        # DOB/YOB line.
        if dob_idx is not None and dob_idx > 0:
            b = boxes[dob_idx - 1]
            if re.fullmatch(r"[A-Za-z .]{4,}", b.text):
                fields["name"] = ExtractedField(value=b.text.strip(), confidence=b.confidence)

        return fields


# ---------------------------------------------------------------------------
# Stronger backend interface: Donut / LayoutLMv3 (integration point)
# ---------------------------------------------------------------------------
class LayoutModelFieldExtractor(FieldExtractor):

    def __init__(self, checkpoint_path: str):
        raise NotImplementedError(
            
        )

    def extract(self, image: np.ndarray, document_type: str) -> ExtractionResult:
        raise NotImplementedError
