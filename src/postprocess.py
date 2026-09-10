"""
- Regex format validation for PAN / Aadhaar numbers.
- Aadhaar checksum validation via the Verhoeff algorithm (used by UIDAI).
- Date normalization to ISO 8601 (YYYY-MM-DD).
- Targeted OCR noise cleanup for numeric ID fields.
- Weighted confidence scoring combining OCR confidence + validation signals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from dateutil import parser as date_parser

from . import config
from .extractor import ExtractedField

_D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_checksum_valid(number: str) -> bool:
    """Return True if `number` (digits only) passes the Verhoeff checksum."""
    digits = re.sub(r"\D", "", number)
    if len(digits) != 12:
        return False
    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = _D_TABLE[c][_P_TABLE[i % 8][int(digit)]]
    return c == 0

def is_valid_pan(value: str) -> bool:
    return bool(re.fullmatch(config.PAN_REGEX, value.strip().upper().replace(" ", "")))


def is_valid_aadhaar_format(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return bool(re.fullmatch(r"\d{12}", digits))

def clean_numeric_id(value: str) -> str:
    """Apply OCR character fixups (O->0, I->1, etc.) to a purely numeric ID
    string (e.g. Aadhaar, which has no letters, so every character can be
    safely coerced towards a digit)."""
    cleaned = value
    for wrong, right in config.OCR_NUMERIC_FIXUPS.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned

_LETTER_FIXUPS = {"0": "O", "1": "I", "5": "S", "8": "B"}


def clean_pan(value: str) -> str:
    value = value.strip().upper().replace(" ", "")
    chars = list(value)

    for i, ch in enumerate(chars):
        if i < 5 or i == 9:  # must be a letter
            if ch in _LETTER_FIXUPS:
                chars[i] = _LETTER_FIXUPS[ch]
        elif 5 <= i < 9:  # must be a digit
            if ch in config.OCR_NUMERIC_FIXUPS:
                chars[i] = config.OCR_NUMERIC_FIXUPS[ch]

    return "".join(chars)


def normalize_date(value: str) -> Optional[str]:
    value = value.strip()

    # Year-of-birth only (Aadhaar sometimes prints just the year)
    if re.fullmatch(r"\d{4}", value):
        return f"{value}-01-01"

    try:
        # dayfirst=True matches Indian DD/MM/YYYY convention
        parsed = date_parser.parse(value, dayfirst=True, fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def score_confidence(ocr_conf: float, regex_pass: bool, checksum_pass: Optional[bool]) -> float:
    w = config.CONF_WEIGHTS
    if checksum_pass is None:
        regex_weight = w["regex"] + w["checksum"]
        score = w["ocr"] * ocr_conf + regex_weight * (1.0 if regex_pass else 0.0)
    else:
        score = (
            w["ocr"] * ocr_conf
            + w["regex"] * (1.0 if regex_pass else 0.0)
            + w["checksum"] * (1.0 if checksum_pass else 0.0)
        )
    return round(min(max(score, 0.0), 1.0), 4)


@dataclass
class ValidatedField:
    value: str
    confidence: float


def postprocess_fields(
    document_type: str, raw_fields: Dict[str, ExtractedField]
) -> Dict[str, ValidatedField]:
    result: Dict[str, ValidatedField] = {}

    for field_name, extracted in raw_fields.items():
        value = extracted.value
        ocr_conf = extracted.confidence
        regex_pass = True
        checksum_pass: Optional[bool] = None

        if field_name == "pan_number":
            value = clean_pan(value)
            regex_pass = is_valid_pan(value)

        elif field_name == "aadhaar_number":
            value = clean_numeric_id(value)
            regex_pass = is_valid_aadhaar_format(value)
            if regex_pass:
                checksum_pass = verhoeff_checksum_valid(value)
            # re-format with spaces for readability once validated
            digits = re.sub(r"\D", "", value)
            if len(digits) == 12:
                value = f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"

        elif field_name == "dob":
            normalized = normalize_date(value)
            regex_pass = normalized is not None
            if normalized:
                value = normalized

        elif field_name in ("name", "father_name"):
            value = re.sub(r"\s+", " ", value).strip()
            regex_pass = bool(re.fullmatch(r"[A-Za-z .'-]{2,60}", value))

        elif field_name == "gender":
            regex_pass = value in ("Male", "Female")

        confidence = score_confidence(ocr_conf, regex_pass, checksum_pass)
        result[field_name] = ValidatedField(value=value, confidence=confidence)

    return result
