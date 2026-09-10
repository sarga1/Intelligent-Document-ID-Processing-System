# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
CLASS_LABELS = ["AADHAAR", "PAN", "UNKNOWN"]

# Below this softmax confidence, the classifier result is downgraded to
# UNKNOWN rather than trusted outright.
CLASSIFIER_CONF_THRESHOLD = 0.55

# Keyword fallback used when no trained weights are available yet, or as a
# secondary signal to sanity-check the CNN/ViT prediction.
CLASSIFICATION_KEYWORDS = {
    "AADHAAR": ["aadhaar", "uidai", "government of india", "unique identification"],
    "PAN": ["income tax department", "permanent account number", "pan card"],
}

# ---------------------------------------------------------------------------
# Field schema per document type
# ---------------------------------------------------------------------------
FIELD_SCHEMA = {
    "PAN": ["name", "father_name", "dob", "pan_number"],
    "AADHAAR": ["name", "dob", "gender", "aadhaar_number", "address"],
}

# ---------------------------------------------------------------------------
# Regex patterns for post-processing / validation
# ---------------------------------------------------------------------------
PAN_REGEX = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"
AADHAAR_REGEX = r"\d{4}\s?\d{4}\s?\d{4}"
DATE_REGEX = r"(\d{1,2}[/\-. ]\d{1,2}[/\-. ]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
# Aadhaar cards often print "Year of Birth : 1994" instead of a full DOB
YOB_REGEX = r"(?:year of birth|yob)\D{0,5}(\d{4})"

# Common OCR character confusions, applied only to strictly numeric ID
# fields (never to name fields, to avoid corrupting real names).
OCR_NUMERIC_FIXUPS = {
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "B": "8",
}
CONF_WEIGHTS = {
    "ocr": 0.5,
    "regex": 0.3,
    "checksum": 0.2,
}
