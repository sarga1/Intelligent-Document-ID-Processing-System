from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from . import config

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    raw_scores: dict


def build_model(backbone: str = "resnet18", num_classes: int = 3) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


class DocumentClassifier:

    def __init__(
        self,
        weights_path: str = "models/classifier.pt",
        backbone: str = "resnet18",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = config.CLASS_LABELS
        self.model = None

        if os.path.exists(weights_path):
            self.model = build_model(backbone, num_classes=len(self.labels))
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()

    def _predict_cnn(self, pil_image) -> ClassificationResult:
        tensor = _TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        confidence = float(probs[idx])

        if confidence < config.CLASSIFIER_CONF_THRESHOLD:
            label = "UNKNOWN"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            raw_scores={self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
        )

    @staticmethod
    def _predict_keywords(ocr_text: str) -> ClassificationResult:
        """Fallback: simple keyword scoring over OCR'd full-page text."""
        text = ocr_text.lower()
        scores = {"AADHAAR": 0, "PAN": 0}
        for label, keywords in config.CLASSIFICATION_KEYWORDS.items():
            scores[label] = sum(1 for kw in keywords if kw in text)

        best_label = max(scores, key=scores.get)
        if scores[best_label] == 0:
            return ClassificationResult("UNKNOWN", 0.0, {"AADHAAR": 0.0, "PAN": 0.0, "UNKNOWN": 1.0})

        total = sum(scores.values()) or 1
        confidence = scores[best_label] / total
        return ClassificationResult(
            label=best_label,
            confidence=confidence,
            raw_scores={k: v / total for k, v in scores.items()},
        )

    def classify(self, pil_image, ocr_text_hint: str = "") -> ClassificationResult:
        if self.model is not None:
            return self._predict_cnn(pil_image)
        return self._predict_keywords(ocr_text_hint)
