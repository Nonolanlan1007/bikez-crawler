"""CLIP embedding + category/subject classification.

The embedding model/pretrained-tag choice (``ViT-B-32`` / ``openai``, 512-dim,
L2-normalized) is a fixed contract shared with the parallel model-training task and
must not change independently here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_classifier.joblib"
SUBJECT_MODEL_PATH = MODELS_DIR / "subject_classifier.joblib"

DEFAULT_SUBJECT = "photograph"


@dataclass
class _ClassifierArtifact:
    estimator: Any
    classes: list[str]


def _load_classifier(path: Path) -> _ClassifierArtifact | None:
    if not path.exists():
        return None
    payload: dict[str, Any] = joblib.load(path)
    return _ClassifierArtifact(estimator=payload["estimator"], classes=list(payload["classes"]))


class Classifier:
    """Loads the CLIP backbone once and the optional trained classifier heads.

    When a head artifact is missing (the training task hasn't produced it yet), the
    corresponding prediction falls back to a safe default so the rest of the pipeline
    still runs end-to-end. No code change is needed once the real artifacts appear at
    ``CATEGORY_MODEL_PATH`` / ``SUBJECT_MODEL_PATH`` -- just re-run.
    """

    def __init__(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._category = _load_classifier(CATEGORY_MODEL_PATH)
        self._subject = _load_classifier(SUBJECT_MODEL_PATH)
        if self._category is None:
            logger.warning("%s not found; category will default to None for all images", CATEGORY_MODEL_PATH)
        if self._subject is None:
            logger.warning(
                "%s not found; subject will default to %r for all images", SUBJECT_MODEL_PATH, DEFAULT_SUBJECT
            )

    def embed(self, image: Image.Image) -> np.ndarray[Any, Any]:
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        result: np.ndarray[Any, Any] = features.squeeze(0).cpu().numpy()
        return result

    def classify(self, image: Image.Image) -> tuple[str | None, str]:
        if self._category is None and self._subject is None:
            return None, DEFAULT_SUBJECT

        embedding = self.embed(image).reshape(1, -1)
        category = self._predict(self._category, embedding) if self._category is not None else None
        subject = self._predict(self._subject, embedding) if self._subject is not None else DEFAULT_SUBJECT
        return category, subject

    @staticmethod
    def _predict(artifact: _ClassifierArtifact, embedding: np.ndarray[Any, Any]) -> str:
        probabilities = artifact.estimator.predict_proba(embedding)[0]
        label: str = artifact.classes[int(np.argmax(probabilities))]
        return label
