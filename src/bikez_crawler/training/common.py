"""Shared helpers for the bootstrap/train/review/retrain scripts in ``scripts/``.

Everything here is CPU-only: frozen CLIP embeddings (512-dim) feed a small
scikit-learn classifier head. Nothing in this module is imported by the runtime crawler
or image pipeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from bikez_crawler.clip_embed import CLIP_MODEL_NAME, CLIP_PRETRAINED, ClipEmbedder

MANIFEST_FIELDS = ["image_key", "local_path", "category", "subject", "category_conf", "subject_conf"]


@dataclass
class LabeledRow:
    """One image's labels. ``category``/``subject`` are "" when that task's zero-shot
    prediction didn't clear its confidence threshold — the two tasks are filtered
    independently, so a row can carry a label for one task but not the other."""

    image_key: str
    local_path: str
    category: str
    subject: str
    category_conf: float = 1.0
    subject_conf: float = 1.0


def read_manifest(path: Path) -> list[LabeledRow]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            LabeledRow(
                image_key=row["image_key"],
                local_path=row["local_path"],
                category=row["category"],
                subject=row["subject"],
                category_conf=float(row.get("category_conf") or 1.0),
                subject_conf=float(row.get("subject_conf") or 1.0),
            )
            for row in reader
        ]


def write_manifest(path: Path, rows: list[LabeledRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_key": row.image_key,
                    "local_path": row.local_path,
                    "category": row.category,
                    "subject": row.subject,
                    "category_conf": row.category_conf,
                    "subject_conf": row.subject_conf,
                }
            )


def prompt_prototypes(embedder: ClipEmbedder, prompts_by_label: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
    """Mean-pooled, re-normalized text embedding per label (prompt ensembling).

    Returns (labels_in_order, prototypes[n_labels, 512]).
    """
    labels = list(prompts_by_label.keys())
    prototypes = np.zeros((len(labels), 512), dtype=np.float32)
    for i, label in enumerate(labels):
        text_embs = embedder.embed_text(prompts_by_label[label])
        mean = text_embs.mean(axis=0)
        prototypes[i] = mean / np.linalg.norm(mean)
    return labels, prototypes


def zero_shot_scores(
    image_embeddings: np.ndarray, prototypes: np.ndarray, logit_scale: float
) -> np.ndarray:
    """Softmax class probabilities from cosine similarity, CLIP-scaled. Shape (n, n_labels)."""
    logits = logit_scale * (image_embeddings @ prototypes.T)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return cast(np.ndarray, exp / exp.sum(axis=1, keepdims=True))


def top1_and_margin(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (top1 probability, margin between top1 and top2)."""
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    return top1, top1 - top2


def fit_classifier(embeddings: np.ndarray, labels: list[str]) -> LogisticRegression:
    """Small, fast linear head on frozen 512-dim CLIP embeddings."""
    estimator = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    estimator.fit(embeddings, labels)
    return estimator


def save_classifier_artifact(path: Path, estimator: LogisticRegression) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "clip_model": CLIP_MODEL_NAME,
        "clip_pretrained": CLIP_PRETRAINED,
        "estimator": estimator,
        "classes": list(estimator.classes_),
    }
    joblib.dump(artifact, path)


def load_classifier_artifact(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", joblib.load(path))


def embeddings_cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "embeddings.npy", cache_dir / "keys.json"


def load_embeddings_cache(cache_dir: Path) -> tuple[list[str], np.ndarray] | None:
    emb_path, keys_path = embeddings_cache_paths(cache_dir)
    if not emb_path.exists() or not keys_path.exists():
        return None
    keys = json.loads(keys_path.read_text())
    embeddings = np.load(emb_path)
    return keys, embeddings


def save_embeddings_cache(cache_dir: Path, keys: list[str], embeddings: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path, keys_path = embeddings_cache_paths(cache_dir)
    np.save(emb_path, embeddings)
    keys_path.write_text(json.dumps(keys))
