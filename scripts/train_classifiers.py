"""Train the category and subject classifiers from a labeled manifest and export the
fixed-contract joblib artifacts consumed by the image pipeline.

Works for both the initial bootstrap manifest and a corrected/retrain manifest — it just
needs a CSV with ``image_key, local_path, category, subject`` columns (see
``bikez_crawler.training.common.LabeledRow``).

Usage:
    uv run python scripts/train_classifiers.py
    uv run python scripts/train_classifiers.py --manifest data/retrain_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from bikez_crawler.clip_embed import get_shared_embedder
from bikez_crawler.training.common import (
    LabeledRow,
    fit_classifier,
    load_embeddings_cache,
    read_manifest,
    save_classifier_artifact,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models"


def embeddings_for(rows: list[LabeledRow], cache_dir: Path) -> np.ndarray:
    cached = load_embeddings_cache(cache_dir)
    if cached is not None:
        keys, embeddings = cached
        index = {key: i for i, key in enumerate(keys)}
        if all(row.image_key in index for row in rows):
            return np.stack([embeddings[index[row.image_key]] for row in rows])

    embedder = get_shared_embedder()
    images = [Image.open(SCRIPT_DIR / row.local_path).convert("RGB") for row in rows]
    return embedder.embed_images(images)


def train_task(embeddings: np.ndarray, labels: list[str], name: str) -> None:
    unique = sorted(set(labels))
    if len(unique) < 2:
        raise SystemExit(f"{name}: need at least 2 distinct labels to train, got {unique}")

    counts = {label: labels.count(label) for label in unique}
    if min(counts.values()) >= 2:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        sanity_estimator = fit_classifier(x_train, y_train)
        accuracy = sanity_estimator.score(x_test, y_test)
        print(f"{name}: held-out accuracy on {len(y_test)} samples = {accuracy:.2f}")
    else:
        print(f"{name}: too few samples per class for a held-out split, skipping sanity check")

    estimator = fit_classifier(embeddings, labels)
    out_path = MODELS_DIR / f"{name}_classifier.joblib"
    save_classifier_artifact(out_path, estimator)
    print(f"{name}: wrote {out_path} ({len(labels)} samples, classes={list(estimator.classes_)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SCRIPT_DIR / "data" / "bootstrap_labels_highconf.csv")
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / ".cache")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"no rows in {args.manifest}")

    embeddings = embeddings_for(rows, args.cache_dir)

    category_idx = [i for i, row in enumerate(rows) if row.category]
    subject_idx = [i for i, row in enumerate(rows) if row.subject]

    train_task(embeddings[category_idx], [rows[i].category for i in category_idx], "category")
    train_task(embeddings[subject_idx], [rows[i].subject for i in subject_idx], "subject")


if __name__ == "__main__":
    main()
