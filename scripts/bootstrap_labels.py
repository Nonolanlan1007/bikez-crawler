"""Bootstrap category/subject labels for the sampled images via zero-shot CLIP.

Embeds every sampled image with CLIP and scores it against the prompt sets in
``bikez_crawler.training.taxonomy`` (prompt-ensembled, softmax over CLIP's own logit
scale). Writes two manifests:

- ``data/bootstrap_labels_all.csv``: every sampled image with its predicted label and
  confidence for both tasks (useful for the review tool and for judging label quality).
- ``data/bootstrap_labels_highconf.csv``: rows that clear the confidence/margin
  thresholds for at least one task — this is the training set ``train_classifiers.py``
  uses by default. The two tasks are thresholded independently (category confidence is
  usually much lower than subject confidence, an 11-way view/framing call is just
  harder than the roughly-binary studio-vs-real-world call), so a row may carry a label
  for only one of the two; ``train_classifiers.py`` trains each task on the rows that
  have a label for it.

Usage:
    uv run python scripts/sample_images.py     # first, if not already done
    uv run python scripts/bootstrap_labels.py
    uv run python scripts/train_classifiers.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from bikez_crawler.clip_embed import get_shared_embedder
from bikez_crawler.training.common import (
    LabeledRow,
    prompt_prototypes,
    save_embeddings_cache,
    top1_and_margin,
    write_manifest,
    zero_shot_scores,
)
from bikez_crawler.training.taxonomy import CATEGORY_PROMPTS, SUBJECT_PROMPTS

SCRIPT_DIR = Path(__file__).resolve().parent


def load_sampled(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [(row["image_key"], row["local_path"]) for row in reader]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled", type=Path, default=SCRIPT_DIR / "data" / "sampled_images.csv")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / ".cache")
    parser.add_argument("--category-min-prob", type=float, default=0.35)
    parser.add_argument("--category-min-margin", type=float, default=0.08)
    parser.add_argument("--subject-min-prob", type=float, default=0.75)
    parser.add_argument("--subject-min-margin", type=float, default=0.35)
    args = parser.parse_args()

    rows = load_sampled(args.sampled)
    if not rows:
        raise SystemExit(f"no rows in {args.sampled}; run sample_images.py first")

    embedder = get_shared_embedder()
    image_keys = [key for key, _ in rows]
    local_paths = [SCRIPT_DIR / path for _, path in rows]

    embeddings = np.zeros((len(rows), 512), dtype=np.float32)
    batch_size = 32
    for start in tqdm(range(0, len(rows), batch_size), desc="embedding"):
        batch_paths = local_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        embeddings[start : start + len(images)] = embedder.embed_images(images)

    save_embeddings_cache(args.cache_dir, image_keys, embeddings)

    category_labels, category_prototypes = prompt_prototypes(embedder, CATEGORY_PROMPTS)
    subject_labels, subject_prototypes = prompt_prototypes(embedder, SUBJECT_PROMPTS)

    category_probs = zero_shot_scores(embeddings, category_prototypes, embedder.logit_scale)
    subject_probs = zero_shot_scores(embeddings, subject_prototypes, embedder.logit_scale)

    category_top1, category_margin = top1_and_margin(category_probs)
    subject_top1, subject_margin = top1_and_margin(subject_probs)

    category_pred = [category_labels[i] for i in category_probs.argmax(axis=1)]
    subject_pred = [subject_labels[i] for i in subject_probs.argmax(axis=1)]

    category_ok = (category_top1 >= args.category_min_prob) & (category_margin >= args.category_min_margin)
    subject_ok = (subject_top1 >= args.subject_min_prob) & (subject_margin >= args.subject_min_margin)

    all_rows = [
        LabeledRow(
            image_key=image_keys[i],
            local_path=str(local_paths[i].relative_to(SCRIPT_DIR)),
            category=category_pred[i],
            subject=subject_pred[i],
            category_conf=float(category_top1[i]),
            subject_conf=float(subject_top1[i]),
        )
        for i in range(len(rows))
    ]

    highconf_rows = [
        LabeledRow(
            image_key=row.image_key,
            local_path=row.local_path,
            category=row.category if category_ok[i] else "",
            subject=row.subject if subject_ok[i] else "",
            category_conf=row.category_conf,
            subject_conf=row.subject_conf,
        )
        for i, row in enumerate(all_rows)
        if category_ok[i] or subject_ok[i]
    ]

    write_manifest(args.out_dir / "bootstrap_labels_all.csv", all_rows)
    write_manifest(args.out_dir / "bootstrap_labels_highconf.csv", highconf_rows)

    category_kept = sum(1 for r in highconf_rows if r.category)
    subject_kept = sum(1 for r in highconf_rows if r.subject)
    print(f"{len(all_rows)} images scored")
    print(f"{len(highconf_rows)} rows kept overall (label for category and/or subject)")
    print(f"  category: {category_kept} labeled ({category_kept / len(all_rows):.0%})")
    print(f"  subject:  {subject_kept} labeled ({subject_kept / len(all_rows):.0%})")

    print("category distribution (high-confidence subset):")
    for label in category_labels:
        count = sum(1 for r in highconf_rows if r.category == label)
        print(f"  {label:20s} {count}")

    print("subject distribution (high-confidence subset):")
    for label in subject_labels:
        count = sum(1 for r in highconf_rows if r.subject == label)
        print(f"  {label:20s} {count}")


if __name__ == "__main__":
    main()
