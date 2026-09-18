"""Merge human corrections (from the review tool) onto the bootstrap labels, then
retrain both classifiers on the merged set.

Human corrections always win over the zero-shot bootstrap label for the same image.

Usage:
    uv run python scripts/generate_review_page.py    # review images, click "export corrections.csv"
    mv ~/Downloads/corrections.csv scripts/data/corrections.csv
    uv run python scripts/retrain_from_corrections.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bikez_crawler.training.common import LabeledRow, read_manifest, write_manifest

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=SCRIPT_DIR / "data" / "bootstrap_labels_highconf.csv")
    parser.add_argument("--corrections", type=Path, default=SCRIPT_DIR / "data" / "corrections.csv")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "data" / "retrain_manifest.csv")
    args = parser.parse_args()

    if not args.corrections.exists():
        raise SystemExit(f"no corrections found at {args.corrections}; run the review tool first")

    base_rows = read_manifest(args.base) if args.base.exists() else []
    correction_rows = read_manifest(args.corrections)

    merged: dict[str, LabeledRow] = {row.image_key: row for row in base_rows}
    for row in correction_rows:
        row.category_conf = 1.0
        row.subject_conf = 1.0
        merged[row.image_key] = row  # human corrections win

    write_manifest(args.out, list(merged.values()))
    print(
        f"merged {len(base_rows)} bootstrap + {len(correction_rows)} corrections "
        f"-> {len(merged)} rows at {args.out}"
    )
    print(f"now run: uv run python scripts/train_classifiers.py --manifest {args.out}")


if __name__ == "__main__":
    main()
