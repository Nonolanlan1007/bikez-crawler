# Image classifier training tooling

Offline, CPU-only tooling to bootstrap and improve the `category`/`subject` image
classifiers used by the image pipeline. Not part of the deployed service — run these
manually from `packages/bikez-crawler/`.

Label taxonomy and prompts live in `src/bikez_crawler/training/taxonomy.py` — see its
docstring for the categories and how to revise them. CLIP loading/embedding is shared
via `src/bikez_crawler/clip_embed.py`.

## 1. Bootstrap: sample, zero-shot label, and train

```sh
uv run python scripts/sample_images.py        # downloads a diverse sample to .cache/images/
uv run python scripts/bootstrap_labels.py     # zero-shot CLIP labels -> data/bootstrap_labels_*.csv
uv run python scripts/train_classifiers.py    # trains + writes ../models/*.joblib
```

`bootstrap_labels.py` keeps only high-confidence zero-shot predictions (per-task
probability + margin thresholds, see `--help`) as the initial training set, so the first
classifiers aren't trained on noisy guesses.

## 2. Review and correct labels

```sh
uv run python scripts/generate_review_page.py
open scripts/review/review.html
```

Static page, no server: shows each sampled image with its predicted labels (lowest
confidence first), lets you correct them via dropdowns. Progress is kept in the
browser's localStorage as you go. Click "export corrections.csv" to download your
reviewed labels, then move the file to `scripts/data/corrections.csv`.

## 3. Retrain from corrections

```sh
uv run python scripts/retrain_from_corrections.py
uv run python scripts/train_classifiers.py --manifest data/retrain_manifest.csv
```

Merges your corrections onto the bootstrap manifest (corrections win) and retrains both
classifiers the same way, overwriting `../models/*.joblib`.

## Notes

- `.cache/` (downloaded images, CLIP embeddings) and `data/` (label manifests) are
  regenerable and gitignored — rerun the scripts above to reproduce them.
- All scripts run entirely on CPU: CLIP is frozen, only a small `LogisticRegression`
  head is fit per task.
