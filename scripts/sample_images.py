"""Sample a broad, diverse set of raw gallery images from S3 for bootstrapping labels.

Picks a random set of bike tags from Mongo, then grabs a few images per bike from the
raw S3 bucket, so the sample spans many different bikes/brands/view-angles rather than
one bike's whole gallery. Downloads images to a local cache dir and writes a manifest
CSV that ``bootstrap_labels.py`` reads next.

Usage:
    uv run python scripts/sample_images.py
    uv run python scripts/sample_images.py --bikes 500 --per-bike-max 2 --target-count 800
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

from tqdm import tqdm

from bikez_crawler.config import load_settings
from bikez_crawler.db import SyncDatabase
from bikez_crawler.s3 import S3Client

SCRIPT_DIR = Path(__file__).resolve().parent


def local_path_for(images_dir: Path, image_key: str) -> Path:
    digest = hashlib.sha1(image_key.encode()).hexdigest()[:16]
    suffix = Path(image_key).suffix or ".jpg"
    return images_dir / f"{digest}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bikes", type=int, default=350, help="distinct bike tags to sample from")
    parser.add_argument("--per-bike-max", type=int, default=2, help="max images kept per sampled bike")
    parser.add_argument("--target-count", type=int, default=500, help="stop once this many images are collected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "data" / "sampled_images.csv")
    parser.add_argument("--images-dir", type=Path, default=SCRIPT_DIR / ".cache" / "images")
    args = parser.parse_args()

    random.seed(args.seed)
    settings = load_settings()
    db = SyncDatabase(settings.mongodb_url)
    s3 = S3Client(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    )

    tags = [
        doc["tag"] for doc in db.bikes.aggregate([{"$sample": {"size": args.bikes}}, {"$project": {"tag": 1}}])
    ]
    print(f"sampled {len(tags)} bike tags")

    args.images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []  # image_key, local_path, bike_tag

    for tag in tqdm(tags, desc="bikes"):
        if len(rows) >= args.target_count:
            break
        keys = list(s3.list_keys(settings.s3_raw_bucket, f"bikes/{tag}/"))
        if not keys:
            continue
        random.shuffle(keys)
        for key in keys[: args.per_bike_max]:
            local_path = local_path_for(args.images_dir, key)
            if not local_path.exists():
                data = s3.download(settings.s3_raw_bucket, key)
                local_path.write_bytes(data)
            rows.append((key, str(local_path.relative_to(SCRIPT_DIR)), tag))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_key", "local_path", "bike_tag"])
        writer.writerows(rows)

    distinct_bikes = len({row[2] for row in rows})
    print(f"downloaded {len(rows)} images across {distinct_bikes} bikes -> {args.out}")


if __name__ == "__main__":
    main()
