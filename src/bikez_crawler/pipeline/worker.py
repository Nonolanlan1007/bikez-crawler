"""Crash-safe claim loop for the CPU image-processing pipeline.

Each claimed ``images`` doc goes through watermark removal -> classification ->
(conditional) background removal -> publish to the production bucket. A claim failure
requeues the doc (or parks it as failed past the attempt cap) via ``retry_or_fail_sync``
rather than losing it.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from PIL import Image

from bikez_crawler.config import Settings
from bikez_crawler.db import SyncDatabase, claim_next_sync, retry_or_fail_sync
from bikez_crawler.s3 import S3Client

from .background import BackgroundRemover, has_uniform_light_or_dark_background
from .classify import Classifier
from .watermark import WatermarkRemover

logger = logging.getLogger(__name__)

READY_STATUS = "raw_downloaded"
IN_FLIGHT_STATUS = "processing"
DONE_STATUS = "processed"
UNCLASSIFIED_CATEGORY = "unclassified"


class PipelineContext:
    """Holds the models shared across every image claimed within this process."""

    def __init__(self) -> None:
        self.watermark_remover = WatermarkRemover()
        self.classifier = Classifier()
        self.background_remover = BackgroundRemover()


def process_image(
    context: PipelineContext, s3: S3Client, settings: Settings, doc: dict[str, Any]
) -> dict[str, Any]:
    bike_tag = doc["bike_tag"]
    pictno = doc["pictno"]
    raw_key = doc["raw_key"]

    raw_bytes = s3.download(settings.s3_raw_bucket, raw_key)
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    cleaned = context.watermark_remover.remove(image)
    category, subject = context.classifier.classify(cleaned)

    output = cleaned
    background_removed = False
    if subject == "illustration" and has_uniform_light_or_dark_background(cleaned):
        output = context.background_remover.remove(cleaned)
        background_removed = True

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    category_slug = category or UNCLASSIFIED_CATEGORY
    production_key = f"bikes/{bike_tag}/{pictno}_{category_slug}.png"
    s3.upload(settings.s3_production_bucket, production_key, buffer.getvalue(), content_type="image/png")

    return {
        "status": DONE_STATUS,
        "production_key": production_key,
        "category": category,
        "subject": subject,
        "background_removed": background_removed,
        "processed_at": datetime.now(UTC),
    }


def _claim_and_process(db: SyncDatabase, s3: S3Client, settings: Settings, context: PipelineContext) -> bool:
    extra_filter = {"bike_tag": settings.pipeline_bike_tag} if settings.pipeline_bike_tag else None
    doc = claim_next_sync(
        db.images,
        ready_statuses=[READY_STATUS],
        in_flight_status=IN_FLIGHT_STATUS,
        lease_seconds=settings.queue_lease_seconds,
        extra_filter=extra_filter,
    )
    if doc is None:
        return False

    try:
        updates = process_image(context, s3, settings, doc)
    except Exception:
        logger.exception("Failed to process image %s/%s", doc.get("bike_tag"), doc.get("pictno"))
        retry_or_fail_sync(
            db.images, doc, ready_status=READY_STATUS, max_attempts=settings.queue_max_attempts
        )
        return True

    db.images.update_one({"_id": doc["_id"]}, {"$set": updates})
    logger.info("Processed %s/%s -> %s", doc.get("bike_tag"), doc.get("pictno"), updates["production_key"])
    return True


def run(settings: Settings) -> None:
    db = SyncDatabase(settings.mongodb_url)
    db.ensure_indexes()
    s3 = S3Client(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    )
    s3.ensure_bucket(settings.s3_production_bucket)
    context = PipelineContext()

    def worker_loop() -> None:
        while _claim_and_process(db, s3, settings, context):
            pass

    concurrency = max(1, settings.pipeline_concurrency)
    if concurrency == 1:
        worker_loop()
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker_loop) for _ in range(concurrency)]
            for future in futures:
                future.result()

    db.close()
