"""Crash-safe, two-stage CPU image-processing pipeline.

Preprocessing removes watermarks and classifies an image, then stores the cleaned
intermediate in object storage. A separate process performs optional background
removal and publishes the final image. Keeping those stages in separate processes
prevents the largest models from being resident at the same time.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Protocol

from PIL import Image

from bikez_crawler.config import Settings
from bikez_crawler.db import SyncDatabase, claim_next_sync, retry_or_fail_sync
from bikez_crawler.s3 import S3Client

from .background import BackgroundRemover, has_uniform_light_or_dark_background
from .classify import Classifier
from .watermark import WatermarkRemover

logger = logging.getLogger(__name__)

RAW_READY_STATUS = "raw_downloaded"
PREPROCESSING_STATUS = "preprocessing"
BACKGROUND_PENDING_STATUS = "background_pending"
BACKGROUND_PROCESSING_STATUS = "background_processing"
DONE_STATUS = "processed"
UNCLASSIFIED_CATEGORY = "unclassified"


class ImageProcessor(Protocol):
    def process(self, image: Image.Image, doc: dict[str, Any]) -> dict[str, Any]:
        ...


class Preprocessor:
    """Watermark removal and classification models for the first stage."""

    def __init__(self) -> None:
        self.watermark_remover = WatermarkRemover()
        self.classifier = Classifier()

    def process(self, image: Image.Image, doc: dict[str, Any]) -> dict[str, Any]:
        cleaned = self.watermark_remover.remove(image)
        category, subject = self.classifier.classify(cleaned)
        return {
            "image": cleaned,
            "category": category,
            "subject": subject,
            "background_eligible": subject == "illustration"
            and has_uniform_light_or_dark_background(cleaned),
        }


class BackgroundProcessor:
    """Background-removal model for the second stage."""

    def __init__(self) -> None:
        self.background_remover = BackgroundRemover()

    def process(self, image: Image.Image, doc: dict[str, Any]) -> dict[str, Any]:
        if not doc.get("background_eligible", False):
            return {"image": image, "background_removed": False}
        return {"image": self.background_remover.remove(image), "background_removed": True}


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _intermediate_key(settings: Settings, doc: dict[str, Any]) -> str:
    return f"{settings.pipeline_intermediate_prefix}/{doc['bike_tag']}/{doc['pictno']}.png"


def _production_key(doc: dict[str, Any]) -> str:
    category_slug = doc.get("category") or UNCLASSIFIED_CATEGORY
    return f"bikes/{doc['bike_tag']}/{doc['pictno']}_{category_slug}.png"


def _load_image(s3: S3Client, bucket: str, key: str) -> Image.Image:
    return Image.open(io.BytesIO(s3.download(bucket, key))).convert("RGB")


def _claim_and_preprocess(
    db: SyncDatabase, s3: S3Client, settings: Settings, processor: Preprocessor
) -> bool:
    extra_filter = {"bike_tag": settings.pipeline_bike_tag} if settings.pipeline_bike_tag else None
    doc = claim_next_sync(
        db.images,
        ready_statuses=[RAW_READY_STATUS],
        in_flight_status=PREPROCESSING_STATUS,
        lease_seconds=settings.queue_lease_seconds,
        extra_filter=extra_filter,
    )
    if doc is None:
        return False

    try:
        image = _load_image(s3, settings.s3_raw_bucket, doc["raw_key"])
        result = processor.process(image, doc)
        intermediate_key = _intermediate_key(settings, doc)
        s3.upload(
            settings.s3_production_bucket,
            intermediate_key,
            _encode_png(result["image"]),
            content_type="image/png",
        )
        updates = {
            "status": BACKGROUND_PENDING_STATUS,
            "intermediate_key": intermediate_key,
            "category": result["category"],
            "subject": result["subject"],
            "background_eligible": result["background_eligible"],
            "background_removed": False,
        }
        db.images.update_one({"_id": doc["_id"]}, {"$set": updates})
    except Exception:
        logger.exception("Failed to preprocess image %s/%s", doc.get("bike_tag"), doc.get("pictno"))
        retry_or_fail_sync(
            db.images,
            doc,
            ready_status=RAW_READY_STATUS,
            max_attempts=settings.queue_max_attempts,
        )
        return True

    logger.info("Preprocessed %s/%s", doc.get("bike_tag"), doc.get("pictno"))
    return True


def _claim_and_finish(
    db: SyncDatabase, s3: S3Client, settings: Settings, processor: BackgroundProcessor
) -> bool:
    extra_filter = {"bike_tag": settings.pipeline_bike_tag} if settings.pipeline_bike_tag else None
    doc = claim_next_sync(
        db.images,
        ready_statuses=[BACKGROUND_PENDING_STATUS],
        in_flight_status=BACKGROUND_PROCESSING_STATUS,
        lease_seconds=settings.queue_lease_seconds,
        extra_filter=extra_filter,
    )
    if doc is None:
        return False

    try:
        image = _load_image(s3, settings.s3_production_bucket, doc["intermediate_key"])
        result = processor.process(image, doc)
        production_key = _production_key(doc)
        s3.upload(
            settings.s3_production_bucket,
            production_key,
            _encode_png(result["image"]),
            content_type="image/png",
        )
        updates = {
            "status": DONE_STATUS,
            "production_key": production_key,
            "background_removed": result["background_removed"],
            "processed_at": datetime.now(UTC),
        }
        db.images.update_one({"_id": doc["_id"]}, {"$set": updates})
    except Exception:
        logger.exception("Failed to finish image %s/%s", doc.get("bike_tag"), doc.get("pictno"))
        retry_or_fail_sync(
            db.images,
            doc,
            ready_status=BACKGROUND_PENDING_STATUS,
            max_attempts=settings.queue_max_attempts,
        )
        return True

    logger.info("Processed %s/%s -> %s", doc.get("bike_tag"), doc.get("pictno"), production_key)
    return True


def _run_stage(
    db: SyncDatabase,
    s3: S3Client,
    settings: Settings,
    processor: ImageProcessor,
    claim_and_process: Any,
) -> None:
    def worker_loop() -> None:
        while claim_and_process(db, s3, settings, processor):
            pass

    concurrency = max(1, settings.pipeline_concurrency)
    if concurrency == 1:
        worker_loop()
        return
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker_loop) for _ in range(concurrency)]
        for future in futures:
            future.result()


def run(settings: Settings, stage: str = "preprocess") -> None:
    db = SyncDatabase(settings.mongodb_url)
    db.ensure_indexes()
    s3 = S3Client(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    )
    s3.ensure_bucket(settings.s3_production_bucket)
    if stage == "preprocess":
        _run_stage(db, s3, settings, Preprocessor(), _claim_and_preprocess)
    elif stage == "background":
        _run_stage(db, s3, settings, BackgroundProcessor(), _claim_and_finish)
    else:
        raise ValueError(f"Unsupported pipeline stage: {stage}")
    db.close()
