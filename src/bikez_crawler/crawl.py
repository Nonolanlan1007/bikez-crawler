"""Entrypoint: ``python -m bikez_crawler.crawl``.

Drains the pending-links queue (crawling year-list and bike pages, discovering new
images along the way), then the raw-image-download queue. Meant to be run periodically
(e.g. a weekly cron/CronJob) rather than as a long-lived daemon: it exits once both
queues are drained, and ``seed_if_due`` re-triggers a full sweep on each run spaced far
enough apart, so re-running costs one HTTP request per already-known page (specs are
overwritten idempotently, ``_discover_images`` only probes past ``image_count``) plus
whatever's genuinely new.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from bikez_crawler.config import Settings, load_settings
from bikez_crawler.crawler import YEARS_INDEX_URL, crawl_page
from bikez_crawler.db import AsyncDatabase, claim_next_async, retry_or_fail_async
from bikez_crawler.http import BikezHttpClient
from bikez_crawler.queues import drain_queue
from bikez_crawler.s3 import S3Client

logger = logging.getLogger(__name__)

PROGRESS_LOG_INTERVAL_SECONDS = 30.0
SEED_STATE_ID = "seed"


async def seed_if_due(db: AsyncDatabase, reseed_interval_days: int) -> None:
    """Re-seed the years-index URL on the first run and every ``reseed_interval_days``
    after, so new bikez.com listings get picked up without a durable "already seeded"
    flag that would otherwise seed exactly once forever.
    """
    now = datetime.now(UTC)
    state = await db.crawler_state.find_one({"_id": SEED_STATE_ID})
    due = state is None or now - state["last_seeded_at"] >= timedelta(days=reseed_interval_days)
    if not due:
        return

    await db.pending_links.update_one(
        {"link": YEARS_INDEX_URL},
        {"$setOnInsert": {"link": YEARS_INDEX_URL, "status": "pending", "crawled_at": now}},
        upsert=True,
    )
    await db.crawler_state.update_one(
        {"_id": SEED_STATE_ID}, {"$set": {"last_seeded_at": now}}, upsert=True
    )


async def log_progress_periodically(db: AsyncDatabase, interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        bikes, pending_links, images = await asyncio.gather(
            db.bikes.count_documents({}),
            db.pending_links.count_documents({}),
            db.images.count_documents({}),
        )
        logger.info("bikes=%d pending_links=%d images=%d", bikes, pending_links, images)


async def run_links_queue(
    db: AsyncDatabase, http: BikezHttpClient, s3: S3Client, settings: Settings
) -> None:
    bikes_crawled = 0

    async def claim_and_process() -> bool:
        nonlocal bikes_crawled
        doc: dict[str, Any] | None = await claim_next_async(
            db.pending_links,
            ready_statuses=["pending"],
            in_flight_status="claimed",
            lease_seconds=settings.queue_lease_seconds,
        )
        if doc is None:
            return False

        try:
            was_bike_page = await crawl_page(doc["link"], db, http, s3, settings)
            await db.pending_links.delete_one({"_id": doc["_id"]})
            if was_bike_page:
                bikes_crawled += 1
        except Exception:
            logger.exception("Failed to process pending link %s", doc.get("link"))
            await retry_or_fail_async(
                db.pending_links, doc, ready_status="pending", max_attempts=settings.queue_max_attempts
            )
        return True

    def should_stop() -> bool:
        return settings.max_bikes_to_crawl is not None and bikes_crawled >= settings.max_bikes_to_crawl

    await drain_queue(
        concurrency=settings.pending_links_concurrency,
        claim_and_process=claim_and_process,
        should_stop=should_stop,
    )


async def run_image_download_queue(
    db: AsyncDatabase, http: BikezHttpClient, s3: S3Client, settings: Settings
) -> None:
    async def claim_and_process() -> bool:
        doc: dict[str, Any] | None = await claim_next_async(
            db.images,
            ready_statuses=["discovered"],
            in_flight_status="downloading",
            lease_seconds=settings.queue_lease_seconds,
        )
        if doc is None:
            return False

        try:
            response = await http.get(doc["url"])
            raw_hash = hashlib.sha256(response.content).hexdigest()
            key = f"bikes/{doc['bike_tag']}/{os.path.basename(urlparse(doc['url']).path)}"
            content_type = response.headers.get("content-type")

            await asyncio.to_thread(s3.upload, settings.s3_raw_bucket, key, response.content, content_type)
            await db.images.update_one(
                {"_id": doc["_id"]},
                {"$set": {"status": "raw_downloaded", "raw_key": key, "raw_hash": raw_hash}},
            )
        except Exception:
            logger.exception("Failed to download image %s", doc.get("url"))
            await retry_or_fail_async(
                db.images, doc, ready_status="discovered", max_attempts=settings.queue_max_attempts
            )
        return True

    await drain_queue(concurrency=settings.image_upload_concurrency, claim_and_process=claim_and_process)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    db = AsyncDatabase(settings.mongodb_url)
    s3 = S3Client(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    )
    http = BikezHttpClient(
        concurrency=settings.bikez_http_concurrency,
        interval_ms=settings.bikez_http_interval_ms,
        interval_cap=settings.bikez_http_interval_cap,
    )

    progress_task: asyncio.Task[None] | None = None
    try:
        await db.ensure_schema()
        await asyncio.to_thread(s3.ensure_bucket, settings.s3_raw_bucket)
        await seed_if_due(db, settings.reseed_interval_days)

        progress_task = asyncio.create_task(log_progress_periodically(db, PROGRESS_LOG_INTERVAL_SECONDS))

        await run_links_queue(db, http, s3, settings)
        await run_image_download_queue(db, http, s3, settings)
    finally:
        if progress_task is not None:
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task
        await http.aclose()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
