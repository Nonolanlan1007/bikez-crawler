"""Mongo schema and crash-safe lease-based queue claiming.

A claimed document is marked with an in-flight status and a ``lease_expires_at``
timestamp instead of being removed, so a crash never loses in-flight work: the lease
simply expires and another worker (or the same one, after restart) can reclaim it —
there is never a window where the work exists nowhere.

Both an async (motor, used by the crawler) and a sync (pymongo, used by the CPU-bound
image pipeline) client are provided, sharing the same collection names, index
definitions, and claim query shape.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypedDict, cast

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database as PyMongoDatabase
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)

DATABASE_NAME = "bikez"

BIKES = "bikes"
PENDING_LINKS = "pendingLinks"
IMAGES = "images"
CRAWLER_STATE = "crawlerState"


class BikeDoc(TypedDict, total=False):
    tag: str
    brand: str
    specs_raw: dict[str, dict[str, str]]
    specs_normalized: dict[str, Any]
    specs_mapping_version: int
    crawled_at: datetime
    image_count: int


PendingLinkStatus = Literal["pending", "claimed"]


class PendingLinkDoc(TypedDict, total=False):
    link: str
    status: PendingLinkStatus
    crawled_at: datetime
    attempts: int
    locked_at: datetime | None
    lease_expires_at: datetime | None


ImageStatus = Literal["discovered", "downloading", "raw_downloaded", "processing", "processed", "failed"]


class ImageDoc(TypedDict, total=False):
    bike_tag: str
    pictno: int
    url: str
    status: ImageStatus
    raw_key: str | None
    raw_hash: str | None
    category: str | None
    subject: Literal["illustration", "photograph"] | None
    background_removed: bool | None
    production_key: str | None
    attempts: int
    locked_at: datetime | None
    lease_expires_at: datetime | None
    discovered_at: datetime
    processed_at: datetime | None


# (collection, keys, options)
_INDEXES: list[tuple[str, list[tuple[str, int]], dict[str, Any]]] = [
    (BIKES, [("tag", ASCENDING)], {"unique": True}),
    (PENDING_LINKS, [("link", ASCENDING)], {"unique": True}),
    (PENDING_LINKS, [("status", ASCENDING), ("lease_expires_at", ASCENDING)], {}),
    (IMAGES, [("bike_tag", ASCENDING), ("pictno", ASCENDING)], {"unique": True}),
    (IMAGES, [("url", ASCENDING)], {}),
    (IMAGES, [("status", ASCENDING), ("lease_expires_at", ASCENDING)], {}),
]


async def ensure_indexes_async(db: AsyncIOMotorDatabase[Any]) -> None:
    for name, keys, options in _INDEXES:
        try:
            await db[name].create_index(keys, **options)
        except OperationFailure:
            logger.warning("Failed to create index %s on %s", keys, name, exc_info=True)


def ensure_indexes_sync(db: PyMongoDatabase[Any]) -> None:
    for name, keys, options in _INDEXES:
        try:
            db[name].create_index(keys, **options)
        except OperationFailure:
            logger.warning("Failed to create index %s on %s", keys, name, exc_info=True)


class AsyncDatabase:
    """Used by the crawler (link discovery, bike scraping, raw image download)."""

    def __init__(self, mongodb_url: str) -> None:
        self.client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(mongodb_url, tz_aware=True)
        self.db: AsyncIOMotorDatabase[Any] = self.client[DATABASE_NAME]
        self.bikes: AsyncIOMotorCollection[Any] = self.db[BIKES]
        self.pending_links: AsyncIOMotorCollection[Any] = self.db[PENDING_LINKS]
        self.images: AsyncIOMotorCollection[Any] = self.db[IMAGES]
        self.crawler_state: AsyncIOMotorCollection[Any] = self.db[CRAWLER_STATE]

    async def ensure_schema(self) -> None:
        await ensure_indexes_async(self.db)

    async def close(self) -> None:
        self.client.close()


class SyncDatabase:
    """Used by the CPU-bound image pipeline."""

    def __init__(self, mongodb_url: str) -> None:
        self.client: MongoClient[Any] = MongoClient(mongodb_url, tz_aware=True)
        self.db: PyMongoDatabase[Any] = self.client[DATABASE_NAME]
        self.bikes: Collection[Any] = self.db[BIKES]
        self.pending_links: Collection[Any] = self.db[PENDING_LINKS]
        self.images: Collection[Any] = self.db[IMAGES]

    def ensure_indexes(self) -> None:
        ensure_indexes_sync(self.db)

    def close(self) -> None:
        self.client.close()


def _claim_filter(ready_statuses: Sequence[str], in_flight_status: str, now: datetime) -> dict[str, Any]:
    return {
        "$or": [
            {"status": {"$in": list(ready_statuses)}},
            {"status": in_flight_status, "lease_expires_at": {"$lt": now}},
        ]
    }


def _claim_update(in_flight_status: str, now: datetime, lease_seconds: int) -> dict[str, Any]:
    return {
        "$set": {
            "status": in_flight_status,
            "locked_at": now,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
        },
        "$inc": {"attempts": 1},
    }


async def claim_next_async(
    collection: AsyncIOMotorCollection[Any],
    *,
    ready_statuses: Sequence[str],
    in_flight_status: str,
    lease_seconds: int,
    extra_filter: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Atomically claim one document: mark it in-flight with a fresh lease.

    A crashed worker's claim is reclaimed automatically once ``lease_expires_at`` passes
    — nothing needs to detect the crash explicitly.
    """
    now = datetime.now(UTC)
    query = _claim_filter(ready_statuses, in_flight_status, now)
    if extra_filter:
        query = {"$and": [query, extra_filter]}
    result = await collection.find_one_and_update(
        query, _claim_update(in_flight_status, now, lease_seconds), return_document=ReturnDocument.AFTER
    )
    return cast("dict[str, Any] | None", result)


def claim_next_sync(
    collection: Collection[Any],
    *,
    ready_statuses: Sequence[str],
    in_flight_status: str,
    lease_seconds: int,
    extra_filter: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    now = datetime.now(UTC)
    query = _claim_filter(ready_statuses, in_flight_status, now)
    if extra_filter:
        query = {"$and": [query, extra_filter]}
    return collection.find_one_and_update(
        query, _claim_update(in_flight_status, now, lease_seconds), return_document=ReturnDocument.AFTER
    )


async def retry_or_fail_async(
    collection: AsyncIOMotorCollection[Any],
    doc: dict[str, Any],
    *,
    ready_status: str,
    max_attempts: int,
) -> None:
    """On processing failure: requeue (clearing the lease) if under the attempt cap, else park as failed."""
    if doc.get("attempts", 0) >= max_attempts:
        await collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed"}})
    else:
        await collection.update_one(
            {"_id": doc["_id"]}, {"$set": {"status": ready_status, "lease_expires_at": None}}
        )


def retry_or_fail_sync(
    collection: Collection[Any],
    doc: dict[str, Any],
    *,
    ready_status: str,
    max_attempts: int,
) -> None:
    if doc.get("attempts", 0) >= max_attempts:
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed"}})
    else:
        collection.update_one(
            {"_id": doc["_id"]}, {"$set": {"status": ready_status, "lease_expires_at": None}}
        )
