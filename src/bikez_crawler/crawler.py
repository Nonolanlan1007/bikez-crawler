"""Page routing and scraping logic.

Three URL shapes are handled: the years-index page and per-year list pages (both list
pages, walked identically), and individual bike pages.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import UTC, datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from pymongo import InsertOne
from pymongo.errors import BulkWriteError

from bikez_crawler.db import AsyncDatabase
from bikez_crawler.http import BikezHttpClient
from bikez_crawler.specs import CURRENT_MAPPING_VERSION, normalize_specs

logger = logging.getLogger(__name__)

YEARS_INDEX_URL = "https://bikez.com/years/index.php"
YEAR_LIST_RE = re.compile(r"^https://bikez\.com/year/([0-9]{4})-motorcycle-models\.php$")
BIKE_PAGE_RE = re.compile(r"^https://bikez\.com/motorcycles/([A-Za-z0-9_]+)\.php$")

# `_poster.php` (not `_picture.php`, which only links a ~400px thumbnail) links the
# full-resolution original. The corner watermark is a fixed pixel size regardless of
# source resolution (see pipeline/watermark.py), so the bigger the source image, the
# smaller a fraction of it the watermark covers -- this is what makes watermark removal
# viable at all.
GALLERY_URL_TEMPLATE = "https://bikez.com/gallery/{tag}_poster.php?pictno={pictno}"

# Absolute safety cap so a website quirk can't turn the resumed walk into an unbounded loop.
MAX_IMAGE_PICTNO = 100

_DATA_ARRAY_RE = re.compile(r"var dataArray = (\[.*?\]);", re.DOTALL)


def rot13(value: str) -> str:
    def shift(ch: str) -> str:
        if "a" <= ch <= "z":
            return chr((ord(ch) - ord("a") + 13) % 26 + ord("a"))
        if "A" <= ch <= "Z":
            return chr((ord(ch) - ord("A") + 13) % 26 + ord("A"))
        return ch

    return "".join(shift(ch) for ch in value)


def descramble_lazy_fields(soup: BeautifulSoup, html: str) -> None:
    """Fill in spec fields the site renders as "Loading..." and populates client-side
    from a ``dataArray`` blob (base64 -> JSON string -> rot13), keyed by span class name.
    """
    match = _DATA_ARRAY_RE.search(html)
    if not match:
        return

    data_array: list[dict[str, str]] = json.loads(match.group(1))

    for obj in data_array:
        for class_name, encoded in obj.items():
            decoded = rot13(json.loads(base64.b64decode(encoded).decode("utf-8")))
            fragment = BeautifulSoup(decoded, "html.parser")
            for element in soup.select(f".{class_name}"):
                element.clear()
                for child in list(fragment.contents):
                    element.append(child)


def extract_brand(soup: BeautifulSoup) -> str:
    header_table = soup.select_one("table.headertable")
    if header_table is None:
        return ""

    link = header_table.select_one("td h1 a")
    return link.get_text() if link else ""


def find_specs_table(soup: BeautifulSoup) -> Tag | None:
    content = soup.select_one("#pagecontent")
    if content is None:
        return None

    for table in content.select("table.Grid"):
        header = table.find("th")
        if header is None:
            continue
        if re.match(r"^general.*information$", header.get_text(strip=True), re.IGNORECASE):
            return table

    return None


def table_to_object(table: Tag | None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if table is None:
        return result

    current_section = "General"

    for row in table.find_all("tr"):
        header = row.find("th")
        if header is not None:
            current_section = header.get_text(strip=True)
            result.setdefault(current_section, {})
            continue

        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        label = cells[0].find("b")
        if label is None:
            continue

        value_cell = cells[1]
        if value_cell.find(["select", "form", "input"]) is not None:
            continue  # skip compare widget

        for hidden in value_cell.select('[style*="display:none"]'):
            hidden.decompose()  # strip anti-scraping noise

        key = label.get_text(strip=True)
        value = re.sub(r"\s+", " ", value_cell.get_text()).strip()

        if key and value:
            section = result.setdefault(current_section, {})
            unique_key = key
            occurrence = 2
            while unique_key in section:
                unique_key = f"{key} ({occurrence})"
                occurrence += 1
            section[unique_key] = value

    return result


async def _insert_pending_links(links: list[str], db: AsyncDatabase) -> None:
    if not links:
        return

    now = datetime.now(UTC)
    operations = [
        InsertOne({"link": link, "status": "pending", "crawled_at": now}) for link in links
    ]

    try:
        await db.pending_links.bulk_write(operations, ordered=False)
    except BulkWriteError as exc:
        write_errors = exc.details.get("writeErrors", [])
        if not all(error.get("code") == 11000 for error in write_errors):
            raise


async def crawl_link_list_page(url: str, db: AsyncDatabase, http: BikezHttpClient) -> None:
    response = await http.get(url)
    soup = BeautifulSoup(response.text, "lxml")

    links: list[str] = []
    for row in soup.select("table.zebra tr"):
        if row.find("th") is not None:
            continue
        first_cell = row.find("td")
        if first_cell is None:
            continue
        anchor = first_cell.find("a")
        if anchor is None:
            continue
        href = anchor.get("href")
        if not href or not isinstance(href, str):
            continue
        links.append(urljoin(url, href))

    await _insert_pending_links(links, db)


async def _discover_images(tag: str, db: AsyncDatabase, http: BikezHttpClient) -> None:
    bike_doc = await db.bikes.find_one({"tag": tag}, {"image_count": 1})
    stored_image_count = bike_doc.get("image_count", 0) if bike_doc else 0

    highest_seen = stored_image_count
    pictno = stored_image_count + 1

    while pictno <= MAX_IMAGE_PICTNO:
        gallery_url = GALLERY_URL_TEMPLATE.format(tag=tag, pictno=pictno)
        response = await http.get(gallery_url)
        gallery_soup = BeautifulSoup(response.text, "lxml")
        image = gallery_soup.select_one("img[src*='/pictures/']")
        image_url = image.get("src") if image else None
        if not image_url or not isinstance(image_url, str):
            break
        image_url = urljoin(gallery_url, image_url)

        await db.images.update_one(
            {"bike_tag": tag, "pictno": pictno},
            {
                "$setOnInsert": {
                    "bike_tag": tag,
                    "pictno": pictno,
                    "url": image_url,
                    "status": "discovered",
                    "attempts": 0,
                    "discovered_at": datetime.now(UTC),
                }
            },
            upsert=True,
        )
        highest_seen = pictno
        pictno += 1

    if highest_seen > stored_image_count:
        await db.bikes.update_one({"tag": tag}, {"$set": {"image_count": highest_seen}})


async def crawl_bike_page(url: str, db: AsyncDatabase, http: BikezHttpClient) -> None:
    match = BIKE_PAGE_RE.match(url)
    if not match:
        raise ValueError(f"Could not extract bike tag from {url}")
    tag = match.group(1)

    response = await http.get(url)
    html = response.text
    soup = BeautifulSoup(html, "lxml")
    descramble_lazy_fields(soup, html)

    brand = extract_brand(soup)
    specs_raw = table_to_object(find_specs_table(soup))

    await db.bikes.update_one(
        {"tag": tag},
        {
            "$set": {
                "tag": tag,
                "brand": brand,
                "specs_raw": specs_raw,
                "specs_normalized": normalize_specs(specs_raw),
                "specs_mapping_version": CURRENT_MAPPING_VERSION,
                "crawled_at": datetime.now(UTC),
            }
        },
        upsert=True,
    )

    await _discover_images(tag, db, http)


async def crawl_page(url: str, db: AsyncDatabase, http: BikezHttpClient) -> bool:
    """Route ``url`` to the matching scraper. Returns True iff it was a bike page."""
    if url == YEARS_INDEX_URL or YEAR_LIST_RE.match(url):
        await crawl_link_list_page(url, db, http)
        return False
    if BIKE_PAGE_RE.match(url):
        await crawl_bike_page(url, db, http)
        return True
    raise ValueError(f"Unknown page: {url}")
