"""Rate-limited HTTP client for bikez.com.

At most ``interval_cap`` requests started per ``interval_ms`` window, and at most
``concurrency`` requests in flight at once.
"""

from __future__ import annotations

import asyncio

import httpx
from aiolimiter import AsyncLimiter


class BikezHttpClient:
    def __init__(self, *, concurrency: int, interval_ms: int, interval_cap: int) -> None:
        self._client = httpx.AsyncClient(follow_redirects=True, timeout=30.0)
        self._limiter = AsyncLimiter(max_rate=interval_cap, time_period=interval_ms / 1000)
        self._semaphore = asyncio.Semaphore(concurrency)

    async def get(self, url: str) -> httpx.Response:
        async with self._semaphore, self._limiter:
            response = await self._client.get(url)
            response.raise_for_status()
            return response

    async def aclose(self) -> None:
        await self._client.aclose()
