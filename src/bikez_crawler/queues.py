"""Generic worker-pool draining, replacing the old p-queue-based ``drainQueue``.

Workers repeatedly claim-and-process items. A worker that finds nothing claimable waits
briefly instead of exiting immediately, in case another still-active worker is mid-process
and about to produce new claimable work (e.g. a list-page crawl inserting new pending
links) — the pool only stops once every worker is simultaneously idle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

IDLE_POLL_INTERVAL_SECONDS = 0.05


async def drain_queue(
    *,
    concurrency: int,
    claim_and_process: Callable[[], Awaitable[bool]],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Run ``concurrency`` workers against ``claim_and_process`` until the queue is drained.

    ``claim_and_process`` claims and handles one item, returning True if it found work and
    False if there was nothing claimable. It is responsible for its own error handling
    (typically via ``retry_or_fail_async``) — an exception here aborts the whole pool.
    """
    active = concurrency
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal active
        idle = False

        while True:
            if should_stop is not None and should_stop():
                return

            did_work = await claim_and_process()

            if did_work:
                if idle:
                    idle = False
                    async with lock:
                        active += 1
                continue

            if not idle:
                idle = True
                async with lock:
                    active -= 1

            async with lock:
                remaining = active
            if remaining <= 0:
                return

            await asyncio.sleep(IDLE_POLL_INTERVAL_SECONDS)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
