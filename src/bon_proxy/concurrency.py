"""Race-free, non-blocking workflow concurrency control."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ConcurrencyGate:
    """Limit active workflows without queueing excess requests."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self.capacity:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError("concurrency gate released without an active slot")
            self._active -= 1

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[bool]:
        acquired = await self.try_acquire()
        try:
            yield acquired
        finally:
            if acquired:
                await self.release()
