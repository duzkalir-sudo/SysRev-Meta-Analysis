"""Asynchronous sliding-window rate limiter."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable


class AsyncRateLimiter:
    """Allow at most ``max_calls`` acquisitions in any rolling ``period`` seconds.

    Safe for concurrent use by many coroutines on one event loop. ``clock`` and
    ``sleep`` are injectable so tests can run deterministically without waiting.
    """

    def __init__(
        self,
        max_calls: int,
        period: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if period <= 0:
            raise ValueError("period must be > 0")
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    @classmethod
    def per_second(cls, rate: float, **kwargs: object) -> AsyncRateLimiter:
        """Build a limiter for a (possibly fractional) requests-per-second rate."""
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if rate >= 1:
            return cls(int(rate), 1.0, **kwargs)  # type: ignore[arg-type]
        return cls(1, 1.0 / rate, **kwargs)  # type: ignore[arg-type]

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                await self._sleep(self.period - (now - self._calls[0]))

    async def __aenter__(self) -> AsyncRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None
