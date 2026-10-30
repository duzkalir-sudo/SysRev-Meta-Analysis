from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest
import respx

from src.apis.base import BaseAPIClient, _parse_retry_after
from src.apis.rate_limit import AsyncRateLimiter
from src.core.exceptions import APIClientError
from tests.conftest import FakeClock

# --------------------------------------------------------------------------- rate limiter


async def test_rate_limiter_enforces_window(fake_clock: FakeClock) -> None:
    limiter = AsyncRateLimiter(3, 1.0, clock=fake_clock, sleep=fake_clock.sleep)
    for _ in range(3):
        await limiter.acquire()
    assert fake_clock.sleeps == []
    await limiter.acquire()  # 4th call within the same second must wait
    assert fake_clock.sleeps == [pytest.approx(1.0)]
    assert fake_clock.now == pytest.approx(1.0)


async def test_rate_limiter_releases_after_window(fake_clock: FakeClock) -> None:
    limiter = AsyncRateLimiter(2, 1.0, clock=fake_clock, sleep=fake_clock.sleep)
    await limiter.acquire()
    fake_clock.now = 0.6
    await limiter.acquire()
    fake_clock.now = 1.1  # first call left the window
    await limiter.acquire()
    assert fake_clock.sleeps == []


async def test_rate_limiter_concurrent_acquirers(fake_clock: FakeClock) -> None:
    limiter = AsyncRateLimiter(3, 1.0, clock=fake_clock, sleep=fake_clock.sleep)
    stamps: list[float] = []

    async def worker() -> None:
        await limiter.acquire()
        stamps.append(fake_clock.now)

    await asyncio.gather(*(worker() for _ in range(10)))
    # No rolling 1-second window may contain more than 3 acquisitions.
    for t in stamps:
        assert sum(1 for s in stamps if t <= s < t + 1.0) <= 3
    assert max(stamps) == pytest.approx(3.0)


async def test_rate_limiter_context_manager(fake_clock: FakeClock) -> None:
    limiter = AsyncRateLimiter(1, 1.0, clock=fake_clock, sleep=fake_clock.sleep)
    async with limiter:
        pass
    async with limiter:
        pass
    assert fake_clock.sleeps == [pytest.approx(1.0)]


def test_rate_limiter_per_second_factory() -> None:
    assert AsyncRateLimiter.per_second(10).max_calls == 10
    slow = AsyncRateLimiter.per_second(0.5)
    assert (slow.max_calls, slow.period) == (1, 2.0)
    with pytest.raises(ValueError, match="rate"):
        AsyncRateLimiter.per_second(0)
    with pytest.raises(ValueError, match="max_calls"):
        AsyncRateLimiter(0)
    with pytest.raises(ValueError, match="period"):
        AsyncRateLimiter(1, 0)


# --------------------------------------------------------------------------- base client


class DummyClient(BaseAPIClient):
    base_url = "https://api.example.test/"
    name = "dummy"


def _client(
    limiter: AsyncRateLimiter, sleep: Callable[[float], Awaitable[None]], retries: int = 3
) -> DummyClient:
    return DummyClient(rate_limiter=limiter, max_retries=retries, sleep=sleep)


@respx.mock
async def test_retries_on_429_and_honours_retry_after(no_wait_limiter: AsyncRateLimiter) -> None:
    sleeps: list[float] = []

    async def record(s: float) -> None:
        sleeps.append(s)

    route = respx.get("https://api.example.test/x").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    async with _client(no_wait_limiter, record) as c:
        assert await c._get_json("x") == {"ok": True}
    assert route.call_count == 3
    assert sleeps == [7.0, 2.0]  # Retry-After, then exponential backoff (1 * 2**1)


@respx.mock
async def test_gives_up_after_max_retries(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Callable[[float], Awaitable[None]]
) -> None:
    respx.get("https://api.example.test/x").mock(return_value=httpx.Response(500))
    async with _client(no_wait_limiter, no_sleep, retries=2) as c:
        with pytest.raises(APIClientError) as info:
            await c._get_json("x")
    assert info.value.status_code == 500


@respx.mock
async def test_transport_errors_are_retried(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Callable[[float], Awaitable[None]]
) -> None:
    route = respx.get("https://api.example.test/x").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json=[1])]
    )
    async with _client(no_wait_limiter, no_sleep) as c:
        assert await c._get_json("x") == [1]
    assert route.call_count == 2


@respx.mock
async def test_transport_errors_exhausted(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Callable[[float], Awaitable[None]]
) -> None:
    respx.get("https://api.example.test/x").mock(side_effect=httpx.ConnectError("down"))
    async with _client(no_wait_limiter, no_sleep, retries=1) as c:
        with pytest.raises(APIClientError, match="after 2 attempts"):
            await c._get_json("x")


@respx.mock
async def test_client_errors_not_retried(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Callable[[float], Awaitable[None]]
) -> None:
    route = respx.get("https://api.example.test/x").mock(return_value=httpx.Response(400))
    async with _client(no_wait_limiter, no_sleep) as c:
        with pytest.raises(APIClientError, match="HTTP 400"):
            await c._get_json("x")
    assert route.call_count == 1


@respx.mock
async def test_404_allowed_and_invalid_json(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Callable[[float], Awaitable[None]]
) -> None:
    respx.get("https://api.example.test/missing").mock(return_value=httpx.Response(404))
    respx.get("https://api.example.test/html").mock(return_value=httpx.Response(200, text="<h"))
    async with _client(no_wait_limiter, no_sleep) as c:
        assert await c._get_json("missing", allow_404=True) is None
        with pytest.raises(APIClientError, match="invalid JSON"):
            await c._get_json("html")


async def test_every_request_passes_through_limiter(
    fake_clock: FakeClock, no_sleep: Callable[[float], Awaitable[None]]
) -> None:
    limiter = AsyncRateLimiter(2, 1.0, clock=fake_clock, sleep=fake_clock.sleep)
    with respx.mock:
        respx.get("https://api.example.test/x").mock(return_value=httpx.Response(200, json={}))
        async with _client(limiter, no_sleep) as c:
            for _ in range(5):
                await c._get_json("x")
    assert fake_clock.now == pytest.approx(2.0)


def test_parse_retry_after() -> None:
    assert _parse_retry_after("12") == 12.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("garbage") is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past
