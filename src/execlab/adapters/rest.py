"""REST market-data interface and the Coinbase Exchange public implementation.

`MarketDataRest` is the venue-agnostic interface; `CoinbaseRest` implements it over an
injectable `HttpTransport` so tests run against canned payloads with no network.
Every request passes through `safety.assert_public_rest` (GET + public path allowlist).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from ..safety import assert_public_rest
from ..types import BookSnapshot, ProductSpec, Side, Trade

PRODUCTION = "https://api.exchange.coinbase.com"
SANDBOX = "https://api-public.sandbox.exchange.coinbase.com"


class HttpTransport(Protocol):
    def get(self, url: str, params: dict[str, Any] | None, timeout: float) -> tuple[int, Any]: ...


class RequestsTransport:
    """GET-only transport. There is deliberately no method for any other verb."""

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()
        self._session.headers["User-Agent"] = "crypto-execution-lab/0.1 (public market data)"

    def get(self, url: str, params: dict[str, Any] | None, timeout: float) -> tuple[int, Any]:
        r = self._session.get(url, params=params, timeout=timeout)
        try:
            body = r.json()
        except ValueError:
            body = r.text
        return r.status_code, body


class RestError(RuntimeError):
    pass


class TokenBucket:
    """Client-side rate limiter (Coinbase public REST allows ~10 req/s per IP)."""

    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate, self.capacity = rate, float(burst)
        self.tokens = float(burst)
        self._clock, self._sleep = clock, sleep
        self._last = clock()

    def acquire(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now
        if self.tokens < 1.0:
            wait = (1.0 - self.tokens) / self.rate
            self._sleep(wait)
            self._last = self._clock()
            self.tokens = 0.0
        else:
            self.tokens -= 1.0


def parse_ts(s: str) -> float:
    """ISO-8601 (Coinbase uses nanosecond precision + 'Z') -> epoch seconds."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s + "+00:00").timestamp()


class MarketDataRest(ABC):
    @abstractmethod
    def product(self, product_id: str) -> ProductSpec: ...

    @abstractmethod
    def book(self, product_id: str, level: int = 2) -> BookSnapshot: ...

    @abstractmethod
    def trades(self, product_id: str, limit: int = 100) -> list[Trade]: ...

    @abstractmethod
    def candles(self, product_id: str, granularity: int = 60) -> list[tuple[float, ...]]: ...


class CoinbaseRest(MarketDataRest):
    def __init__(
        self,
        base_url: str = PRODUCTION,
        transport: HttpTransport | None = None,
        limiter: TokenBucket | None = None,
        max_retries: int = 4,
        timeout: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport if transport is not None else RequestsTransport()
        self.limiter = limiter if limiter is not None else TokenBucket(rate=8.0, burst=8)
        self.max_retries, self.timeout, self._sleep = max_retries, timeout, sleep

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        assert_public_rest(url, "GET")
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                status, body = self.transport.get(url, params, self.timeout)
            except Exception as exc:  # network error -> retry with backoff
                if attempt == self.max_retries:
                    raise RestError(f"GET {path} failed: {exc}") from exc
            else:
                if status == 200:
                    return body
                if status not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise RestError(f"GET {path} -> HTTP {status}: {body}")
            self._sleep(delay)
            delay = min(delay * 2, 8.0)
        raise RestError("unreachable")

    def product(self, product_id: str) -> ProductSpec:
        b = self._get(f"/products/{product_id}")
        return ProductSpec(
            product=b["id"],
            tick=float(b["quote_increment"]),
            lot=float(b["base_increment"]),
            min_notional=float(b.get("min_market_funds") or 0.0),
        )

    def book(self, product_id: str, level: int = 2) -> BookSnapshot:
        b = self._get(f"/products/{product_id}/book", {"level": level})
        ts = time.time()
        return BookSnapshot(
            ts=ts,
            product=product_id,
            bids=tuple((float(p), float(q)) for p, q, *_ in b["bids"]),
            asks=tuple((float(p), float(q)) for p, q, *_ in b["asks"]),
        )

    def trades(self, product_id: str, limit: int = 100) -> list[Trade]:
        rows = self._get(f"/products/{product_id}/trades", {"limit": limit})
        out = [
            Trade(
                ts=parse_ts(r["time"]),
                product=product_id,
                price=float(r["price"]),
                size=float(r["size"]),
                # Coinbase Exchange reports the *maker* side; the aggressor is the opposite.
                aggressor=Side(r["side"]).opposite,
                trade_id=str(r["trade_id"]),
            )
            for r in rows
        ]
        return sorted(out, key=lambda t: (t.ts, t.trade_id))

    def candles(self, product_id: str, granularity: int = 60) -> list[tuple[float, ...]]:
        rows = self._get(f"/products/{product_id}/candles", {"granularity": granularity})
        # [time, low, high, open, close, volume] -> ascending time
        return sorted(tuple(float(x) for x in r) for r in rows)
