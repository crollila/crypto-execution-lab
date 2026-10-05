"""Real-money trading is disabled *by construction*.

Guarantees enforced here and covered by tests (tests/test_safety.py):

* The package contains no request-signing code and never reads API keys/secrets.
* The only HTTP transport issues GET requests, and only to an allowlist of public
  market-data paths on allowlisted Coinbase hosts.
* WebSocket subscriptions are restricted to public market-data channels.
* The only order gateway is the in-process simulator. `LiveOrderGateway` exists solely
  to make the boundary explicit: constructing it always raises.
"""

from __future__ import annotations

import re
from typing import Final
from urllib.parse import urlparse

LIVE_TRADING_ENABLED: Final[bool] = False

PUBLIC_REST_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "api.exchange.coinbase.com",  # Coinbase Exchange public market data
        "api-public.sandbox.exchange.coinbase.com",  # public sandbox market data
    }
)
PUBLIC_WS_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "advanced-trade-ws.coinbase.com",
        "ws-feed.exchange.coinbase.com",
        "ws-feed-public.sandbox.exchange.coinbase.com",
    }
)
PUBLIC_WS_CHANNELS: Final[frozenset[str]] = frozenset(
    {"level2", "market_trades", "heartbeats", "ticker", "ticker_batch"}
)

_PUBLIC_PATHS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p)
    for p in (
        r"^/products$",
        r"^/products/[A-Z0-9]+-[A-Z0-9]+$",
        r"^/products/[A-Z0-9]+-[A-Z0-9]+/(book|trades|candles|ticker|stats)$",
        r"^/time$",
    )
)


class LiveTradingDisabledError(RuntimeError):
    """Raised on any attempt to cross the simulation/live boundary."""


def assert_public_rest(url: str, method: str = "GET") -> None:
    if method.upper() != "GET":
        raise LiveTradingDisabledError(f"HTTP {method} is not permitted (read-only market data)")
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in PUBLIC_REST_HOSTS:
        raise LiveTradingDisabledError(f"host not in public allowlist: {u.hostname!r}")
    if not any(p.match(u.path) for p in _PUBLIC_PATHS):
        raise LiveTradingDisabledError(f"path not in public market-data allowlist: {u.path!r}")


def assert_public_ws(url: str, channels: list[str]) -> None:
    u = urlparse(url)
    if u.scheme != "wss" or u.hostname not in PUBLIC_WS_HOSTS:
        raise LiveTradingDisabledError(f"websocket host not in public allowlist: {u.hostname!r}")
    bad = [c for c in channels if c not in PUBLIC_WS_CHANNELS]
    if bad:
        raise LiveTradingDisabledError(f"non-public websocket channels requested: {bad}")


class LiveOrderGateway:
    """Placeholder that documents the boundary. There is no live implementation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise LiveTradingDisabledError(
            "crypto-execution-lab never sends orders to a real venue; use execlab.sim.SimExchange"
        )
