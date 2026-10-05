"""Coinbase Advanced Trade public WebSocket adapter (level2 + market_trades + heartbeats).

Responsibilities:
* maintain an incremental L2 book per product and emit throttled top-N `BookSnapshot`s;
* normalize prints into `Trade`s (Coinbase's `side` is the maker side);
* detect connection-level `sequence_num` gaps and stale feeds, then resync by
  reconnecting (book is discarded until a fresh snapshot arrives);
* reconnect with capped exponential backoff + jitter, surfacing every disconnect /
  reconnect / gap as a `Status` event so downstream replay can model outages.

The socket factory and sleep are injectable so reconnect logic is unit-tested offline.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..book import OrderBook
from ..safety import assert_public_ws
from ..types import Event, Side, Status, Trade
from .rest import parse_ts

ADVANCED_TRADE = "wss://advanced-trade-ws.coinbase.com"
CHANNELS = ["level2", "market_trades", "heartbeats"]


class Resync(Exception):
    """Internal: force a reconnect + full book resync."""


class Backoff:
    def __init__(self, initial: float = 0.5, cap: float = 30.0, seed: int | None = None) -> None:
        self.initial, self.cap = initial, cap
        self.attempt = 0
        self._rng = random.Random(seed)

    def next(self) -> float:
        base = min(self.cap, self.initial * (2**self.attempt))
        self.attempt += 1
        return base * (0.5 + 0.5 * self._rng.random())  # "equal jitter"

    def reset(self) -> None:
        self.attempt = 0


def _default_connect(url: str) -> Any:
    import websockets

    return websockets.connect(url, max_size=2**25, ping_interval=20, ping_timeout=20)


class CoinbaseWs:
    def __init__(
        self,
        products: list[str],
        on_event: Callable[[Event], None],
        url: str = ADVANCED_TRADE,
        depth: int = 50,
        snapshot_interval: float = 0.25,
        stale_timeout: float = 15.0,
        connect: Callable[[str], Any] = _default_connect,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        backoff: Backoff | None = None,
        clock: Callable[[], float] = time.time,
        max_reconnects: int | None = None,
    ) -> None:
        assert_public_ws(url, CHANNELS)
        self.products, self.on_event, self.url = products, on_event, url
        self.depth, self.snapshot_interval, self.stale_timeout = depth, snapshot_interval, stale_timeout
        self._connect, self._sleep, self._clock = connect, sleep, clock
        self.backoff = backoff or Backoff()
        self.max_reconnects = max_reconnects
        self.books = {p: OrderBook(p) for p in products}
        self._last_emit = {p: 0.0 for p in products}
        self._last_seq: int | None = None
        self.stats = {"messages": 0, "gaps": 0, "reconnects": 0, "crossed_skipped": 0, "trades": 0}

    # ---------------------------------------------------------------- parsing
    def handle(self, msg: dict[str, Any]) -> None:
        """Process one decoded message. Raises `Resync` on a sequence gap."""
        self.stats["messages"] += 1
        if msg.get("type") == "error":
            raise Resync(f"server error: {msg.get('message')}")
        seq = msg.get("sequence_num")
        if seq is not None:
            if self._last_seq is not None and seq != self._last_seq + 1:
                self.stats["gaps"] += 1
                ts = self._msg_ts(msg)
                for p in self.products:
                    self.on_event(Status(ts, p, "gap", f"{self._last_seq}->{seq}"))
                raise Resync(f"sequence gap {self._last_seq}->{seq}")
            self._last_seq = seq
        ch = msg.get("channel")
        if ch == "l2_data":
            self._on_l2(msg)
        elif ch == "market_trades":
            self._on_trades(msg)

    @staticmethod
    def _msg_ts(msg: dict[str, Any]) -> float:
        ts = msg.get("timestamp")
        return parse_ts(ts) if ts else time.time()

    def _on_l2(self, msg: dict[str, Any]) -> None:
        msg_ts = self._msg_ts(msg)
        for ev in msg.get("events", []):
            pid = ev.get("product_id")
            book = self.books.get(pid)
            if book is None:
                continue
            ups = ev.get("updates", [])
            parsed = [
                (Side.BUY if u["side"] == "bid" else Side.SELL, float(u["price_level"]), float(u["new_quantity"]))
                for u in ups
            ]
            ev_ts = max((parse_ts(u["event_time"]) for u in ups if u.get("event_time")), default=msg_ts)
            if ev.get("type") == "snapshot":
                book.apply_snapshot(
                    [(p, q) for s, p, q in parsed if s is Side.BUY],
                    [(p, q) for s, p, q in parsed if s is Side.SELL],
                )
            elif book.initialized:
                for s, p, q in parsed:
                    book.apply_update(s, p, q)
            self._maybe_emit(book, ev_ts)

    def _maybe_emit(self, book: OrderBook, ts: float) -> None:
        if not book.initialized or not book.bids or not book.asks:
            return
        if ts - self._last_emit[book.product] < self.snapshot_interval:
            return
        if book.is_crossed():
            self.stats["crossed_skipped"] += 1
            return
        self._last_emit[book.product] = ts
        self.on_event(book.top(ts, self.depth))

    def _on_trades(self, msg: dict[str, Any]) -> None:
        for ev in msg.get("events", []):
            if ev.get("type") != "update":  # initial snapshot = historical prints, skip
                continue
            for t in ev.get("trades", []):
                if t["product_id"] not in self.books:
                    continue
                self.stats["trades"] += 1
                self.on_event(
                    Trade(
                        ts=parse_ts(t["time"]),
                        product=t["product_id"],
                        price=float(t["price"]),
                        size=float(t["size"]),
                        aggressor=Side(t["side"].lower()).opposite,
                        trade_id=str(t["trade_id"]),
                    )
                )

    # ------------------------------------------------------------- connection
    async def _session(self, deadline: float | None) -> None:
        self._last_seq = None
        for b in self.books.values():
            b.clear()
        async with self._connect(self.url) as ws:
            for ch in CHANNELS:
                sub: dict[str, Any] = {"type": "subscribe", "channel": ch}
                if ch != "heartbeats":
                    sub["product_ids"] = self.products
                await ws.send(json.dumps(sub))
            while deadline is None or self._clock() < deadline:
                timeout, bound_by_deadline = self.stale_timeout, False
                if deadline is not None and deadline - self._clock() <= timeout:
                    timeout, bound_by_deadline = max(0.01, deadline - self._clock()), True
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if bound_by_deadline:
                        return
                    raise Resync("stale feed") from None
                self.handle(json.loads(raw))
                self.backoff.reset()

    async def run(self, duration: float | None = None) -> None:
        deadline = None if duration is None else self._clock() + duration
        reconnects = 0
        while deadline is None or self._clock() < deadline:
            try:
                await self._session(deadline)
                return
            except Exception as exc:  # Resync, socket errors, ConnectionClosed, bad payloads
                now = self._clock()
                for p in self.products:
                    self.on_event(Status(now, p, "disconnect", type(exc).__name__ + ": " + str(exc)[:120]))
                reconnects += 1
                self.stats["reconnects"] = reconnects
                if self.max_reconnects is not None and reconnects > self.max_reconnects:
                    raise
                await self._sleep(self.backoff.next())
                for p in self.products:
                    self.on_event(Status(self._clock(), p, "reconnect", f"attempt {reconnects}"))
