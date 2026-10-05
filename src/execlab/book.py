"""Incremental L2 order book maintained from snapshot + delta messages."""

from __future__ import annotations

import heapq

from .types import BookSnapshot, Side


class OrderBook:
    def __init__(self, product: str) -> None:
        self.product = product
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.initialized = False

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.initialized = False

    def apply_snapshot(
        self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]
    ) -> None:
        self.bids = {p: q for p, q in bids if q > 0}
        self.asks = {p: q for p, q in asks if q > 0}
        self.initialized = True

    def apply_update(self, side: Side, price: float, qty: float) -> None:
        levels = self.bids if side is Side.BUY else self.asks
        if qty <= 0:
            levels.pop(price, None)
        else:
            levels[price] = qty

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid, self.best_ask
        return bb is not None and ba is not None and bb >= ba

    def top(self, ts: float, depth: int) -> BookSnapshot:
        bid_px = heapq.nlargest(depth, self.bids)
        ask_px = heapq.nsmallest(depth, self.asks)
        return BookSnapshot(
            ts=ts,
            product=self.product,
            bids=tuple((p, self.bids[p]) for p in bid_px),
            asks=tuple((p, self.asks[p]) for p in ask_px),
        )
