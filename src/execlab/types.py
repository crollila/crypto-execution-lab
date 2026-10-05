"""Normalized market-data and order primitives shared by adapters, replay and the simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Union


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True, slots=True)
class ProductSpec:
    """Exchange trading rules for one instrument (Coinbase `/products/{id}`)."""

    product: str
    tick: float  # quote_increment
    lot: float  # base_increment
    min_notional: float = 1.0  # min_market_funds

    def round_price(self, px: float, side: Side | None = None) -> float:
        """Round to tick. With a side, round *passively* (bids down, asks up)."""
        n = px / self.tick
        if side is Side.BUY:
            n = math.floor(n + 1e-9)
        elif side is Side.SELL:
            n = math.ceil(n - 1e-9)
        else:
            n = round(n)
        return round(n * self.tick, 10)

    def round_qty(self, qty: float) -> float:
        return round(math.floor(qty / self.lot + 1e-9) * self.lot, 10)


@dataclass(frozen=True, slots=True)
class Trade:
    """A public print. `aggressor` is the taker side (Coinbase reports the maker side)."""

    ts: float
    product: str
    price: float
    size: float
    aggressor: Side
    trade_id: str = ""


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Top-of-book L2 snapshot. bids descending, asks ascending; (price, size) tuples."""

    ts: float
    product: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else math.nan

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else math.nan

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        return self.spread / self.mid * 1e4

    def depth_notional(self, bps: float) -> tuple[float, float]:
        """Quote-currency depth within `bps` of mid on (bid side, ask side)."""
        m = self.mid
        lo, hi = m * (1 - bps / 1e4), m * (1 + bps / 1e4)
        bid = sum(p * q for p, q in self.bids if p >= lo)
        ask = sum(p * q for p, q in self.asks if p <= hi)
        return bid, ask

    def level_qty(self, side: Side, price: float) -> float:
        levels = self.bids if side is Side.BUY else self.asks
        for p, q in levels:
            if abs(p - price) < 1e-9:
                return q
        return 0.0


@dataclass(frozen=True, slots=True)
class Status:
    """Connectivity event: 'disconnect', 'reconnect', 'gap' (sequence gap -> resync)."""

    ts: float
    product: str
    kind: str
    detail: str = ""


Event = Union[Trade, BookSnapshot, Status]
