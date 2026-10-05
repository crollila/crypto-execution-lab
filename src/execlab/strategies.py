"""Parent-order execution strategies: aggressive, passive, and passive-then-aggressive.

Strategies only ever talk to `SimExchange`. They see market data through the runner,
which withholds it while the session is disconnected (the strategy is "blind").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .sim.exchange import TIF, Order, OrderStatus, OrderType, SimExchange
from .types import BookSnapshot, Event, ProductSpec, Side, Status


@dataclass(frozen=True)
class ParentOrder:
    product: str
    side: Side
    qty: float
    start_ts: float
    horizon_s: float

    @property
    def deadline(self) -> float:
        return self.start_ts + self.horizon_s


class ExecStrategy:
    name = "base"

    def __init__(self, parent: ParentOrder, spec: ProductSpec) -> None:
        self.parent, self.spec = parent, spec
        self.children: list[Order] = []
        self.book: BookSnapshot | None = None
        self.blind = False
        self.done = False

    # -- bookkeeping ---------------------------------------------------------
    @property
    def filled(self) -> float:
        return sum(o.filled for o in self.children)

    @property
    def working(self) -> list[Order]:
        return [o for o in self.children if o.active]

    def unallocated(self) -> float:
        """Quantity not filled and not committed to a live child (prevents overfill)."""
        q = self.parent.qty - self.filled - sum(o.remaining for o in self.working)
        return self.spec.round_qty(max(0.0, q))

    def _min_qty(self) -> float:
        ref = self.book.mid if self.book else 1.0
        return max(self.spec.lot, self.spec.min_notional / ref)

    def _send(self, ex: SimExchange, **kw: object) -> Order:
        o = ex.submit(self.parent.product, self.parent.side, **kw)  # type: ignore[arg-type]
        self.children.append(o)
        return o

    def _cancel_all(self, ex: SimExchange) -> None:
        for o in self.working:
            if o.status is OrderStatus.OPEN and not o.cancel_requested:
                ex.cancel(o.id)

    # -- runner hooks --------------------------------------------------------
    def start(self, ex: SimExchange, book: BookSnapshot) -> None:
        self.book = book
        self.on_tick(ex)

    def on_event(self, ex: SimExchange, ev: Event) -> None:
        if isinstance(ev, Status):
            if ev.kind == "disconnect":
                self.blind = True
            elif ev.kind == "reconnect":
                self.blind = False
            return
        if isinstance(ev, BookSnapshot):
            self.book = ev
        if not self.blind:
            self.on_tick(ex)

    def on_tick(self, ex: SimExchange) -> None:
        if self.done:
            # keep re-sending cancels that lost the race against their order
            self._cancel_all(ex)
            return
        if ex.now >= self.parent.deadline or self.unallocated() + sum(o.remaining for o in self.working) <= 0:
            self._cancel_all(ex)
            self.done = True
            return
        self.step(ex)

    def step(self, ex: SimExchange) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class Aggressive(ExecStrategy):
    """Marketable IOC limit with a price collar; retry the remainder every `retry_s`."""

    name = "aggressive"

    def __init__(self, parent: ParentOrder, spec: ProductSpec, collar_bps: float = 25.0, retry_s: float = 1.0):
        super().__init__(parent, spec)
        self.collar_bps, self.retry_s = collar_bps, retry_s
        self._next_try = -math.inf

    def step(self, ex: SimExchange) -> None:
        if self.working or ex.now < self._next_try or self.book is None:
            return
        qty = self.unallocated()
        if qty < self._min_qty():
            return
        s = self.parent.side
        touch = self.book.best_ask if s is Side.BUY else self.book.best_bid
        px = self.spec.round_price(touch * (1 + s.sign * self.collar_bps / 1e4), s.opposite)
        self._send(ex, qty=qty, order_type=OrderType.LIMIT, limit_price=px, tif=TIF.IOC)
        self._next_try = ex.now + self.retry_s


class Passive(ExecStrategy):
    """Post-only limit joining the touch; cancel/replace when the touch runs away.

    No completion guarantee: whatever is unfilled at the deadline is opportunity cost.
    """

    name = "passive"

    def __init__(
        self,
        parent: ParentOrder,
        spec: ProductSpec,
        reprice_bps: float = 0.5,
        min_requote_s: float = 0.5,
        improve_if_spread_ticks: int = 3,
    ):
        super().__init__(parent, spec)
        self.reprice_bps, self.min_requote_s = reprice_bps, min_requote_s
        self.improve_if_spread_ticks = improve_if_spread_ticks
        self._last_quote = -math.inf

    def target_price(self) -> float:
        assert self.book is not None
        s, tick = self.parent.side, self.spec.tick
        touch = self.book.best_bid if s is Side.BUY else self.book.best_ask
        if self.book.spread >= self.improve_if_spread_ticks * tick - 1e-12:
            touch += s.sign * tick  # step inside a wide spread to get queue priority
        return self.spec.round_price(touch, s)

    def passive_step(self, ex: SimExchange) -> None:
        if self.book is None:
            return
        tgt = self.target_price()
        live = self.working
        if live:
            o = live[0]
            if o.status is OrderStatus.OPEN and not o.cancel_requested:
                gap_bps = abs(tgt - float(o.limit_price)) / self.book.mid * 1e4  # type: ignore[arg-type]
                if gap_bps > self.reprice_bps and ex.now - self._last_quote >= self.min_requote_s:
                    ex.cancel(o.id)
            return
        qty = self.unallocated()
        if qty < self._min_qty():
            return
        self._send(ex, qty=qty, order_type=OrderType.LIMIT, limit_price=tgt, tif=TIF.GTC, post_only=True)
        self._last_quote = ex.now

    def step(self, ex: SimExchange) -> None:
        self.passive_step(ex)


class PassiveThenAggressive(Passive):
    """Rest passively for `switch_frac` of the horizon, then cross for the remainder."""

    name = "hybrid"

    def __init__(self, parent: ParentOrder, spec: ProductSpec, switch_frac: float = 0.7, collar_bps: float = 25.0, **kw: float):
        super().__init__(parent, spec, **kw)  # type: ignore[arg-type]
        self.switch_ts = parent.start_ts + switch_frac * parent.horizon_s
        self._aggr = Aggressive(parent, spec, collar_bps=collar_bps)
        self._aggr.children = self.children  # share child list for accounting

    def step(self, ex: SimExchange) -> None:
        if ex.now < self.switch_ts:
            self.passive_step(ex)
            return
        if any(o.post_only for o in self.working):
            self._cancel_all(ex)
            return
        self._aggr.book = self.book
        self._aggr.step(ex)


STRATEGIES: dict[str, type[ExecStrategy]] = {
    "aggressive": Aggressive,
    "passive": Passive,
    "hybrid": PassiveThenAggressive,
}
