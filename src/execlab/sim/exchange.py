"""Event-driven exchange simulator driven by replayed (or synthetic) public market data.

Model summary (see README "Simulator assumptions"):

* Our orders are invisible to the replayed market: taker fills consume displayed depth
  in a *shadow* copy of the latest snapshot (restored when the next snapshot arrives);
  there is no permanent impact on the replayed path.
* Orders and cancels reach the matching engine after a sampled one-way latency and are
  evaluated against the book *at arrival*, so the market can move while in flight; a
  resting order can still fill while its cancel is in flight.
* Resting limit orders use a FIFO queue-position model: queue-ahead = displayed size at
  our price when we arrive; prints at our price consume queue-ahead first; level
  shrinkage on later snapshots is treated as cancellations (queue-ahead = min(queue,
  level)); prints *through* our price, or opposite quotes crossing our price, fill us up
  to the size that printed/crossed (`fill_on_cross=False` disables the quote-crossing
  rule, leaving a stricter prints-only fill model).
* Order types: MARKET, LIMIT x {GTC, IOC, FOK}; flags post_only (reject if marketable on
  arrival) and reduce_only (clip to the reducing position, reject if it would increase).
* Fees: maker/taker bps per `FeeSchedule`.
* Connectivity: `Status(disconnect)` blocks new requests; with `cancel_on_disconnect`
  the venue pulls resting orders and drops in-flight requests.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..ledger import Fill, Ledger
from ..types import BookSnapshot, Event, ProductSpec, Side, Status, Trade
from .fees import MID_TIER, FeeSchedule
from .latency import LatencyModel

EPS = 1e-12


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TIF(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    PENDING = "pending"  # sent, not yet at the matching engine
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"  # includes IOC/market remainders (may carry partial fills)
    REJECTED = "rejected"


@dataclass(eq=False)
class Order:
    id: int
    product: str
    side: Side
    qty: float
    type: OrderType
    limit_price: float | None
    tif: TIF
    post_only: bool
    reduce_only: bool
    submit_ts: float
    arrive_ts: float = math.nan
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0
    notional: float = 0.0
    fees: float = 0.0
    queue_ahead: float = 0.0
    reject_reason: str = ""
    cancel_requested: bool = False
    done_ts: float = math.nan
    fills: list[Fill] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled)

    @property
    def avg_price(self) -> float:
        return self.notional / self.filled if self.filled > 0 else math.nan

    @property
    def active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN)


class SimExchange:
    def __init__(
        self,
        specs: dict[str, ProductSpec],
        fees: FeeSchedule = MID_TIER,
        latency: LatencyModel | None = None,
        seed: int = 0,
        cancel_on_disconnect: bool = True,
        ledger: Ledger | None = None,
        fill_on_cross: bool = True,
    ) -> None:
        self.specs = specs
        self.fees = fees
        self.latency = latency if latency is not None else LatencyModel()
        self.rng = np.random.default_rng(seed)
        self.cancel_on_disconnect = cancel_on_disconnect
        self.fill_on_cross = fill_on_cross  # False = conservative "fills only from public prints"
        self.ledger = ledger if ledger is not None else Ledger()
        self.books: dict[str, BookSnapshot] = {}
        self._taken: dict[tuple[str, Side, float], float] = {}  # shadow depletion by our takes
        self.orders: dict[int, Order] = {}
        self._resting: dict[str, list[Order]] = {p: [] for p in specs}
        self._pending: list[tuple[float, int, str, int]] = []
        self._seq = 0
        self._next_id = 1
        self.connected: dict[str, bool] = {p: True for p in specs}
        self.now = -math.inf
        self.fill_listeners: list[Callable[[Order, Fill], None]] = []
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------ client API
    def submit(
        self,
        product: str,
        side: Side,
        qty: float,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        tif: TIF = TIF.GTC,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> Order:
        spec = self.specs[product]
        o = Order(
            id=self._next_id,
            product=product,
            side=side,
            qty=spec.round_qty(qty),
            type=order_type,
            limit_price=limit_price,
            tif=tif,
            post_only=post_only,
            reduce_only=reduce_only,
            submit_ts=self.now,
        )
        self._next_id += 1
        self.orders[o.id] = o
        self.stats["submitted"] += 1
        reason = self._validate(o, spec)
        if reason:
            self._reject(o, reason)
            return o
        o.arrive_ts = self.now + self.latency.sample(self.rng)
        self._push(o.arrive_ts, "new", o.id)
        return o

    def cancel(self, order_id: int) -> bool:
        o = self.orders[order_id]
        if not o.active or o.cancel_requested or not self.connected[o.product]:
            return False
        o.cancel_requested = True
        self.stats["cancel_requests"] += 1
        self._push(self.now + self.latency.sample(self.rng), "cancel", o.id)
        return True

    def open_orders(self, product: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.active and (product is None or o.product == product)]

    # ---------------------------------------------------------- market input
    def on_event(self, ev: Event) -> None:
        self.advance(ev.ts)
        self.now = max(self.now, ev.ts)
        if isinstance(ev, BookSnapshot):
            self.books[ev.product] = ev
            for key in [k for k in self._taken if k[0] == ev.product]:
                del self._taken[key]
            self._on_book(ev)
        elif isinstance(ev, Trade):
            self._on_trade(ev)
        elif isinstance(ev, Status):
            self._on_status(ev)

    def advance(self, ts: float) -> None:
        """Deliver in-flight requests whose arrival time is <= ts, in time order."""
        while self._pending and self._pending[0][0] <= ts:
            t, _, kind, oid = heapq.heappop(self._pending)
            self.now = max(self.now, t)
            o = self.orders[oid]
            if kind == "new":
                self._arrive(o)
            else:
                self._arrive_cancel(o)

    # -------------------------------------------------------------- internals
    def _push(self, t: float, kind: str, oid: int) -> None:
        self._seq += 1
        heapq.heappush(self._pending, (t, self._seq, kind, oid))

    def _validate(self, o: Order, spec: ProductSpec) -> str:
        if not self.connected[o.product]:
            return "disconnected"
        if o.qty <= 0:
            return "qty below lot size"
        if o.type is OrderType.LIMIT:
            if o.limit_price is None or o.limit_price <= 0:
                return "limit price required"
            if abs(o.limit_price / spec.tick - round(o.limit_price / spec.tick)) > 1e-6:
                return "price not on tick"
            if o.post_only and o.tif is not TIF.GTC:
                return "post_only requires GTC"
        else:
            if o.post_only:
                return "post_only invalid for market orders"
            if o.limit_price is not None:
                return "market orders take no price"
        ref = o.limit_price
        if ref is None:
            book = self.books.get(o.product)
            ref = book.mid if book else math.nan
        if ref == ref and o.qty * ref < spec.min_notional:
            return "below min notional"
        return ""

    def _reject(self, o: Order, reason: str) -> None:
        o.status, o.reject_reason, o.done_ts = OrderStatus.REJECTED, reason, self.now
        self.stats["rejected"] += 1
        self.stats["rejected:" + reason] += 1

    def _finish(self, o: Order) -> None:
        o.status = OrderStatus.FILLED if o.remaining <= EPS else OrderStatus.CANCELLED
        o.done_ts = self.now
        if o in self._resting[o.product]:
            self._resting[o.product].remove(o)

    def _effective(self, product: str, side: Side, price: float, qty: float) -> float:
        return max(0.0, qty - self._taken.get((product, side, price), 0.0))

    def _opposite_levels(self, o: Order, book: BookSnapshot) -> tuple[Side, tuple[tuple[float, float], ...]]:
        return (Side.SELL, book.asks) if o.side is Side.BUY else (Side.BUY, book.bids)

    def _marketable_qty(self, o: Order, book: BookSnapshot, limit: float) -> float:
        lside, levels = self._opposite_levels(o, book)
        tot = 0.0
        for px, q in levels:
            if (o.side is Side.BUY and px > limit + EPS) or (o.side is Side.SELL and px < limit - EPS):
                break
            tot += self._effective(o.product, lside, px, q)
        return tot

    def _take(self, o: Order, book: BookSnapshot, limit: float) -> None:
        lside, levels = self._opposite_levels(o, book)
        for px, q in levels:
            if o.remaining <= EPS:
                break
            if (o.side is Side.BUY and px > limit + EPS) or (o.side is Side.SELL and px < limit - EPS):
                break
            avail = self._effective(o.product, lside, px, q)
            if avail <= EPS:
                continue
            fq = min(o.remaining, avail)
            key = (o.product, lside, px)
            self._taken[key] = self._taken.get(key, 0.0) + fq
            self._fill(o, px, fq, "taker")

    def _fill(self, o: Order, px: float, qty: float, liquidity: str) -> None:
        qty = min(qty, o.remaining)
        if qty <= EPS:
            return
        fee = self.fees.fee(px * qty, liquidity)
        f = Fill(self.now, o.id, o.product, o.side, px, qty, liquidity, fee)
        o.filled += qty
        o.notional += px * qty
        o.fees += fee
        o.fills.append(f)
        self.ledger.on_fill(f)
        self.stats["fills_" + liquidity] += 1
        if o.remaining <= EPS and o.status is OrderStatus.OPEN:
            self._finish(o)
        for cb in self.fill_listeners:
            cb(o, f)

    def _arrive(self, o: Order) -> None:
        if o.status is not OrderStatus.PENDING:
            return
        if not self.connected[o.product] and self.cancel_on_disconnect:
            self._reject(o, "session dropped in flight")
            return
        book = self.books.get(o.product)
        if book is None or not book.bids or not book.asks:
            self._reject(o, "no market")
            return
        if o.reduce_only:
            pos = self.ledger.position(o.product).qty
            reducible = abs(pos) if pos * o.side.sign < 0 else 0.0
            reducible -= sum(
                x.remaining for x in self._resting[o.product] if x.reduce_only and x.side is o.side
            )
            reducible = self.specs[o.product].round_qty(max(0.0, reducible))
            if reducible <= EPS:
                self._reject(o, "reduce_only would increase position")
                return
            o.qty = min(o.qty, reducible)
        if o.type is OrderType.MARKET:
            self._take(o, book, math.inf if o.side is Side.BUY else -math.inf)
            if o.remaining > EPS:
                self.stats["market_depth_exhausted"] += 1
            self._finish(o)
            return
        limit = float(o.limit_price)  # type: ignore[arg-type]
        crosses = (o.side is Side.BUY and limit >= book.best_ask - EPS) or (
            o.side is Side.SELL and limit <= book.best_bid + EPS
        )
        if o.post_only and crosses:
            self._reject(o, "post_only would take liquidity")
            return
        if o.tif is TIF.FOK and self._marketable_qty(o, book, limit) + EPS < o.qty:
            self.stats["fok_killed"] += 1
            self._finish(o)
            return
        if crosses:
            self._take(o, book, limit)
        if o.remaining <= EPS or o.tif is not TIF.GTC:
            self._finish(o)
            return
        o.status = OrderStatus.OPEN
        o.queue_ahead = book.level_qty(o.side, limit)
        self._resting[o.product].append(o)

    def _arrive_cancel(self, o: Order) -> None:
        if o.status is OrderStatus.OPEN:
            self._finish(o)
            self.stats["cancelled"] += 1
        elif o.status is OrderStatus.PENDING:
            # cancel overtook its order (latency jitter): venue rejects unknown id; order
            # will still arrive and must be cancelled again by the client.
            o.cancel_requested = False
            self.stats["cancel_unknown_order"] += 1
        else:
            self.stats["cancel_too_late"] += 1

    def _on_book(self, book: BookSnapshot) -> None:
        for o in list(self._resting[book.product]):
            limit = float(o.limit_price)  # type: ignore[arg-type]
            if o.side is Side.BUY:
                crossing = sum(q for p, q in book.asks if p <= limit + EPS)
            else:
                crossing = sum(q for p, q in book.bids if p >= limit - EPS)
            if crossing > EPS and self.fill_on_cross:
                self.stats["fills_by_cross"] += 1
                self._fill(o, limit, min(o.remaining, crossing), "maker")
                if not o.active:
                    continue
            o.queue_ahead = min(o.queue_ahead, book.level_qty(o.side, limit))

    def _on_trade(self, t: Trade) -> None:
        for o in list(self._resting[t.product]):
            if o.side is t.aggressor:  # same-side aggressor never hits our resting order
                continue
            limit = float(o.limit_price)  # type: ignore[arg-type]
            through = (o.side is Side.BUY and t.price < limit - EPS) or (
                o.side is Side.SELL and t.price > limit + EPS
            )
            at = abs(t.price - limit) <= EPS
            if through:
                self.stats["fills_by_tradethrough"] += 1
                o.queue_ahead = 0.0
                self._fill(o, limit, t.size, "maker")
            elif at:
                avail = t.size - o.queue_ahead
                o.queue_ahead = max(0.0, o.queue_ahead - t.size)
                if avail > EPS:
                    self.stats["fills_by_queue"] += 1
                    self._fill(o, limit, avail, "maker")

    def _on_status(self, s: Status) -> None:
        if s.kind == "disconnect":
            self.connected[s.product] = False
            self.stats["disconnects"] += 1
            if self.cancel_on_disconnect:
                for o in list(self._resting[s.product]):
                    self._finish(o)
                    self.stats["cancelled_on_disconnect"] += 1
        elif s.kind == "reconnect":
            self.connected[s.product] = True
