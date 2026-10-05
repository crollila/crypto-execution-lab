"""Drive one parent order through the simulator over a (replayed or synthetic) event stream."""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .ledger import Fill
from .sim.exchange import Order, SimExchange
from .sim.fees import MID_TIER, FeeSchedule
from .sim.latency import LatencyModel
from .strategies import ExecStrategy, ParentOrder
from .types import BookSnapshot, Event, ProductSpec, Status


@dataclass
class ExecResult:
    strategy: str
    parent: ParentOrder
    arrival: BookSnapshot
    fills: list[Fill]
    children: list[Order]
    mid_ts: np.ndarray
    mid_px: np.ndarray
    stats: dict[str, int] = field(default_factory=dict)
    blind_s: float = 0.0
    deadline_book: BookSnapshot | None = None  # last book at/before the deadline (cleanup pricing)
    taker_bps: float = 0.0

    def mid_at(self, t: float) -> float:
        i = int(np.searchsorted(self.mid_ts, t, side="right")) - 1
        return float(self.mid_px[max(i, 0)])


def run_parent(
    events: Sequence[Event],
    spec: ProductSpec,
    parent: ParentOrder,
    make_strategy: Callable[[ParentOrder, ProductSpec], ExecStrategy],
    fees: FeeSchedule = MID_TIER,
    latency: LatencyModel | None = None,
    seed: int = 0,
    cancel_on_disconnect: bool = True,
    markout_s: float = 30.0,
    warmup_s: float = 5.0,
    event_ts: Sequence[float] | None = None,
    fill_on_cross: bool = True,
) -> ExecResult | None:
    """Returns None if no book is available at the start time."""
    ts = event_ts if event_ts is not None else [e.ts for e in events]
    i0 = bisect.bisect_left(ts, parent.start_ts - warmup_s)
    i1 = bisect.bisect_right(ts, parent.deadline + markout_s)
    ex = SimExchange({spec.product: spec}, fees=fees, latency=latency, seed=seed,
                     cancel_on_disconnect=cancel_on_disconnect, fill_on_cross=fill_on_cross)
    strat = make_strategy(parent, spec)
    arrival: BookSnapshot | None = None
    mid_ts: list[float] = []
    mid_px: list[float] = []
    blind_since: float | None = None
    blind_s = 0.0
    deadline_book: BookSnapshot | None = None
    for k in range(i0, i1):
        ev = events[k]
        if ev.product != spec.product:
            continue
        ex.on_event(ev)
        if isinstance(ev, BookSnapshot) and ev.bids and ev.asks:
            mid_ts.append(ev.ts)
            mid_px.append(ev.mid)
            if ev.ts <= parent.deadline:
                deadline_book = ev
        if isinstance(ev, Status):
            if ev.kind == "disconnect" and blind_since is None:
                blind_since = ev.ts
            elif ev.kind == "reconnect" and blind_since is not None:
                if arrival is not None:
                    blind_s += max(0.0, min(ev.ts, parent.deadline) - max(blind_since, arrival.ts))
                blind_since = None
        if ev.ts < parent.start_ts:
            continue
        if arrival is None:
            if isinstance(ev, BookSnapshot) and ev.bids and ev.asks and blind_since is None:
                arrival = ev
                strat.start(ex, ev)
            continue
        if blind_since is not None and not isinstance(ev, Status):
            continue  # market data is not reaching us
        strat.on_event(ex, ev)
    if arrival is None:
        return None
    ex.advance(parent.deadline + markout_s)
    return ExecResult(
        strategy=strat.name,
        parent=parent,
        arrival=arrival,
        fills=list(ex.ledger.fills),
        children=strat.children,
        mid_ts=np.asarray(mid_ts),
        mid_px=np.asarray(mid_px),
        stats=dict(ex.stats),
        blind_s=blind_s,
        deadline_book=deadline_book,
        taker_bps=fees.taker_bps,
    )
