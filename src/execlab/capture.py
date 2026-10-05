"""Record live public Coinbase market data to a replayable capture file."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .adapters.rest import CoinbaseRest
from .adapters.ws import CoinbaseWs
from .io import EventWriter


def capture(
    products: list[str], seconds: float, out: str | Path, depth: int = 50, interval: float = 0.25
) -> dict:
    out = Path(out)
    rest = CoinbaseRest()
    specs = {p: rest.product(p) for p in products}
    with EventWriter(out) as w:
        ws = CoinbaseWs(products, w.write, depth=depth, snapshot_interval=interval)
        t0 = time.time()
        asyncio.run(ws.run(duration=seconds))
        meta = {
            "products": products,
            "start_utc": t0,
            "seconds": seconds,
            "depth": depth,
            "snapshot_interval": interval,
            "events": w.count,
            "ws_stats": ws.stats,
            "specs": {p: {"tick": s.tick, "lot": s.lot, "min_notional": s.min_notional} for p, s in specs.items()},
            "source": "Coinbase Advanced Trade public WebSocket (level2, market_trades)",
        }
    out.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta
