"""Command line: `python -m execlab <command>`.

  reproduce   run every experiment offline from the committed capture (deterministic)
  capture     record live public Coinbase L2 + trades to a replay file
  snapshot    print a one-off REST book/trades snapshot (production or public sandbox)
  stats       microstructure statistics of a capture file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="execlab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reproduce", help="run all experiments + figures offline")
    r.add_argument("--quick", action="store_true", help="few seeds (CI smoke run)")
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--out", type=Path, default=Path("results"))
    r.add_argument("--figures", type=Path, default=Path("figures"))
    r.add_argument("--sample", type=Path, default=None)

    c = sub.add_parser("capture", help="record live public market data")
    c.add_argument("--products", default="BTC-USD,ETH-USD")
    c.add_argument("--seconds", type=float, default=300)
    c.add_argument("--depth", type=int, default=100)
    c.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("snapshot", help="REST book + recent trades")
    s.add_argument("--product", default="BTC-USD")
    s.add_argument("--sandbox", action="store_true", help="use the public sandbox market-data host")

    st = sub.add_parser("stats", help="microstructure stats of a capture")
    st.add_argument("path", type=Path)

    a = ap.parse_args(argv)
    if a.cmd == "reproduce":
        from .experiments import DEFAULT_SAMPLE, ReproConfig, reproduce

        cfg = ReproConfig(quick=a.quick, workers=a.workers, sample=a.sample or DEFAULT_SAMPLE,
                          out_dir=a.out, fig_dir=a.figures)
        print(json.dumps(reproduce(cfg), indent=2))
    elif a.cmd == "capture":
        from .capture import capture

        print(json.dumps(capture(a.products.split(","), a.seconds, a.out, depth=a.depth), indent=2))
    elif a.cmd == "snapshot":
        from .adapters.rest import PRODUCTION, SANDBOX, CoinbaseRest

        rest = CoinbaseRest(SANDBOX if a.sandbox else PRODUCTION)
        b = rest.book(a.product)
        tr = rest.trades(a.product, limit=5)
        print(f"{a.product} bid {b.best_bid} ask {b.best_ask} spread {b.spread_bps:.3f} bps "
              f"depth10 bid/ask ${b.depth_notional(10)[0]:,.0f} / ${b.depth_notional(10)[1]:,.0f}")
        for t in tr:
            print(f"  {t.ts:.3f} {t.aggressor.value:4s} {t.size:.8f} @ {t.price}")
    elif a.cmd == "stats":
        from .calibrate import market_stats
        from .io import read_events

        ev = read_events(a.path)
        for p in sorted({e.product for e in ev}):
            print(json.dumps(market_stats(ev, p), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
