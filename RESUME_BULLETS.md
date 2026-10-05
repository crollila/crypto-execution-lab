# Resume bullets — crypto-execution-lab

Pick 2–4. Every number is reproducible with `python -m execlab reproduce` (see ANALYSIS.md).

**Crypto Execution Lab (Python, NumPy, pandas, asyncio/WebSockets)** · github.com/crollila/crypto-execution-lab

- Built a CEX execution research stack for Coinbase BTC-USD/ETH-USD: public REST and WebSocket L2 adapters with sequence-gap resync, stale-feed detection and jittered exponential-backoff reconnect, plus a delta-encoded capture format. Recorded 30 min of live L2 and prints (63.6k messages, 0 gaps) for deterministic offline replay.
- Wrote an event-driven matching simulator with FIFO queue-position modeling, partial fills, maker/taker fees, IOC/FOK/post-only/reduce-only semantics, lognormal order-entry latency with cancel/fill races, and disconnect handling with and without cancel-on-disconnect. Covered it with 72 deterministic tests, including a P&L-ledger accounting identity.
- Designed TCA (arrival slippage, Perold and completion-adjusted implementation shortfall, fill rate, maker adverse-selection markouts) and ran 9,100+ paired simulations across calibrated volatility/liquidity regimes, order sizes, six fee tiers, latency and outage scenarios.
- Showed on real Coinbase data that a $100k order's cost is fee-dominated: aggressive costs 18.7 bp vs 8.8 bp passive, with spread plus impact under 1 bp. Bounded passive's true edge at 3–10 bp depending on fill-probability modeling, and identified the size/liquidity point where passive execution loses (BTC $1M in a thin book: +1.9 bp vs crossing).
- Enforced "no live trading" by construction: GET-only transport, public host/path/channel allowlists, no signing code (checked by a source-scanning test). Shipped one-command, bit-for-bit reproducible results with CI that re-runs and byte-compares the pipeline.

**One-liner:** Coinbase L2 market-data adapters plus a queue/latency/fee-aware execution simulator and TCA; quantified passive vs aggressive cost across regimes on real and calibrated synthetic data.
