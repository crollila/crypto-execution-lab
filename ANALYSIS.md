# Analysis: passive vs aggressive execution on Coinbase BTC-USD / ETH-USD

All numbers below come from `python -m execlab reproduce` (files in `results/`, figures in
`figures/`). "±" is a 95% confidence half-width. Costs are in bps of arrival-mid notional and
are **positive when they hurt**.

## 1. Question and setup

**Question.** For a $100k parent order that must complete within 60 seconds, is it cheaper to
cross the spread immediately (aggressive) or to rest post-only at the touch (passive), and how
does that depend on volatility, liquidity, size, fee tier, latency and connectivity?

**Strategies** (`src/execlab/strategies.py`)

* **aggressive**: marketable IOC limit with a 25 bp collar; any remainder is retried every 1 s.
* **passive**: post-only GTC limit joining the touch (stepping one tick inside if the spread is
  ≥ 3 ticks), cancel/replace when the touch moves > 0.5 bp away, rate-limited to one requote
  per 0.5 s. It has no completion guarantee.
* **hybrid**: passive for 70% of the horizon, then cancels and crosses the remainder.

**Markets**

* **Real tape (replay).** 30 minutes of live Coinbase L2 (top 100 levels, 4 Hz) plus every
  print, for BTC-USD and ETH-USD. Each strategy starts at the same 29 staggered start times
  per product (60 s apart; horizons overlap, so windows are not independent), on both sides:
  58 runs per strategy per product.
* **Synthetic regimes.** A seeded jump-diffusion L2 generator whose base regime is *fitted to
  the capture*: vol, spread, level density, size per level, trade rate, and trade-size
  distribution (lognormal fitted by mean/median), with traded volume matched to the tape
  (calm-deep generated $374k/min vs $279k/min real for BTC; $72k vs $89k for ETH). A 2×2 grid
  then stresses it: **volatile** = 4× diffusion vol + jumps; **thin** = ¼ size per level, 4×
  level spacing, ≥ 1.5 bp half-spread, slower refill. Measured outcomes (not parameters):

| BTC-USD regime | spread (mean) | depth ±10 bp/side | realized vol |
|---|---|---|---|
| calm_deep | 0.12 bp | $2.17M | 0.56 bp/√s |
| volatile_deep | 0.17 bp | $2.33M | 2.62 bp/√s |
| calm_thin | 2.07 bp | $0.42M | 0.32 bp/√s |
| volatile_thin | 2.67 bp | $0.38M | 2.31 bp/√s |

  There are 60 paths per regime and product, and every strategy and side runs on the *same*
  path with the *same* latency draws, so comparisons are paired.

**Metrics** (`src/execlab/tca.py`): spread and depth at arrival, fill rate, slippage of the
filled quantity vs arrival mid, fees, Perold implementation shortfall (IS), **completion-
adjusted IS**, and **adverse selection** (negative markout of each fill vs the mid 1/5/30 s
later; for maker fills this nets the spread captured against the subsequent move).

**Why completion-adjusted IS.** Perold IS charges unfilled size only for mid drift, which has
roughly zero mean. A strategy that barely trades therefore looks cheap: passive in calm_thin
ETH shows a Perold IS of 2.0 bp with a 27% fill rate. Completion-adjusted IS adds the cost of
crossing the unfilled remainder against the deadline book, plus the taker fee. That puts every
strategy on the same footing (all of them finish), and it is the headline metric below.

**Two fill models.** Whether a resting order fills when opposite *quotes* (not prints) move
through its price is not observable from public data. We report *prints+quotes*, which is
optimistic, and *prints-only*, which is conservative, as bounds. On the real tape, passive
maker fills split roughly evenly between queue depletion, trade-throughs, and quote crossings
(BTC: 3.6 / 6.5 / 5.2 fill events per run).

## 2. Results on the real Coinbase tape

`results/summary_replay.csv`, `results/summary_replay_fillmodel.csv`, `figures/replay_tape_and_tca.png`

| | BTC aggressive | BTC passive | BTC passive, prints-only | ETH aggressive | ETH passive | ETH passive, prints-only |
|---|---|---|---|---|---|---|
| spread at arrival | 0.02 bp | | | 0.24 bp | | |
| fill rate | 100% | 99% | 58% | 100% | 100% | 28% |
| slippage of fills | 0.72 bp | 0.71 bp | | 1.21 bp | 0.90 bp | |
| fees on fills | 18.0 bp | 8.0 bp | 8.0 bp | 18.0 bp | 8.0 bp | 8.0 bp |
| completion-adj. IS | **18.72 ± 0.08** | **8.82 ± 0.42** | **13.19 ± 1.45** | **19.21 ± 0.07** | **8.90 ± 0.57** | **16.35 ± 1.67** |
| maker adverse sel. @5 s | — | 0.69 bp | 0.98 bp | — | 0.85 bp | 0.92 bp |

Paired, passive minus aggressive: −9.89 ± 0.44 bp (BTC) and −10.31 ± 0.56 bp (ETH). Passive is
cheaper in 100% of paired runs under the optimistic fill model.

**Reading.** On a quiet Sunday-evening tape, BTC's spread is one cent (0.001 bp), so crossing
is almost free *before fees*. Even so, $100k walks about 0.7 bp into the book; ETH is about
1.2 bp. The 10 bp maker/taker fee gap therefore decides the outcome. How much of that gap
passive actually keeps depends on fill probability: between 10 bp (optimistic) and 3–5.5 bp
(conservative, BTC/ETH). The capture did not contain a genuinely thin or volatile market; the
"live_*" sub-labels split windows at the median and differ little (depth $1.78M vs $1.92M).
That is why the synthetic stress regimes exist.

## 3. Regime grid (synthetic, calibrated)

`results/summary_regime_grid.csv`, `results/paired_regime_grid.csv`, `figures/is_by_regime.png`,
`figures/cost_decomposition.png`, `figures/fill_model_bounds.png`

Completion-adjusted IS, $100k / 60 s, 120 runs per cell:

| | BTC aggr | BTC passive | BTC passive (prints-only) | ETH aggr | ETH passive | ETH passive (prints-only) |
|---|---|---|---|---|---|---|
| calm_deep | 18.17 | 8.23 (fill 100%) | 12.38 (60%) | 18.38 | 8.43 (100%) | 15.38 (30%) |
| volatile_deep | 18.31 | 8.83 (100%) | 10.28 (94%) | 18.66 | 9.35 (100%) | 12.52 (70%) |
| calm_thin | 20.03 | 13.96 (44%) | 13.96 (44%) | 21.21 | 17.13 (27%) | 17.13 (27%) |
| volatile_thin | 21.04 | 9.30 (93%) | 10.28 (77%) | 22.79 | 12.22 (74%) | 15.11 (45%) |

Findings:

1. **Aggressive cost is fee plus a small, liquidity-driven impact.** Slippage for $100k rises
   from 0.17 bp (BTC calm_deep) to 3.0 bp (BTC volatile_thin) and 4.8 bp (ETH volatile_thin).
   It is nearly deterministic: CIs are ≤ 0.3 bp.
2. **Thin books hurt passive through fill rate, not adverse selection.** In calm_thin, passive
   fills only 44% (BTC) and 27% (ETH) in 60 s. Those fills have *negative* adverse selection
   (−1.0 bp @5 s: they come from uninformed flow and earn the wider spread), but the remainder
   must be crossed in a 2 bp-wide book. ETH's paired edge falls from 10 bp to 4.1 ± 1.0 bp.
3. **Volatility hurts passive through adverse selection.** Maker adverse selection @5 s rises
   from 0.22 bp (BTC calm_deep) to 0.86 bp (volatile_deep), and to 1.65 bp for ETH
   volatile_deep under prints-only. Volatility also *raises* passive fill rates, because the
   price wanders through the order more often; the cost is that more of those fills are the
   picked-off kind (`figures/adverse_selection_horizons.png`).
4. **Hybrid never beat pure passive on completion-adjusted IS.** It pays the taker fee earlier
   on more quantity. Its value is operational: it completes inside the horizon by construction
   rather than relying on an end-of-horizon cleanup.

## 4. Size: where passive loses

`results/summary_size_sweep.csv`, `figures/size_sweep.png` (BTC, 20 paths × 2 sides per point)

| notional | calm_deep aggr | calm_deep passive (prints-only) | calm_thin aggr | calm_thin passive |
|---|---|---|---|---|
| $10k | 18.08 | 8.54 (fill 98%) | 19.18 | 7.07 (99%) |
| $50k | 18.12 | 10.06 (85%) | 19.64 | 10.39 (75%) |
| $250k | 18.26 | 15.38 (30%) | 20.86 | 17.47 (21%) |
| $1M | 18.57 | 17.32 (12%) | 21.62 | **23.54 (5%)** |

Under the conservative fill model, the passive edge decays with size as fill probability
collapses. At $1M in a thin book, passive is **worse** than crossing immediately, because it
burns 60 s of drift and then crosses an equally thin book anyway. The optimistic model never
shows this decay; it fills $1M in a calm_deep book against ~$374k/min of traded volume. That is
the clearest sign the optimistic model overstates passive fills for large orders.

## 5. Fee tier

`results/summary_fee_tiers.csv`, `results/paired_fee_tiers.csv`, `figures/fee_tiers.png`

Fills don't depend on fees (no strategy is fee-aware), so every run is re-priced exactly
under each schedule. ETH calm_deep, completion-adjusted IS:

| schedule (maker/taker) | aggressive | passive | passive (prints-only) |
|---|---|---|---|
| retail 40/60 | 60.4 | 40.4 | 54.4 |
| 1M tier 8/18 | 18.4 | 8.4 | 15.4 |
| 75M tier 3/12 | 12.4 | 3.4 | 9.7 |
| top 0/5 | 5.4 | 0.4 | 3.9 |
| rebate −1/5 | 5.4 | −0.6 | 3.6 |

The passive advantage scales with the taker−maker gap. At the top tier, where the gap is
5 bp, the conservative edge on ETH is only 1.5 bp, which is the same order as the adverse
selection plus cleanup risk. Within the tested range, no tier produced a full crossover at
$100k: the gap is never zero, and spreads are tiny.

## 6. Latency and connectivity

`results/summary_latency_outage.csv`, `figures/latency_outage.png` (BTC, volatile regimes, 60 paths)

* **Latency (5 → 40 → 250 ms) has a second-order effect on 60 s parent orders.** Aggressive IS
  barely moves (18.30 → 18.32 bp in volatile_deep), but its CI widens 6× at 250 ms
  (±0.05 → ±0.29) as the book moves while the IOC is in flight. For passive in volatile_thin,
  maker adverse selection @5 s rises from 0.42 bp to 2.30 bp at 250 ms (stale quotes get
  picked off more). Completion-adjusted IS differences stay inside their ±3 bp CIs.
* **Outages** (Poisson, 3/min, mean 4 s; ~15% of the horizon blind). With cancel-on-disconnect,
  passive fill rate drops from ~0.9 to 0.82 in volatile_thin (orders pulled, requoted later).
  Without it, resting orders keep filling while the strategy is blind, and maker adverse
  selection rises (0.51 → 0.76 bp). Neither changes completion-adjusted IS significantly at
  n = 60. This is a null result for this horizon and order size, not a claim that
  connectivity doesn't matter.

## 7. Conclusions

1. On Coinbase at retail-to-mid fee tiers, **execution cost for liquid pairs is mostly fees.**
   Spread plus impact for $100k is under 1 bp on the live tape; the fee gap is 10 bp.
2. **Passive is cheaper for small and medium orders in every regime tested.** The size of the
   saving is model-dependent: 10 bp under optimistic fills, 3–6 bp under prints-only fills on
   the real tape.
3. **Size and thin liquidity erode, then reverse, the passive edge** (BTC $1M in a thin book:
   +1.9 bp worse than aggressive). Volatility mostly shows up as adverse selection on maker
   fills.
4. **The biggest uncertainty is fill-probability modeling, not price impact.** The next step
   would be to calibrate the quote-crossing fill fraction against real order-entry data, for
   example by measuring the fill rate of a small post-only order in the public sandbox or in
   live production with proper controls.

## 8. Limitations

* One 30-minute capture, on a quiet weekend evening. Replay windows overlap, and real regime
  variety is narrow.
* No market impact on the replayed path, so impact for orders over about $1M is understated.
  The synthetic generator also has no feedback from our orders.
* The captured book is top-100 levels: about 7.5 bp deep for BTC and 10 bp for ETH. Sweeps
  beyond that are charged at the last visible price.
* Fees are illustrative, shaped like Coinbase Advanced Trade tiers; check current schedules.
* reduce_only is a derivatives concept. It is implemented and tested in the simulator but not
  exercised by these spot experiments.
* Strategies are deliberately simple: no alpha signal, no fee-aware or volatility-aware
  switching. The point is the measurement apparatus, not an optimized algo.
