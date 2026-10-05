"""Microstructure statistics from a capture, and a base synthetic regime fitted to them."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .synth import RegimeParams, generate
from .types import BookSnapshot, Event, ProductSpec, Side, Trade


def market_stats(events: Sequence[Event], product: str) -> dict[str, float]:
    books = [e for e in events if isinstance(e, BookSnapshot) and e.product == product and e.bids and e.asks]
    trades = [e for e in events if isinstance(e, Trade) and e.product == product]
    if len(books) < 10:
        raise ValueError(f"not enough book snapshots for {product}")
    span = books[-1].ts - books[0].ts
    spreads = np.array([b.spread_bps for b in books])
    d10 = np.array([b.depth_notional(10.0) for b in books])
    # Level density and size per level, measured inside the window the capture covers
    # (top-N levels may not reach 10 bps on BTC, so 10-bps depth can be truncated).
    dens, lvl_usd, width = [], [], []
    for b in books[:: max(1, len(books) // 400)]:
        m = b.mid
        w = min(10.0, (m - b.bids[-1][0]) / m * 1e4, (b.asks[-1][0] - m) / m * 1e4)
        if w <= 0:
            continue
        inside = [(p, q) for p, q in b.bids if p >= m * (1 - w / 1e4)]
        inside += [(p, q) for p, q in b.asks if p <= m * (1 + w / 1e4)]
        width.append(w)
        dens.append(len(inside) / (2 * w))
        lvl_usd.append(sum(p * q for p, q in inside) / max(1, len(inside)))
    # realized vol from mids sampled on a 5s grid (dampens bid-ask bounce)
    ts = np.array([b.ts for b in books])
    mids = np.array([b.mid for b in books])
    grid = np.arange(ts[0], ts[-1], 5.0)
    idx = np.searchsorted(ts, grid, side="right") - 1
    r = np.diff(np.log(mids[idx])) * 1e4
    sigma = float(r.std(ddof=1) / math.sqrt(5.0)) if len(r) > 2 else math.nan
    notionals = np.array([t.price * t.size for t in trades]) if trades else np.array([math.nan])
    # aggressor check: Coinbase reports the maker side; compare our inferred aggressor to
    # the quote rule (print above prevailing mid => buyer-initiated)
    agree = n = 0
    bts = ts
    for t in trades:
        i = int(np.searchsorted(bts, t.ts, side="right")) - 1
        if i < 0:
            continue
        m = mids[i]
        if abs(t.price - m) < 1e-9:
            continue
        n += 1
        agree += (t.price > m) == (t.aggressor is Side.BUY)
    return {
        "product": product,
        "seconds": float(span),
        "snapshots": len(books),
        "trades": len(trades),
        "mid_first": float(mids[0]),
        "mid_last": float(mids[-1]),
        "spread_bps_median": float(np.median(spreads)),
        "spread_bps_mean": float(spreads.mean()),
        "spread_bps_p95": float(np.percentile(spreads, 95)),
        "depth10_bid_usd_median": float(np.median(d10[:, 0])),
        "depth10_ask_usd_median": float(np.median(d10[:, 1])),
        "captured_width_bps_median": float(np.median(width)),
        "levels_per_bp_median": float(np.median(dens)),
        "usd_per_level_median": float(np.median(lvl_usd)),
        "sigma_bps_sqrt_s": sigma,
        "trades_per_s": len(trades) / span if span > 0 else math.nan,
        "trade_usd_median": float(np.nanmedian(notionals)),
        "trade_usd_log_sd": float(np.nanstd(np.log(notionals[notionals > 0]))) if trades else math.nan,
        "volume_usd_per_min": float(np.nansum(notionals) / span * 60) if span > 0 else math.nan,
        "aggressor_quote_rule_agreement": agree / n if n else math.nan,
    }


def fit_base_regime(stats: dict[str, float], spec: ProductSpec) -> RegimeParams:
    mid = stats["mid_last"]
    spacing = max(1.0 / max(stats["levels_per_bp_median"], 1e-6), spec.tick / mid * 1e4)
    # Trade sizes are heavy-tailed; fit the lognormal by matching the empirical *mean* to
    # median ratio (mean = median * exp(sigma^2 / 2)) so volume, not just counts, matches.
    mean_usd = stats["volume_usd_per_min"] / 60.0 / max(stats["trades_per_s"], 1e-9)
    median_usd = max(stats["trade_usd_median"], 1.0)
    sigma = math.sqrt(2.0 * math.log(max(mean_usd / median_usd, 1.01)))
    base = RegimeParams(
        name="base",
        sigma_bps_sqrt_s=float(np.clip(stats["sigma_bps_sqrt_s"], 0.2, 5.0)),
        half_spread_bps=max(0.5 * stats["spread_bps_median"], 0.5 * spec.tick / mid * 1e4),
        level_spacing_bps=spacing,
        n_levels=int(np.clip(15.0 / spacing, 40, 150)),
        level_usd=stats["usd_per_level_median"],
        noise_trades_per_s=float(np.clip(0.7 * stats["trades_per_s"], 0.05, 20.0)),
        noise_trade_usd=float(np.clip(median_usd, 1.0, 50_000.0)),
        noise_trade_sigma=float(np.clip(sigma, 0.3, 3.2)),
    )
    return calibrate_volume(base, spec, mid, stats["volume_usd_per_min"])


def _volume_per_min(events: Sequence[Event], seconds: float) -> float:
    return sum(t.price * t.size for t in events if isinstance(t, Trade)) / seconds * 60


def calibrate_volume(base: RegimeParams, spec: ProductSpec, p0: float, target_usd_per_min: float,
                     informed_share: float = 0.3, seconds: float = 300.0, seed: int = 12345) -> RegimeParams:
    """Match generated traded volume to the live tape: ~70% uninformed flow, ~30% informed
    pick-offs of stale quotes (the adverse-selection channel)."""
    v_noise = _volume_per_min(generate(spec, replace(base, pickoff_frac=0.0), p0, seconds, seed), seconds)
    v_pick = _volume_per_min(generate(spec, replace(base, pickoff_frac=1.0, noise_trades_per_s=0.0), p0,
                                      seconds, seed), seconds)
    rate = base.noise_trades_per_s
    if v_noise > 0:  # small correction only: counts and sizes already match the tape
        rate = float(np.clip(rate * (1 - informed_share) * target_usd_per_min / v_noise, 0.5 * rate, 2.0 * rate))
    frac = float(np.clip(informed_share * target_usd_per_min / v_pick, 0.0005, 0.8)) if v_pick > 0 else 0.0
    return replace(base, noise_trades_per_s=rate, pickoff_frac=frac)
