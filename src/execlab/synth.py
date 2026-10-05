"""Seeded synthetic L2 market generator with volatility/liquidity regimes.

A latent fair price follows a jump-diffusion. Liquidity providers quote a ladder around
it; when fair value moves through a stale quote, that quote is either picked off by an
informed taker (a print — this is the adverse-selection channel for resting orders) or
cancelled. Uninformed takers arrive as a Poisson process with lognormal notional and
walk the book. Level sizes diffuse, randomly cancel, and refill at a regime-dependent
rate, so queue-ahead, spread and depth all evolve endogenously.

The generator is independent of our orders (no feedback), which lets every strategy be
evaluated on the *same* path (common random numbers -> tight paired comparisons).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from .types import BookSnapshot, Event, ProductSpec, Side, Trade


@dataclass(frozen=True)
class RegimeParams:
    name: str
    sigma_bps_sqrt_s: float = 0.9  # diffusion vol of fair value, bps per sqrt(second)
    jump_rate_per_min: float = 0.0
    jump_bps: float = 0.0
    half_spread_bps: float = 0.01  # LP quote half-width around fair (touch is tick-rounded)
    level_spacing_bps: float = 0.1  # distance between ladder levels behind the touch
    n_levels: int = 120
    level_usd: float = 15_000.0  # median displayed notional per level
    depth_slope: float = 0.01  # size growth per level index
    refill_per_s: float = 4.0  # rate a missing ladder level gets re-posted
    cancel_per_s: float = 0.15  # per-level random cancel rate
    size_vol: float = 0.25  # per-sqrt(s) lognormal diffusion of level sizes
    pickoff_frac: float = 0.4  # P(stale quote is hit rather than cancelled)
    noise_trades_per_s: float = 4.0
    noise_trade_usd: float = 300.0  # median uninformed taker notional
    noise_trade_sigma: float = 1.6


def regime_grid(base: RegimeParams) -> dict[str, RegimeParams]:
    """2x2 grid: {calm, volatile} x {deep, thin} around a calibrated base."""
    volatile = dict(
        sigma_bps_sqrt_s=base.sigma_bps_sqrt_s * 4.0,
        jump_rate_per_min=2.0,
        jump_bps=8.0,
        noise_trades_per_s=base.noise_trades_per_s * 2.0,
    )
    thin = dict(
        half_spread_bps=max(base.half_spread_bps, 1.5),
        level_spacing_bps=base.level_spacing_bps * 4.0,
        level_usd=base.level_usd * 0.25,
        refill_per_s=base.refill_per_s * 0.25,
        n_levels=max(40, base.n_levels // 3),
    )
    return {
        "calm_deep": replace(base, name="calm_deep"),
        "volatile_deep": replace(base, name="volatile_deep", **volatile),
        "calm_thin": replace(base, name="calm_thin", **thin),
        "volatile_thin": replace(base, name="volatile_thin", **{**volatile, **thin}),
    }


def generate(
    spec: ProductSpec,
    params: RegimeParams,
    p0: float,
    seconds: float,
    seed: int,
    dt: float = 0.1,
    depth: int = 100,
    t0: float = 0.0,
) -> list[Event]:
    rng = np.random.default_rng(seed)
    tick = spec.tick
    sp = params
    spacing = max(tick, round(p0 * sp.level_spacing_bps / 1e4 / tick) * tick)
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    events: list[Event] = []
    logp = math.log(p0)
    sq = math.sqrt(dt)
    tid = 0

    def rt(x: float) -> float:
        return round(round(x / tick) * tick, 10)

    def level_qty(k: int, p: float) -> float:
        usd = sp.level_usd * (1 + sp.depth_slope * k) * math.exp(0.6 * rng.standard_normal() - 0.18)
        return spec.round_qty(max(usd / p, spec.lot * 100))

    def quote(p: float, fill_all: bool) -> None:
        bb = rt(math.floor(p * (1 - sp.half_spread_bps / 1e4) / tick) * tick)
        ba = rt(math.ceil(p * (1 + sp.half_spread_bps / 1e4) / tick) * tick)
        if ba <= bb:
            ba = rt(bb + tick)
        prob = 1.0 if fill_all else 1.0 - math.exp(-sp.refill_per_s * dt)
        for k in range(sp.n_levels):
            pb, pa = rt(bb - k * spacing), rt(ba + k * spacing)
            if pb not in bids and pb > 0 and (k == 0 and not bids or rng.random() < prob):
                bids[pb] = level_qty(k, p)
            if pa not in asks and (k == 0 and not asks or rng.random() < prob):
                asks[pa] = level_qty(k, p)

    def emit_trade(ts: float, px: float, q: float, aggr: Side) -> None:
        nonlocal tid
        if q <= 0:
            return
        tid += 1
        events.append(Trade(ts, spec.product, px, spec.round_qty(q) or spec.lot, aggr, str(tid)))

    quote(p0, fill_all=True)
    n_steps = int(round(seconds / dt))
    for i in range(1, n_steps + 1):
        t = round(t0 + i * dt, 6)
        dlog = sp.sigma_bps_sqrt_s / 1e4 * sq * rng.standard_normal()
        if sp.jump_rate_per_min > 0:
            nj = rng.poisson(sp.jump_rate_per_min / 60 * dt)
            if nj:
                dlog += sp.jump_bps / 1e4 * float(rng.standard_normal(nj).sum())
        logp += dlog
        p = math.exp(logp)
        sub = sorted(round(float(x), 6) for x in rng.uniform(t - dt, t, size=8))  # intra-step print times
        si = 0

        # 1) stale quotes vs. new fair value: picked off (print) or pulled
        for px in sorted(a for a in asks if a < p):
            q = asks.pop(px)
            if rng.random() < sp.pickoff_frac:
                emit_trade(sub[si % 8], px, q, Side.BUY)
                si += 1
        for px in sorted((b for b in bids if b > p), reverse=True):
            q = bids.pop(px)
            if rng.random() < sp.pickoff_frac:
                emit_trade(sub[si % 8], px, q, Side.SELL)
                si += 1

        # 2) uninformed takers walk the book
        for _ in range(rng.poisson(sp.noise_trades_per_s * dt)):
            aggr = Side.BUY if rng.random() < 0.5 else Side.SELL
            rem = sp.noise_trade_usd * math.exp(sp.noise_trade_sigma * rng.standard_normal()) / p
            book = asks if aggr is Side.BUY else bids
            ts = sub[si % 8]
            si += 1
            for px in sorted(book, reverse=aggr is Side.SELL):
                if rem <= spec.lot:
                    break
                q = book[px]
                take = min(q, rem)
                emit_trade(ts, px, take, aggr)
                rem -= take
                if q - take <= spec.lot:
                    del book[px]
                else:
                    book[px] = spec.round_qty(q - take)

        # 3) level-size diffusion and random cancels
        pc = 1.0 - math.exp(-sp.cancel_per_s * dt)
        for book in (bids, asks):
            for px in list(book):
                if rng.random() < pc:
                    del book[px]
                else:
                    book[px] = spec.round_qty(book[px] * math.exp(sp.size_vol * sq * rng.standard_normal()))
                    if book[px] <= spec.lot:
                        del book[px]

        # 4) liquidity providers re-post the ladder; trim the far tail
        quote(p, fill_all=False)
        cap = int(sp.n_levels * 1.5)
        if len(bids) > cap:
            for px in sorted(bids)[: len(bids) - cap]:
                del bids[px]
        if len(asks) > cap:
            for px in sorted(asks, reverse=True)[: len(asks) - cap]:
                del asks[px]

        bpx = sorted(bids, reverse=True)[:depth]
        apx = sorted(asks)[:depth]
        events.append(
            BookSnapshot(t, spec.product, tuple((x, bids[x]) for x in bpx), tuple((x, asks[x]) for x in apx))
        )
    events.sort(key=lambda e: (e.ts, 0 if isinstance(e, Trade) else 1))
    return events


def params_dict(p: RegimeParams) -> dict:
    return asdict(p)
