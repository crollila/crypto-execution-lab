"""Experiment harness: passive vs aggressive execution across regimes, sizes, fees, latency.

Every synthetic path is generated once and all strategies x sides run on it (common
random numbers), so strategy differences are measured as *paired* differences.
All randomness flows from explicit integer seeds -> results are bit-for-bit reproducible.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrate import fit_base_regime, market_stats
from .io import read_events, sort_key
from .runner import run_parent
from .sim.fees import MAKER_REBATE, MID_TIER, RETAIL, TOP_TIER, FeeSchedule, coinbase_tier
from .sim.latency import LatencyModel, OutageModel
from .strategies import STRATEGIES, ParentOrder
from .synth import RegimeParams, generate, regime_grid
from .tca import analyze
from .types import BookSnapshot, Event, ProductSpec, Side, Trade

PRODUCTS = ("BTC-USD", "ETH-USD")
DEFAULT_SPECS = {
    "BTC-USD": ProductSpec("BTC-USD", 0.01, 1e-8, 1.0),
    "ETH-USD": ProductSpec("ETH-USD", 0.01, 1e-8, 1.0),
}
DEFAULT_SAMPLE = Path(__file__).resolve().parents[2] / "data" / "samples" / "coinbase_btc_eth_20261005.jsonl.gz"


@dataclass(frozen=True)
class RunConfig:
    strategy: str
    label: str = ""
    latency: LatencyModel = field(default_factory=LatencyModel)
    outage: OutageModel = field(default_factory=OutageModel)
    cancel_on_disconnect: bool = True
    fees: FeeSchedule = MID_TIER
    fill_on_cross: bool = True

    @property
    def tag(self) -> str:
        return self.label or self.strategy


@dataclass(frozen=True)
class PathTask:
    experiment: str
    spec: ProductSpec
    regime: RegimeParams
    p0: float
    seed: int
    notional: float
    horizon_s: float
    configs: tuple[RunConfig, ...]
    warmup_s: float = 10.0
    markout_s: float = 30.0


def path_stats(events: Sequence[Event], t0: float, t1: float) -> dict[str, float]:
    """What the market *actually* looked like over [t0, t1] (measured, not parameters)."""
    books = [e for e in events if isinstance(e, BookSnapshot) and t0 <= e.ts <= t1 and e.bids and e.asks]
    trades = [e for e in events if isinstance(e, Trade) and t0 <= e.ts <= t1]
    if not books:
        return {}
    mids = np.array([b.mid for b in books])
    ts = np.array([b.ts for b in books])
    grid = np.arange(ts[0], ts[-1], 1.0)
    idx = np.searchsorted(ts, grid, side="right") - 1
    r = np.diff(np.log(mids[idx])) * 1e4 if len(idx) > 2 else np.array([np.nan])
    d = np.array([b.depth_notional(10.0) for b in books[::4]])
    return {
        "path_spread_bps_mean": float(np.mean([b.spread_bps for b in books])),
        "path_depth10_usd_median": float(np.median(d.sum(axis=1) / 2)),
        "path_vol_bps_sqrt_s": float(np.nanstd(r)),
        "path_trades_per_s": len(trades) / max(t1 - t0, 1e-9),
        "path_move_bps": float((mids[-1] / mids[0] - 1) * 1e4),
    }


def _run_configs(
    task_meta: dict[str, Any],
    events: Sequence[Event],
    spec: ProductSpec,
    start: float,
    notional: float,
    horizon_s: float,
    configs: Sequence[RunConfig],
    seed: int,
    warmup_s: float,
    markout_s: float,
) -> list[dict[str, Any]]:
    base_events = list(events)
    m0 = next(e.mid for e in base_events if isinstance(e, BookSnapshot) and e.ts >= start)
    qty = spec.round_qty(notional / m0)
    pstats = path_stats(base_events, start, start + horizon_s)
    rows = []
    rng = np.random.default_rng(seed + 7)
    outage_cache: dict[OutageModel, list[Event]] = {}
    for cfg in configs:
        evs: Sequence[Event] = base_events
        if cfg.outage.rate_per_min > 0:
            if cfg.outage not in outage_cache:
                st = cfg.outage.generate(spec.product, start, start + horizon_s, rng)
                merged = sorted(base_events + st, key=sort_key)
                outage_cache[cfg.outage] = merged
            evs = outage_cache[cfg.outage]
        ts = [e.ts for e in evs]
        for side in (Side.BUY, Side.SELL):
            parent = ParentOrder(spec.product, side, qty, start, horizon_s)
            res = run_parent(
                evs, spec, parent, STRATEGIES[cfg.strategy], fees=cfg.fees, latency=cfg.latency,
                seed=seed, cancel_on_disconnect=cfg.cancel_on_disconnect,
                markout_s=markout_s, warmup_s=warmup_s, event_ts=ts, fill_on_cross=cfg.fill_on_cross,
            )
            if res is None:
                continue
            row = {**task_meta, "config": cfg.tag, "latency_ms": cfg.latency.median_ms,
                   "outage_rate_per_min": cfg.outage.rate_per_min,
                   "cancel_on_disconnect": cfg.cancel_on_disconnect, "fee_schedule": cfg.fees.name,
                   "fill_model": "prints+quotes" if cfg.fill_on_cross else "prints-only",
                   "seed": seed, "start_ts": start, **pstats, **analyze(res)}
            rows.append(row)
    return rows


def run_path_task(task: PathTask) -> list[dict[str, Any]]:
    seconds = task.warmup_s + task.horizon_s + task.markout_s + 2.0
    events = generate(task.spec, task.regime, task.p0, seconds, seed=task.seed)
    meta = {"experiment": task.experiment, "regime": task.regime.name, "notional": task.notional}
    return _run_configs(meta, events, task.spec, task.warmup_s, task.notional, task.horizon_s,
                        task.configs, task.seed, task.warmup_s, task.markout_s)


def run_tasks(tasks: list[PathTask], workers: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if workers == 1 or len(tasks) < 4:
        for t in tasks:
            rows.extend(run_path_task(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for out in pool.map(run_path_task, tasks, chunksize=1):
                rows.extend(out)
    df = pd.DataFrame(rows)
    return df.sort_values(["experiment", "product", "regime", "notional", "seed", "config", "side"],
                          kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- experiments
def load_capture(path: Path) -> tuple[list[Event], dict[str, ProductSpec], dict[str, Any]]:
    events = read_events(path)
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    specs = dict(DEFAULT_SPECS)
    for p, s in meta.get("specs", {}).items():
        specs[p] = ProductSpec(p, s["tick"], s["lot"], s["min_notional"])
    return events, specs, meta


def calibrate_all(events: list[Event], specs: dict[str, ProductSpec]) -> dict[str, Any]:
    out = {}
    for p in PRODUCTS:
        st = market_stats(events, p)
        base = fit_base_regime(st, specs[p])
        out[p] = {"stats": st, "base_regime": asdict(base)}
    return out


GRID_CONFIGS = tuple(RunConfig(s) for s in STRATEGIES) + tuple(
    RunConfig(s, f"{s}|prints-only", fill_on_cross=False) for s in ("passive", "hybrid"))


def main_grid_tasks(cal: dict[str, Any], specs: dict[str, ProductSpec], n_seeds: int,
                    notional: float, horizon_s: float) -> list[PathTask]:
    configs = GRID_CONFIGS
    tasks = []
    for pi, p in enumerate(PRODUCTS):
        base = RegimeParams(**cal[p]["base_regime"])
        p0 = cal[p]["stats"]["mid_last"]
        for ri, rp in enumerate(regime_grid(base).values()):
            for i in range(n_seeds):
                seed = 1_000_000 * (pi + 1) + 10_000 * ri + i
                tasks.append(PathTask("regime_grid", specs[p], rp, p0, seed, notional, horizon_s, configs))
    return tasks


def size_sweep_tasks(cal: dict[str, Any], specs: dict[str, ProductSpec], n_seeds: int,
                     notionals: Sequence[float], horizon_s: float) -> list[PathTask]:
    configs = GRID_CONFIGS
    p = "BTC-USD"
    base = RegimeParams(**cal[p]["base_regime"])
    grid = regime_grid(base)
    tasks = []
    for ri, rname in enumerate(("calm_deep", "calm_thin")):
        for ni, n in enumerate(notionals):
            for i in range(n_seeds):
                seed = 5_000_000 + 100_000 * ri + 1_000 * ni + i
                tasks.append(PathTask("size_sweep", specs[p], grid[rname], cal[p]["stats"]["mid_last"],
                                      seed, n, horizon_s, configs))
    return tasks


def latency_outage_tasks(cal: dict[str, Any], specs: dict[str, ProductSpec], n_seeds: int,
                         notional: float, horizon_s: float) -> list[PathTask]:
    p = "BTC-USD"
    base = RegimeParams(**cal[p]["base_regime"])
    grid = regime_grid(base)
    outage = OutageModel(rate_per_min=3.0, mean_duration_s=4.0)
    configs: list[RunConfig] = []
    for s in STRATEGIES:
        for ms in (5.0, 40.0, 250.0):
            configs.append(RunConfig(s, f"{s}|lat{ms:g}ms", latency=LatencyModel(median_ms=ms)))
        configs.append(RunConfig(s, f"{s}|outage|cod", outage=outage, cancel_on_disconnect=True))
        configs.append(RunConfig(s, f"{s}|outage|no-cod", outage=outage, cancel_on_disconnect=False))
        configs.append(RunConfig(s, f"{s}|prints-only", fill_on_cross=False))
    tasks = []
    for ri, rname in enumerate(("volatile_deep", "volatile_thin")):
        for i in range(n_seeds):
            seed = 9_000_000 + 100_000 * ri + i
            tasks.append(PathTask("latency_outage", specs[p], grid[rname], cal[p]["stats"]["mid_last"],
                                  seed, notional, horizon_s, tuple(configs)))
    return tasks


def replay_experiment(events: list[Event], specs: dict[str, ProductSpec], notional: float,
                      horizon_s: float, spacing_s: float, markout_s: float = 30.0,
                      configs: Sequence[RunConfig] | None = None) -> pd.DataFrame:
    """Run every strategy on the *real* captured Coinbase tape at staggered start times."""
    rows: list[dict[str, Any]] = []
    configs = list(configs) if configs is not None else [RunConfig(s) for s in STRATEGIES]
    for pi, p in enumerate(PRODUCTS):
        evs = [e for e in events if e.product == p]
        books = [e for e in evs if isinstance(e, BookSnapshot)]
        if not books:
            continue
        t_first, t_last = books[0].ts, books[-1].ts
        start = t_first + 15.0
        k = 0
        while start + horizon_s + markout_s < t_last:
            window = [e for e in evs if start - 15.0 <= e.ts <= start + horizon_s + markout_s + 1]
            meta = {"experiment": "replay", "regime": "coinbase_live", "notional": notional, "window": k}
            rows.extend(_run_configs(meta, window, specs[p], start, notional, horizon_s, configs,
                                     seed=100 * pi + k, warmup_s=15.0, markout_s=markout_s))
            start += spacing_s
            k += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # label real windows by realized vol / depth relative to the product's median window
    for p in df["product"].unique():
        m = df["product"] == p
        v_med = df.loc[m, "path_vol_bps_sqrt_s"].median()
        d_med = df.loc[m, "path_depth10_usd_median"].median()
        df.loc[m, "regime"] = np.where(df.loc[m, "path_vol_bps_sqrt_s"] > v_med, "live_volatile", "live_calm") + \
            np.where(df.loc[m, "path_depth10_usd_median"] >= d_med, "_deep", "_thin")
    return df


FEE_SCHEDULES = {
    "retail(40/60)": RETAIL,
    "tier_100k(10/20)": coinbase_tier(100_000),
    "tier_1m(8/18)": MID_TIER,
    "tier_75m(3/12)": coinbase_tier(75_000_000),
    "top(0/5)": TOP_TIER,
    "rebate(-1/5)": MAKER_REBATE,
}


def fee_recost(df: pd.DataFrame) -> pd.DataFrame:
    """Fills don't depend on the fee schedule (no strategy is fee-aware), so IS under any
    schedule is exact from maker/taker notional: IS_f = IS - fee_cost + new_fees; the
    completion-adjusted IS also re-prices the cleanup taker fee."""
    out = []
    ex_fee = df["is_bps"] - df["fee_cost_bps"]
    for name, fs in FEE_SCHEDULES.items():
        fee = (df["maker_notional"] * fs.maker_bps + df["taker_notional"] * fs.taker_bps) / df["target_notional"]
        is_f = ex_fee + fee
        cleanup = df["cleanup_exec_bps"] + df["cleanup_notional"] * fs.taker_bps / df["target_notional"]
        out.append(df[["product", "regime", "strategy", "config", "seed", "side"]].assign(
            fee_schedule=name, is_bps=is_f, is_completed_bps=is_f + cleanup))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- aggregation
METRICS = [
    "arrival_spread_bps", "arrival_depth10_usd", "path_spread_bps_mean", "path_depth10_usd_median",
    "path_vol_bps_sqrt_s", "fill_rate", "maker_share", "slippage_bps", "fee_bps", "exec_cost_bps",
    "fee_cost_bps", "opportunity_cost_bps", "is_bps", "cleanup_exec_bps", "is_completed_bps",
    "adverse_sel_bps_1s_all", "adverse_sel_bps_5s_all", "adverse_sel_bps_30s_all", "adverse_sel_bps_1s_maker",
    "adverse_sel_bps_5s_maker", "adverse_sel_bps_30s_maker", "post_fill_drift_bps_5s_all",
    "time_to_last_fill_s", "n_children", "n_cancel_requests", "blind_s", "fills_after_deadline_qty",
    "maker_fills_by_queue", "maker_fills_by_tradethrough", "maker_fills_by_cross",
]


def summarize(df: pd.DataFrame, keys: list[str], metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row["n"] = len(g)
        for m in metrics:
            if m not in g:
                continue
            x = g[m].astype(float).dropna()
            row[m] = x.mean() if len(x) else math.nan
            row[m + "_ci95"] = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_diffs(df: pd.DataFrame, keys: list[str], base: str = "aggressive",
                 metric: str = "is_bps", col: str = "strategy") -> pd.DataFrame:
    rows = []
    idx = keys + ["seed", "side"]
    for k, g in df.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        piv = g.pivot_table(index=idx, columns=col, values=metric, aggfunc="first")
        if base not in piv:
            continue
        for other in piv.columns:
            if other == base:
                continue
            d = (piv[other] - piv[base]).dropna()
            if len(d) < 2:
                continue
            rows.append({**dict(zip(keys, k)), "comparison": f"{other} - {base}", "metric": metric,
                         "n_pairs": len(d), "mean_diff": d.mean(),
                         "ci95": 1.96 * d.std(ddof=1) / math.sqrt(len(d)),
                         "t_stat": d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std(ddof=1) > 0 else math.nan,
                         "frac_other_cheaper": float((d < 0).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- entry point
@dataclass
class ReproConfig:
    quick: bool = False
    workers: int | None = None
    sample: Path = DEFAULT_SAMPLE
    out_dir: Path = Path("results")
    fig_dir: Path = Path("figures")

    @property
    def n_seeds(self) -> int:
        return 3 if self.quick else 60


def reproduce(cfg: ReproConfig) -> dict[str, Any]:
    from . import figures

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    workers = cfg.workers or max(1, min(8, (os.cpu_count() or 2) - 1))

    events, specs, meta = load_capture(cfg.sample)
    cal = calibrate_all(events, specs)
    cal["capture_meta"] = meta
    (cfg.out_dir / "calibration.json").write_text(json.dumps(cal, indent=2, default=float))

    notional, horizon = 100_000.0, 60.0
    grid = run_tasks(main_grid_tasks(cal, specs, cfg.n_seeds, notional, horizon), workers)
    sizes = (10_000.0, 50_000.0, 250_000.0, 1_000_000.0) if not cfg.quick else (10_000.0, 250_000.0)
    sweep = run_tasks(size_sweep_tasks(cal, specs, max(2, cfg.n_seeds // 3), sizes, horizon), workers)
    lat = run_tasks(latency_outage_tasks(cal, specs, max(2, cfg.n_seeds // 2), notional, horizon), workers)
    replay = replay_experiment(events, specs, notional, horizon,
                               spacing_s=60.0 if not cfg.quick else 600.0)
    replay_fm = replay_experiment(
        events, specs, notional, horizon, spacing_s=60.0 if not cfg.quick else 600.0,
        configs=[RunConfig(s, f"{s}|prints-only", fill_on_cross=False) for s in ("passive", "hybrid")])
    fees = fee_recost(pd.concat([grid, replay], ignore_index=True))

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(cfg.out_dir / name, index=False, float_format="%.6g")

    save(grid, "runs_regime_grid.csv")
    save(sweep, "runs_size_sweep.csv")
    save(lat, "runs_latency_outage.csv")
    save(replay, "runs_replay.csv")
    save(replay_fm, "runs_replay_fillmodel.csv")

    s_grid = summarize(grid, ["product", "regime", "config"])
    s_replay = summarize(replay, ["product", "strategy"]) if not replay.empty else pd.DataFrame()
    s_replay_reg = summarize(replay, ["product", "regime", "strategy"]) if not replay.empty else pd.DataFrame()
    s_fm = summarize(pd.concat([replay, replay_fm], ignore_index=True), ["product", "strategy", "fill_model"])         if not replay.empty else pd.DataFrame()
    s_sweep = summarize(sweep, ["regime", "notional", "config"])
    s_lat = summarize(lat, ["regime", "config"])
    s_fee = summarize(fees, ["fee_schedule", "product", "regime", "config"], ["is_bps", "is_completed_bps"])
    d_grid = pd.concat([paired_diffs(grid, ["product", "regime"], metric=m, col="config")
                        for m in ("is_bps", "is_completed_bps", "adverse_sel_bps_5s_all", "fill_rate")],
                       ignore_index=True)
    d_replay = pd.concat([paired_diffs(replay, ["product"], metric=m)
                          for m in ("is_bps", "is_completed_bps")], ignore_index=True)         if not replay.empty else pd.DataFrame()
    d_fee = pd.concat([paired_diffs(fees, ["fee_schedule", "product", "regime"], metric=m, col="config")
                       for m in ("is_bps", "is_completed_bps")], ignore_index=True)
    for df, name in ((s_grid, "summary_regime_grid.csv"), (s_replay, "summary_replay.csv"),
                     (s_replay_reg, "summary_replay_by_regime.csv"), (s_fm, "summary_replay_fillmodel.csv"), (s_sweep, "summary_size_sweep.csv"),
                     (s_lat, "summary_latency_outage.csv"), (s_fee, "summary_fee_tiers.csv"),
                     (d_grid, "paired_regime_grid.csv"), (d_replay, "paired_replay.csv"),
                     (d_fee, "paired_fee_tiers.csv")):
        save(df, name)

    figures.make_all(cfg.fig_dir, events, grid, sweep, lat, replay, s_fee, cal)
    summary = {
        "n_runs": {"regime_grid": len(grid), "size_sweep": len(sweep), "latency_outage": len(lat),
                   "replay": len(replay), "replay_fillmodel": len(replay_fm)},
        "quick": cfg.quick,
        "workers": workers,
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
