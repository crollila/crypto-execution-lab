"""Static figures (PNG) for README / ANALYSIS. Deterministic given the result frames."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .types import BookSnapshot, Event, Trade  # noqa: E402

# Fixed categorical order (validated all-pairs for <=3 series); color follows the entity.
COLORS = {"aggressive": "#2a78d6", "passive": "#eb6834", "hybrid": "#1baf7a"}
STRATS = ["aggressive", "passive", "hybrid"]
REGIMES = ["calm_deep", "volatile_deep", "calm_thin", "volatile_thin"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
})


def _ci(x: pd.Series) -> float:
    x = x.dropna()
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _grouped_bars(ax: plt.Axes, df: pd.DataFrame, cat: str, cats: Sequence[str], metric: str) -> None:
    w = 0.26
    x = np.arange(len(cats))
    for i, s in enumerate(STRATS):
        sub = df[df["strategy"] == s]
        m = [sub.loc[sub[cat] == c, metric].mean() for c in cats]
        e = [_ci(sub.loc[sub[cat] == c, metric]) for c in cats]
        ax.bar(x + (i - 1) * w, m, w * 0.92, yerr=e, color=COLORS[s], label=s,
               error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
    ax.set_xticks(x, [c.replace("_", "\n") for c in cats])
    ax.axhline(0, color=INK2, lw=0.8)


def fig_is_by_regime(out: Path, grid: pd.DataFrame) -> None:
    prods = sorted(grid["product"].unique())
    fig, axes = plt.subplots(2, len(prods), figsize=(5.4 * len(prods), 6.6), sharey="row", squeeze=False)
    for col, p in enumerate(prods):
        for row, (metric, lab) in enumerate((("is_bps", "Perold IS"), ("is_completed_bps", "completion-adjusted IS"))):
            ax = axes[row, col]
            _grouped_bars(ax, grid[grid["product"] == p], "regime", REGIMES, metric)
            ax.set_title(f"{p}: {lab}, $100k over 60s")
            ax.set_ylabel("bps of arrival notional (incl. fees)")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "is_by_regime.png")
    plt.close(fig)


def fig_cost_decomposition(out: Path, grid: pd.DataFrame) -> None:
    comps = [("exec_cost_bps", "spread + impact (exec)"), ("fee_cost_bps", "fees"),
             ("opportunity_cost_bps", "opportunity (unfilled)"), ("cleanup_extra_bps", "cleanup")]
    hatches = ["", "//", "..", "xx"]
    grid = grid.assign(cleanup_extra_bps=grid["is_completed_bps"] - grid["is_bps"])
    g = grid.groupby(["regime", "strategy"])[[c for c, _ in comps]].mean()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xt, xl = [], []
    x = 0.0
    for r in REGIMES:
        for s in STRATS:
            if (r, s) not in g.index:
                continue
            pos = neg = 0.0
            for (c, _lab), h in zip(comps, hatches):
                v = g.loc[(r, s), c]
                base = pos if v >= 0 else neg
                ax.bar(x, v, 0.8, bottom=base, color=COLORS[s], hatch=h, edgecolor=SURFACE, lw=1.5,
                       alpha=1.0 if c == "exec_cost_bps" else 0.65)
                if v >= 0:
                    pos += v
                else:
                    neg += v
            ax.plot([x - 0.4, x + 0.4], [pos + neg] * 2, color=INK, lw=1.6)
            xt.append(x)
            xl.append(s[:4])
            x += 1
        x += 0.8
    ax.set_xticks(xt, xl, fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    for i, r in enumerate(REGIMES):
        ax.text(i * 3.8 + 1, ax.get_ylim()[1] * 0.98, r.replace("_", " "), ha="center", va="top", color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("bps of arrival notional")
    ax.set_title("Completion-adjusted IS decomposition (both products, both sides)\n"
                 "solid = spread+impact on fills, // = fees, .. = opportunity (drift on unfilled), "
                 "xx = cleanup cross at deadline; black tick = total", fontsize=9.5)
    fig.savefig(out / "cost_decomposition.png")
    plt.close(fig)


def fig_fill_vs_adverse(out: Path, grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _grouped_bars(axes[0], grid, "regime", REGIMES, "fill_rate")
    axes[0].set_title("Fill rate by the deadline")
    axes[0].set_ylabel("filled / target")
    axes[0].legend(loc="lower left")
    _grouped_bars(axes[1], grid, "regime", REGIMES, "adverse_sel_bps_5s_all")
    axes[1].set_title("Adverse selection: mid move against fills, 5s after")
    axes[1].set_ylabel("bps (positive = adverse)")
    fig.savefig(out / "fill_rate_adverse_selection.png")
    plt.close(fig)


def fig_markout_curves(out: Path, grid: pd.DataFrame) -> None:
    hs = ["1s", "5s", "30s"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, r in zip(axes, REGIMES):
        sub = grid[grid["regime"] == r]
        for s in STRATS:
            ss = sub[sub["strategy"] == s]
            y = [ss[f"adverse_sel_bps_{h}_all"].mean() for h in hs]
            e = [_ci(ss[f"adverse_sel_bps_{h}_all"]) for h in hs]
            ax.errorbar([1, 5, 30], y, yerr=e, color=COLORS[s], marker="o", ms=5, lw=2, label=s, capsize=2)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 30], hs)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(r.replace("_", " "))
        ax.set_xlabel("horizon after fill")
    axes[0].set_ylabel("adverse selection (bps)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Adverse selection = negative markout of fills vs mid h seconds later (both products)",
                 fontweight="bold", y=1.03)
    fig.savefig(out / "adverse_selection_horizons.png")
    plt.close(fig)


LINESTYLE = {"": "-", "prints-only": "--"}


def _cfg_style(cfg: str) -> tuple[str, str, str]:
    strat, _, variant = cfg.partition("|")
    return COLORS[strat], LINESTYLE.get(variant, "-"), cfg.replace("|", " ")


def fig_size_sweep(out: Path, sweep: pd.DataFrame) -> None:
    regs = [r for r in ("calm_deep", "calm_thin") if r in set(sweep["regime"])]
    fig, axes = plt.subplots(1, len(regs), figsize=(5.4 * len(regs), 3.8), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], regs):
        sub = sweep[sweep["regime"] == r]
        for cfg in ("aggressive", "passive", "passive|prints-only", "hybrid", "hybrid|prints-only"):
            g = sub[sub["config"] == cfg].groupby("notional")
            if not len(g):
                continue
            color, ls, label = _cfg_style(cfg)
            y = g["is_completed_bps"].mean()
            ax.errorbar(y.index, y, yerr=g["is_completed_bps"].apply(_ci), color=color, ls=ls, marker="o",
                        ms=5, lw=2, label=label, capsize=2)
            if cfg.startswith("passive"):
                for xx, yy, f in zip(y.index, y, g["fill_rate"].mean()):
                    ax.annotate(f"{f:.0%}", (xx, yy), textcoords="offset points", xytext=(5, -10 if ls == "-" else 6),
                                fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("parent notional (USD, log)")
        ax.set_title(f"BTC-USD {r.replace('_', ' ')}: cost vs size\n(labels = passive fill rate)", fontsize=9.5)
    axes[0, 0].set_ylabel("completion-adjusted IS (bps)")
    axes[0, 0].legend(fontsize=7.5)
    fig.savefig(out / "size_sweep.png")
    plt.close(fig)


def fig_fee_tiers(out: Path, s_fee: pd.DataFrame) -> None:
    order = ["retail(40/60)", "tier_100k(10/20)", "tier_1m(8/18)", "tier_75m(3/12)", "top(0/5)", "rebate(-1/5)"]
    panels = [("BTC-USD", "calm_deep"), ("ETH-USD", "calm_deep"), ("BTC-USD", "live_volatile_deep"),
              ("ETH-USD", "volatile_thin")]
    panels = [p for p in panels if ((s_fee["product"] == p[0]) & (s_fee["regime"] == p[1])).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (prod, r) in zip(axes[0], panels):
        sub = s_fee[(s_fee["regime"] == r) & (s_fee["product"] == prod)]
        for cfg in ("aggressive", "passive", "passive|prints-only"):
            ss = sub[sub["config"] == cfg].set_index("fee_schedule").reindex(order)
            if ss["is_completed_bps"].isna().all():
                continue
            color, ls, label = _cfg_style(cfg)
            ax.errorbar(range(len(order)), ss["is_completed_bps"], yerr=ss["is_completed_bps_ci95"], color=color,
                        ls=ls, marker="o", ms=5, lw=2, label=label, capsize=2)
        ax.set_xticks(range(len(order)), [o.replace("(", "\n(") for o in order], fontsize=7)
        ax.set_title(f"{prod} {r.replace('_', ' ')}", fontsize=9.5)
    axes[0, 0].set_ylabel("completion-adjusted IS (bps)")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Cost vs fee tier (maker/taker bps)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fee_tiers.png")
    plt.close(fig)


def fig_latency_outage(out: Path, lat: pd.DataFrame) -> None:
    variants = ["lat5ms", "lat40ms", "lat250ms", "outage|cod", "outage|no-cod", "prints-only"]
    regs = list(dict.fromkeys(lat["regime"]))
    fig, axes = plt.subplots(1, len(regs), figsize=(6 * len(regs), 3.6), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], regs):
        sub = lat[lat["regime"] == r].copy()
        sub["variant"] = sub["config"].str.split("|", n=1).str[1]
        w = 0.26
        x = np.arange(len(variants))
        for i, s in enumerate(STRATS):
            ss = sub[sub["strategy"] == s]
            m = [ss.loc[ss["variant"] == v, "is_completed_bps"].mean() for v in variants]
            e = [_ci(ss.loc[ss["variant"] == v, "is_completed_bps"]) for v in variants]
            ax.bar(x + (i - 1) * w, m, w * 0.92, yerr=e, color=COLORS[s], label=s,
                   error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
        ax.set_xticks(x, ["5ms", "40ms", "250ms", "outages\n+cancel", "outages\nno-cancel",
                          "prints-only\nfill model"], fontsize=8)
        ax.set_title(f"Latency / reconnect / fill model: {r.replace('_', ' ')} (BTC)")
    axes[0, 0].set_ylabel("completion-adjusted IS (bps)")
    axes[0, 0].legend(loc="upper left")
    fig.savefig(out / "latency_outage.png")
    plt.close(fig)


def fig_replay(out: Path, events: Sequence[Event], replay: pd.DataFrame) -> None:
    prods = ["BTC-USD", "ETH-USD"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for col, p in enumerate(prods):
        books = [e for e in events if isinstance(e, BookSnapshot) and e.product == p and e.bids and e.asks]
        trades = [e for e in events if isinstance(e, Trade) and e.product == p]
        if not books:
            continue
        t0 = books[0].ts
        ts = np.array([b.ts - t0 for b in books]) / 60
        ax = axes[0, col]
        ax.plot(ts, [b.mid for b in books], color=INK, lw=1.0, label="mid")
        if trades:
            buys = [t for t in trades if t.aggressor.value == "buy"]
            sells = [t for t in trades if t.aggressor.value == "sell"]
            ax.scatter([(t.ts - t0) / 60 for t in buys], [t.price for t in buys], s=5, color="#1baf7a",
                       label="buyer-initiated print", alpha=0.6, lw=0)
            ax.scatter([(t.ts - t0) / 60 for t in sells], [t.price for t in sells], s=5, color="#e34948",
                       label="seller-initiated print", alpha=0.6, lw=0)
        ax.set_title(f"{p}: captured Coinbase tape (public WS)")
        ax.set_xlabel("minutes")
        ax.legend(loc="best", markerscale=4, fontsize=7.5)
        ax2 = axes[1, col]
        if not replay.empty:
            sub = replay[replay["product"] == p]
            metrics = ["slippage_bps", "fee_cost_bps", "opportunity_cost_bps", "is_bps", "is_completed_bps"]
            w = 0.26
            x = np.arange(len(metrics))
            for i, s in enumerate(STRATS):
                ss = sub[sub["strategy"] == s]
                ax2.bar(x + (i - 1) * w, [ss[m].mean() for m in metrics], w * 0.92,
                        yerr=[_ci(ss[m]) for m in metrics], color=COLORS[s], label=s,
                        error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
            ax2.set_xticks(x, ["slippage\n(filled qty)", "fees", "opportunity", "Perold IS",
                               "completion-\nadjusted IS"], fontsize=8)
            ax2.axhline(0, color=INK2, lw=0.8)
            n = sub["window"].nunique() if "window" in sub else 0
            ax2.set_title(f"{p}: replay TCA, {n} windows x 2 sides")
            ax2.set_ylabel("bps")
            ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "replay_tape_and_tca.png")
    plt.close(fig)


def fig_book_profile(out: Path, events: Sequence[Event], grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for p, color in (("BTC-USD", "#2a78d6"), ("ETH-USD", "#eb6834")):
        books = [e for e in events if isinstance(e, BookSnapshot) and e.product == p and e.bids and e.asks]
        if not books:
            continue
        xs = np.linspace(0, 10, 41)
        prof = []
        for b in books[:: max(1, len(books) // 300)]:
            m = b.mid
            prof.append([sum(px * q for px, q in b.asks if px <= m * (1 + x / 1e4)) for x in xs])
        ax.plot(xs, np.median(prof, axis=0) / 1e3, color=color, lw=2, label=f"{p} live (median)")
    ax.set_xlabel("distance from mid (bps)")
    ax.set_ylabel("cumulative ask depth ($k)")
    ax.set_title("Captured cumulative ask depth (median snapshot)\nBTC flattens near 9 bps = top-100-level capture limit",
                 fontsize=9.5)
    ax.legend()
    fig.savefig(out / "book_depth_profile.png")
    plt.close(fig)


def fig_fill_model_bounds(out: Path, grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(REGIMES))
    w = 0.2
    specs = [("passive", "passive", ""), ("passive|prints-only", "passive", "//"),
             ("hybrid", "hybrid", ""), ("hybrid|prints-only", "hybrid", "//")]
    for ax, metric, title in ((axes[0], "fill_rate", "Fill rate by deadline"),
                              (axes[1], "is_completed_bps", "Completion-adjusted IS (bps)")):
        for i, (cfg, strat, hatch) in enumerate(specs):
            sub = grid[grid["config"] == cfg]
            m = [sub.loc[sub["regime"] == r, metric].mean() for r in REGIMES]
            e = [_ci(sub.loc[sub["regime"] == r, metric]) for r in REGIMES]
            ax.bar(x + (i - 1.5) * w, m, w * 0.92, yerr=e, color=COLORS[strat], hatch=hatch, edgecolor=SURFACE,
                   label=cfg.replace("|", " "), error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
        agg = grid[grid["config"] == "aggressive"]
        if metric == "is_completed_bps":
            ax.scatter(x, [agg.loc[agg["regime"] == r, metric].mean() for r in REGIMES], marker="_", s=400,
                       color=COLORS["aggressive"], lw=2.5, label="aggressive", zorder=5)
        ax.set_xticks(x, [r.replace("_", "\n") for r in REGIMES])
        ax.set_title(title)
    fig.suptitle("Passive fill-model bounds: prints+quotes (solid) vs prints-only (hatched)", fontweight="bold")
    axes[1].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(-0.1, -0.16), ncol=5)
    fig.savefig(out / "fill_model_bounds.png")
    plt.close(fig)


def make_all(out: Path, events: Sequence[Event], grid_all: pd.DataFrame, sweep: pd.DataFrame,
             lat: pd.DataFrame, replay: pd.DataFrame, s_fee: pd.DataFrame, cal: dict[str, Any]) -> None:
    grid = grid_all[grid_all["config"].isin(STRATS)]
    fig_fill_model_bounds(out, grid_all)
    fig_is_by_regime(out, grid)
    fig_cost_decomposition(out, grid)
    fig_fill_vs_adverse(out, grid)
    fig_markout_curves(out, grid)
    fig_size_sweep(out, sweep)
    fig_fee_tiers(out, s_fee)
    fig_latency_outage(out, lat)
    fig_replay(out, events, replay)
    fig_book_profile(out, events, grid)
