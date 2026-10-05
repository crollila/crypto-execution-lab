"""Transaction-cost analysis for a simulated parent order.

Sign convention: every *cost* metric is positive when it hurts the trader.

* arrival mid  m0 = mid of the first snapshot at/after the decision time
* slippage_bps = s * (VWAP - m0) / m0 * 1e4, on filled qty, excl. fees
* fee_bps      = fees / filled notional * 1e4
* implementation shortfall (Perold), in bps of the *target* notional Q*m0:
      IS = [ s*sum(q_i*(p_i - m0)) + fees + s*(Q - q)*(m_T - m0) ] / (Q*m0) * 1e4
  where m_T is the mid at the deadline (opportunity cost of the unfilled part).
* completion-adjusted IS (`is_completed_bps`): IS plus the cost of finishing the unfilled
  remainder with a marketable order against the *deadline* book (walk the visible depth,
  pay the taker fee). Perold IS alone charges unfilled size only for mid drift, which is
  ~zero on average, and so flatters strategies that simply do not trade.
* adverse_sel_bps@h = -s * (m(t_i+h) - p_i) / p_i * 1e4, qty-weighted over fills: the
  negative markout of each fill against the mid h seconds later (positive = the fill was
  followed by the market moving through it). For maker fills this nets the spread captured
  against the subsequent move — the classic passive-fill toxicity measure.
* post_fill_drift_bps@h = -s * (m(t_i+h) - m(t_i)) / m(t_i) * 1e4: mid drift against the
  position after the fill, excluding the spread component.
"""

from __future__ import annotations

import math
from typing import Any

from .runner import ExecResult

HORIZONS = (1.0, 5.0, 30.0)


def _wavg(vals: list[float], w: list[float]) -> float:
    tw = sum(w)
    return sum(v * x for v, x in zip(vals, w)) / tw if tw > 0 else math.nan


def analyze(r: ExecResult) -> dict[str, Any]:
    p = r.parent
    s = p.side.sign
    a = r.arrival
    m0 = a.mid
    q = sum(f.qty for f in r.fills)
    notional = sum(f.price * f.qty for f in r.fills)
    fees = sum(f.fee for f in r.fills)
    maker_q = sum(f.qty for f in r.fills if f.liquidity == "maker")
    vwap = notional / q if q > 0 else math.nan
    m_T = r.mid_at(p.deadline)
    target_notional = p.qty * m0
    exec_cost = s * (notional - q * m0)
    opp_cost = s * (p.qty - q) * (m_T - m0)
    bid_depth, ask_depth = a.depth_notional(10.0)
    unfilled = max(0.0, p.qty - q)
    cleanup_exec, cleanup_notional = _cleanup(r, unfilled, m_T)
    out: dict[str, Any] = {
        "strategy": r.strategy,
        "product": p.product,
        "side": p.side.value,
        "target_qty": p.qty,
        "target_notional": target_notional,
        "arrival_mid": m0,
        "arrival_spread_bps": a.spread_bps,
        "arrival_depth10_usd": ask_depth if s > 0 else bid_depth,  # liquidity we would consume
        "filled_qty": q,
        "fill_rate": q / p.qty if p.qty > 0 else math.nan,
        "maker_share": maker_q / q if q > 0 else math.nan,
        "vwap": vwap,
        "slippage_bps": s * (vwap - m0) / m0 * 1e4 if q > 0 else math.nan,
        "fee_bps": fees / notional * 1e4 if notional > 0 else math.nan,
        "fees_usd": fees,
        "maker_notional": sum(f.price * f.qty for f in r.fills if f.liquidity == "maker"),
        "taker_notional": sum(f.price * f.qty for f in r.fills if f.liquidity == "taker"),
        "exec_cost_bps": exec_cost / target_notional * 1e4,
        "fee_cost_bps": fees / target_notional * 1e4,
        "opportunity_cost_bps": opp_cost / target_notional * 1e4,
        "is_bps": (exec_cost + fees + opp_cost) / target_notional * 1e4,
        "cleanup_exec_bps": cleanup_exec / target_notional * 1e4,
        "cleanup_notional": cleanup_notional,
        "is_completed_bps": (exec_cost + fees + opp_cost + cleanup_exec + cleanup_notional * r.taker_bps / 1e4)
        / target_notional * 1e4,
        "drift_bps": s * (m_T - m0) / m0 * 1e4,  # what the market did over the horizon
        "n_children": len(r.children),
        "n_fills": len(r.fills),
        "n_rejects": sum(1 for o in r.children if o.status.value == "rejected"),
        "n_cancel_requests": r.stats.get("cancel_requests", 0),
        "time_to_last_fill_s": (max(f.ts for f in r.fills) - p.start_ts) if r.fills else math.nan,
        "blind_s": r.blind_s,
        "maker_fills_by_queue": r.stats.get("fills_by_queue", 0),
        "maker_fills_by_tradethrough": r.stats.get("fills_by_tradethrough", 0),
        "maker_fills_by_cross": r.stats.get("fills_by_cross", 0),
        "fills_after_deadline_qty": sum(f.qty for f in r.fills if f.ts > p.deadline),
    }
    for h in HORIZONS:
        tag = f"{h:g}s"
        for subset, fills in (("all", r.fills), ("maker", [f for f in r.fills if f.liquidity == "maker"])):
            if not fills:
                out[f"adverse_sel_bps_{tag}_{subset}"] = math.nan
                out[f"post_fill_drift_bps_{tag}_{subset}"] = math.nan
                continue
            w = [f.qty for f in fills]
            adv = []
            drift = []
            for f in fills:
                mt, mh = r.mid_at(f.ts), r.mid_at(f.ts + h)
                adv.append(-s * (mh - f.price) / f.price * 1e4)
                drift.append(-s * (mh - mt) / mt * 1e4)
            out[f"adverse_sel_bps_{tag}_{subset}"] = _wavg(adv, w)
            out[f"post_fill_drift_bps_{tag}_{subset}"] = _wavg(drift, w)
    return out


def _cleanup(r: ExecResult, unfilled: float, m_T: float) -> tuple[float, float]:
    """Cost vs. the *arrival*-relative opportunity mark of crossing `unfilled` at the deadline:
    returns (s * sum(q_j * (p_j - m_T)), notional). Depth beyond the visible book is
    charged at the last visible price."""
    if unfilled <= 1e-12 or r.deadline_book is None:
        return 0.0, 0.0
    s = r.parent.side.sign
    levels = r.deadline_book.asks if s > 0 else r.deadline_book.bids
    rem, cost, notional, px = unfilled, 0.0, 0.0, m_T
    for px, lq in levels:
        take = min(rem, lq)
        cost += s * (px - m_T) * take
        notional += px * take
        rem -= take
        if rem <= 1e-12:
            break
    if rem > 1e-12:
        cost += s * (px - m_T) * rem
        notional += px * rem
    return cost, notional
