import math

import numpy as np
import pytest

from execlab.experiments import PathTask, RunConfig, run_path_task, run_tasks
from execlab.io import read_events, write_events
from execlab.ledger import Fill
from execlab.runner import ExecResult, run_parent
from execlab.sim.latency import OutageModel
from execlab.strategies import STRATEGIES, ParentOrder
from execlab.synth import RegimeParams, generate, regime_grid
from execlab.tca import analyze
from execlab.types import BookSnapshot, ProductSpec, Side

SPEC = ProductSpec("BTC-USD", 0.01, 1e-8, 1.0)
BASE = RegimeParams("base", n_levels=60, level_spacing_bps=0.25, level_usd=20_000)
GRID = regime_grid(BASE)


@pytest.fixture(scope="module")
def paths():
    return {r: generate(SPEC, rp, 60_000.0, 80, seed=11) for r, rp in GRID.items()}


# ----------------------------------------------------------------------------- synth
def test_generator_is_deterministic():
    a = generate(SPEC, GRID["calm_thin"], 60_000.0, 10, seed=3)
    b = generate(SPEC, GRID["calm_thin"], 60_000.0, 10, seed=3)
    c = generate(SPEC, GRID["calm_thin"], 60_000.0, 10, seed=4)
    assert a == b and a != c


def test_generated_books_are_valid_and_regimes_ordered(paths):
    stats = {}
    for r, ev in paths.items():
        books = [e for e in ev if isinstance(e, BookSnapshot)]
        assert all(b.best_bid < b.best_ask for b in books)
        assert all(all(x[0] > y[0] for x, y in zip(b.bids, b.bids[1:])) for b in books[::50])
        mids = np.array([b.mid for b in books])
        stats[r] = (np.mean([b.spread_bps for b in books]), np.median([sum(b.depth_notional(10)) for b in books]),
                    np.diff(np.log(mids[::10])).std())
    assert stats["calm_thin"][0] > stats["calm_deep"][0]  # wider spread
    assert stats["calm_thin"][1] < stats["calm_deep"][1]  # less depth
    assert stats["volatile_deep"][2] > 2 * stats["calm_deep"][2]  # more vol


# ------------------------------------------------------------------------ strategies
@pytest.mark.parametrize("regime", list(GRID))
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_strategy_invariants(paths, regime, side):
    ev = paths[regime]
    ts = [e.ts for e in ev]
    for name, S in STRATEGIES.items():
        po = ParentOrder("BTC-USD", side, 2.0, 5.0, 40.0)
        r = run_parent(ev, SPEC, po, S, seed=1, event_ts=ts)
        m = analyze(r)
        assert m["filled_qty"] <= po.qty + 1e-9, "never overfill"
        assert all(f.side is side for f in r.fills)
        if name == "aggressive":
            assert m["maker_share"] == 0.0
        if name == "passive":
            assert all(f.liquidity == "maker" for f in r.fills), "post-only never takes"
            assert all(o.post_only for o in r.children)


def test_hybrid_completes_more_often_than_passive(paths):
    ev = paths["calm_thin"]
    fills = {}
    for name in ("passive", "hybrid"):
        po = ParentOrder("BTC-USD", Side.BUY, 8.0, 5.0, 40.0)  # large vs thin book
        fills[name] = analyze(run_parent(ev, SPEC, po, STRATEGIES[name], seed=1))["fill_rate"]
    assert fills["hybrid"] >= fills["passive"]
    assert fills["hybrid"] > 0.99


def test_replay_from_file_matches_in_memory(tmp_path, paths):
    ev = paths["calm_deep"]
    p = tmp_path / "x.jsonl.gz"
    write_events(p, ev)
    ev2 = read_events(p)
    po = ParentOrder("BTC-USD", Side.SELL, 1.0, 5.0, 30.0)
    a = analyze(run_parent(ev, SPEC, po, STRATEGIES["passive"], seed=2))
    b = analyze(run_parent(ev2, SPEC, po, STRATEGIES["passive"], seed=2))
    assert a == b


# -------------------------------------------------------------------------------- tca
def test_tca_hand_computed():
    a = BookSnapshot(0.0, "X", ((99.0, 10.0),), ((101.0, 10.0),))  # mid 100
    po = ParentOrder("X", Side.BUY, 2.0, 0.0, 10.0)
    fills = [Fill(1.0, 1, "X", Side.BUY, 101.0, 1.0, "taker", 0.2)]
    mid_ts = np.array([0.0, 2.0, 6.0, 10.0])
    mid_px = np.array([100.0, 102.0, 103.0, 104.0])
    r = ExecResult("t", po, a, fills, [], mid_ts, mid_px)
    m = analyze(r)
    assert m["fill_rate"] == 0.5
    assert m["slippage_bps"] == pytest.approx(100.0)  # 101 vs 100
    assert m["fee_bps"] == pytest.approx(0.2 / 101 * 1e4)
    # IS = exec 1.0 + fee 0.2 + opportunity 1 * (104 - 100) = 5.2 on 200 notional
    assert m["is_bps"] == pytest.approx(5.2 / 200 * 1e4)
    assert m["opportunity_cost_bps"] == pytest.approx(4 / 200 * 1e4)
    # fill at t=1 @101; mid 1s later = 102 -> markout +99 bps, i.e. adverse selection -99 bps
    assert m["adverse_sel_bps_1s_all"] == pytest.approx(-(102 - 101) / 101 * 1e4)
    assert m["post_fill_drift_bps_1s_all"] == pytest.approx(-200.0)  # mid 100 -> 102 moved *with* the buy
    assert m["adverse_sel_bps_5s_all"] == pytest.approx(-(103 - 101) / 101 * 1e4)
    assert math.isnan(m["adverse_sel_bps_5s_maker"])
    assert m["is_completed_bps"] == pytest.approx(m["is_bps"])  # no deadline book -> no cleanup


def test_completion_adjusted_is_charges_crossing_the_remainder():
    a = BookSnapshot(0.0, "X", ((99.0, 10.0),), ((101.0, 10.0),))
    po = ParentOrder("X", Side.BUY, 2.0, 0.0, 10.0)
    fills = [Fill(1.0, 1, "X", Side.BUY, 101.0, 1.0, "taker", 0.2)]
    dl = BookSnapshot(10.0, "X", ((103.0, 5.0),), ((105.0, 0.5), (106.0, 10.0)))  # mid 104
    r = ExecResult("t", po, a, fills, [], np.array([0.0, 10.0]), np.array([100.0, 104.0]),
                   deadline_book=dl, taker_bps=20.0)
    m = analyze(r)
    cleanup = (105 - 104) * 0.5 + (106 - 104) * 0.5
    fee = (105 * 0.5 + 106 * 0.5) * 20 / 1e4
    assert m["cleanup_exec_bps"] == pytest.approx(cleanup / 200 * 1e4)
    assert m["is_completed_bps"] == pytest.approx((1.0 + 0.2 + 4.0 + cleanup + fee) / 200 * 1e4)


# ------------------------------------------------------------------------ experiments
def _task(seed, configs):
    return PathTask("t", SPEC, GRID["volatile_deep"], 60_000.0, seed, 50_000.0, 20.0, configs,
                    warmup_s=3.0, markout_s=5.0)


def test_experiment_runs_are_bitwise_reproducible():
    cfgs = tuple(RunConfig(s) for s in STRATEGIES)
    a = run_tasks([_task(1, cfgs), _task(2, cfgs)], workers=1)
    b = run_tasks([_task(1, cfgs), _task(2, cfgs)], workers=1)
    assert a.equals(b) and len(a) == 2 * 3 * 2


def test_outages_blind_strategy_and_cancel_on_disconnect():
    out = OutageModel(rate_per_min=30.0, mean_duration_s=2.0)
    cfgs = (RunConfig("passive", "cod", outage=out, cancel_on_disconnect=True),
            RunConfig("passive", "nocod", outage=out, cancel_on_disconnect=False))
    rows = run_path_task(_task(5, cfgs))
    assert all(r["blind_s"] > 0 for r in rows)
