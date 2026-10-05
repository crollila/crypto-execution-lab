import numpy as np
import pytest

from execlab.ledger import Fill, Ledger
from execlab.types import Side


def f(side, px, q, fee=0.0):
    return Fill(0.0, 1, "X", side, px, q, "taker", fee)


def test_avg_cost_realized_and_flip():
    led = Ledger()
    led.on_fill(f(Side.BUY, 100, 1, 0.1))
    led.on_fill(f(Side.BUY, 110, 1, 0.1))
    p = led.position("X")
    assert p.qty == 2 and p.avg_cost == pytest.approx(105)
    led.on_fill(f(Side.SELL, 120, 3, 0.2))  # close 2 @ +15 each, open short 1 @ 120
    assert p.realized == pytest.approx(30) and p.qty == pytest.approx(-1) and p.avg_cost == 120
    led.on_fill(f(Side.BUY, 100, 1))
    assert p.realized == pytest.approx(50) and p.qty == 0
    assert p.fees == pytest.approx(0.4)


@pytest.mark.parametrize("seed", range(5))
def test_equity_equals_realized_plus_unrealized_minus_fees(seed):
    rng = np.random.default_rng(seed)
    led = Ledger()
    for _ in range(200):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        led.on_fill(f(side, 100 + rng.normal(), rng.uniform(0.01, 2), rng.uniform(0, 0.05)))
    mark = 101.3
    assert led.equity({"X": mark}) == pytest.approx(led.pnl({"X": mark})["net"], abs=1e-6)
