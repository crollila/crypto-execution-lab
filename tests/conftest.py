import pytest

from execlab.sim.exchange import SimExchange
from execlab.sim.fees import FeeSchedule
from execlab.sim.latency import ZERO_LATENCY
from execlab.types import BookSnapshot, ProductSpec

SPEC = ProductSpec("BTC-USD", 0.01, 1e-8, 1.0)
FEES = FeeSchedule("test", maker_bps=10.0, taker_bps=20.0)


def book(ts=0.0, bids=((99.0, 1.0), (98.0, 2.0)), asks=((101.0, 1.0), (102.0, 2.0))):
    return BookSnapshot(ts, "BTC-USD", tuple(bids), tuple(asks))


@pytest.fixture
def ex():
    e = SimExchange({"BTC-USD": SPEC}, fees=FEES, latency=ZERO_LATENCY, seed=0)
    e.on_event(book())
    return e
