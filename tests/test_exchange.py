import math

import pytest
from conftest import FEES, SPEC, book

from execlab.sim.exchange import TIF, OrderStatus, OrderType, SimExchange
from execlab.sim.latency import ZERO_LATENCY, LatencyModel
from execlab.types import Side, Status, Trade

BUY, SELL = Side.BUY, Side.SELL
FIXED_50MS = LatencyModel(median_ms=50, sigma=0, spike_prob=0)


def test_market_order_walks_book_with_taker_fees(ex):
    o = ex.submit("BTC-USD", BUY, 1.5, OrderType.MARKET)
    ex.advance(0.0)
    assert o.status is OrderStatus.FILLED
    assert [(f.price, f.qty) for f in o.fills] == [(101.0, 1.0), (102.0, 0.5)]
    assert o.avg_price == pytest.approx((101 + 51) / 1.5)
    assert o.fees == pytest.approx((101 + 51) * 20 / 1e4)
    assert all(f.liquidity == "taker" for f in o.fills)


def test_market_order_depth_exhaustion_is_partial(ex):
    o = ex.submit("BTC-USD", BUY, 5.0, OrderType.MARKET)
    ex.advance(0.0)
    assert o.status is OrderStatus.CANCELLED and o.filled == pytest.approx(3.0)
    assert ex.stats["market_depth_exhausted"] == 1


def test_shadow_depletion_until_next_snapshot(ex):
    a = ex.submit("BTC-USD", BUY, 1.0, OrderType.MARKET)
    b = ex.submit("BTC-USD", BUY, 1.0, OrderType.MARKET)
    ex.advance(0.0)
    assert a.avg_price == 101.0 and b.avg_price == 102.0  # 101 level already consumed by us
    ex.on_event(book(1.0))
    c = ex.submit("BTC-USD", BUY, 1.0, OrderType.MARKET)
    ex.advance(1.0)
    assert c.avg_price == 101.0  # replenished by the replayed snapshot


def test_ioc_partial_then_cancel(ex):
    o = ex.submit("BTC-USD", BUY, 2.0, limit_price=101.0, tif=TIF.IOC)
    ex.advance(0.0)
    assert o.status is OrderStatus.CANCELLED and o.filled == pytest.approx(1.0)
    assert not ex.open_orders()


def test_fok(ex):
    k = ex.submit("BTC-USD", BUY, 2.0, limit_price=101.0, tif=TIF.FOK)
    f = ex.submit("BTC-USD", BUY, 2.0, limit_price=102.0, tif=TIF.FOK)
    ex.advance(0.0)
    assert k.status is OrderStatus.CANCELLED and k.filled == 0
    assert f.status is OrderStatus.FILLED and f.filled == pytest.approx(2.0)


def test_gtc_marketable_remainder_rests(ex):
    o = ex.submit("BTC-USD", BUY, 1.5, limit_price=101.0)
    ex.advance(0.0)
    assert o.filled == pytest.approx(1.0) and o.status is OrderStatus.OPEN and o.queue_ahead == 0.0


def test_post_only_rejected_when_marketable_and_rests_otherwise(ex):
    r = ex.submit("BTC-USD", BUY, 1.0, limit_price=101.0, post_only=True)
    ok = ex.submit("BTC-USD", BUY, 1.0, limit_price=99.0, post_only=True)
    ex.advance(0.0)
    assert r.status is OrderStatus.REJECTED and "post_only" in r.reject_reason
    assert ok.status is OrderStatus.OPEN and ok.queue_ahead == pytest.approx(1.0)


def test_post_only_requires_gtc(ex):
    o = ex.submit("BTC-USD", BUY, 1.0, limit_price=99.0, tif=TIF.IOC, post_only=True)
    assert o.status is OrderStatus.REJECTED


def test_queue_position_then_partial_maker_fills(ex):
    o = ex.submit("BTC-USD", BUY, 1.0, limit_price=99.0)
    ex.advance(0.0)
    ex.on_event(Trade(0.1, "BTC-USD", 99.0, 0.6, SELL))  # eats 0.6 of the 1.0 ahead of us
    assert o.filled == 0 and o.queue_ahead == pytest.approx(0.4)
    ex.on_event(Trade(0.2, "BTC-USD", 99.0, 0.7, SELL))  # 0.4 ahead, 0.3 reaches us
    assert o.filled == pytest.approx(0.3) and o.fills[0].liquidity == "maker"
    assert o.fees == pytest.approx(99 * 0.3 * 10 / 1e4)
    ex.on_event(Trade(0.3, "BTC-USD", 99.0, 5.0, BUY))  # buyer-initiated: cannot hit our bid
    assert o.filled == pytest.approx(0.3)
    ex.on_event(Trade(0.4, "BTC-USD", 98.5, 0.2, SELL))  # prints through our price
    assert o.filled == pytest.approx(0.5)


def test_level_shrink_is_cancellation_ahead(ex):
    o = ex.submit("BTC-USD", BUY, 1.0, limit_price=98.0)
    ex.advance(0.0)
    assert o.queue_ahead == pytest.approx(2.0)
    ex.on_event(book(1.0, bids=((99.0, 1.0), (98.0, 0.5))))
    assert o.queue_ahead == pytest.approx(0.5)
    ex.on_event(book(2.0, bids=((99.0, 1.0), (98.0, 3.0))))  # growth joins *behind* us
    assert o.queue_ahead == pytest.approx(0.5)


def test_crossing_quotes_fill_resting_order_at_our_price(ex):
    o = ex.submit("BTC-USD", SELL, 1.0, limit_price=101.0)
    ex.advance(0.0)
    ex.on_event(book(1.0, bids=((101.5, 0.4), (99.0, 1.0)), asks=((102.0, 1.0),)))
    assert o.filled == pytest.approx(0.4) and o.fills[0].price == 101.0


def test_reduce_only(ex):
    r = ex.submit("BTC-USD", SELL, 1.0, OrderType.MARKET, reduce_only=True)
    ex.advance(0.0)
    assert r.status is OrderStatus.REJECTED  # flat: any order would open a position
    ex.submit("BTC-USD", BUY, 0.5, OrderType.MARKET)
    ex.advance(0.0)
    up = ex.submit("BTC-USD", BUY, 0.1, OrderType.MARKET, reduce_only=True)
    clip = ex.submit("BTC-USD", SELL, 2.0, OrderType.MARKET, reduce_only=True)
    ex.advance(0.0)
    assert up.status is OrderStatus.REJECTED
    assert clip.qty == pytest.approx(0.5) and clip.filled == pytest.approx(0.5)
    assert ex.ledger.position("BTC-USD").qty == pytest.approx(0.0)


def test_validation():
    e = SimExchange({"BTC-USD": SPEC}, fees=FEES, latency=ZERO_LATENCY)
    e.on_event(book())
    assert e.submit("BTC-USD", BUY, 1.0, limit_price=99.005).reject_reason == "price not on tick"
    assert e.submit("BTC-USD", BUY, 0.001, limit_price=99.0).reject_reason == "below min notional"
    assert e.submit("BTC-USD", BUY, 1.0, OrderType.MARKET, limit_price=99.0).status is OrderStatus.REJECTED


def test_latency_order_sees_book_at_arrival():
    e = SimExchange({"BTC-USD": SPEC}, fees=FEES, latency=FIXED_50MS)
    e.on_event(book(0.0))
    o = e.submit("BTC-USD", BUY, 1.0, OrderType.MARKET)
    e.on_event(book(0.01, asks=((105.0, 1.0),)))  # market moves before our order lands
    e.on_event(book(0.2))
    assert o.arrive_ts == pytest.approx(0.05) and o.avg_price == 105.0


def test_fill_while_cancel_in_flight():
    e = SimExchange({"BTC-USD": SPEC}, fees=FEES, latency=FIXED_50MS)
    e.on_event(book(0.0))
    o = e.submit("BTC-USD", BUY, 1.0, limit_price=99.0)
    e.on_event(book(0.1, bids=((99.0, 0.0001), (98.0, 2.0))))
    assert o.status is OrderStatus.OPEN
    e.cancel(o.id)  # lands at 0.15
    e.on_event(Trade(0.12, "BTC-USD", 98.0, 3.0, SELL))  # trade-through before the cancel arrives
    e.on_event(book(0.2))
    assert o.status is OrderStatus.FILLED and e.stats["cancel_too_late"] == 1


@pytest.mark.parametrize("cod", [True, False])
def test_disconnect_semantics(cod):
    e = SimExchange({"BTC-USD": SPEC}, fees=FEES, latency=ZERO_LATENCY, cancel_on_disconnect=cod)
    e.on_event(book())
    o = e.submit("BTC-USD", BUY, 1.0, limit_price=99.0)
    e.advance(0.0)
    e.on_event(Status(1.0, "BTC-USD", "disconnect"))
    assert e.submit("BTC-USD", BUY, 1.0, OrderType.MARKET).reject_reason == "disconnected"
    e.on_event(Trade(1.5, "BTC-USD", 98.0, 2.0, SELL))
    if cod:
        assert o.status is OrderStatus.CANCELLED and o.filled == 0
    else:
        assert o.status is OrderStatus.FILLED  # stale order picked off while we were blind
    e.on_event(Status(2.0, "BTC-USD", "reconnect"))
    assert e.connected["BTC-USD"]


def test_no_market_rejects():
    e = SimExchange({"BTC-USD": SPEC}, fees=FEES, latency=ZERO_LATENCY)
    o = e.submit("BTC-USD", BUY, 1.0, limit_price=99.0)
    e.advance(0.0)
    assert o.status is OrderStatus.REJECTED and math.isnan(o.avg_price)
