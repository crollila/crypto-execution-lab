import pytest

from execlab.adapters.rest import CoinbaseRest, RestError, TokenBucket, parse_ts
from execlab.types import Side


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(responses):
    t = FakeTransport(responses)
    return CoinbaseRest(transport=t, limiter=TokenBucket(1e9, 1000), sleep=lambda s: None), t


def test_product_spec():
    c, _ = client([(200, {"id": "BTC-USD", "quote_increment": "0.01", "base_increment": "0.00000001",
                          "min_market_funds": "1"})])
    s = c.product("BTC-USD")
    assert (s.tick, s.lot, s.min_notional) == (0.01, 1e-8, 1.0)


def test_book_normalization():
    c, t = client([(200, {"bids": [["100.00", "1.5", 3]], "asks": [["100.01", "2", 1]]})])
    b = c.book("BTC-USD")
    assert b.best_bid == 100.0 and b.best_ask == 100.01 and b.bids[0][1] == 1.5
    assert t.calls[0][1] == {"level": 2}


def test_trades_flip_maker_side_to_aggressor_and_sort():
    c, _ = client([(200, [
        {"trade_id": 2, "side": "buy", "size": "0.1", "price": "100", "time": "2026-10-05T00:00:01.5Z"},
        {"trade_id": 1, "side": "sell", "size": "0.2", "price": "101", "time": "2026-10-05T00:00:00.123456789Z"},
    ])])
    tr = c.trades("BTC-USD")
    assert [x.trade_id for x in tr] == ["1", "2"]
    assert tr[0].aggressor is Side.BUY  # maker sold -> taker bought
    assert tr[1].aggressor is Side.SELL


def test_retry_on_429_then_success():
    c, t = client([(429, "slow down"), RuntimeError("reset"), (200, {"bids": [], "asks": []})])
    c.book("BTC-USD")
    assert len(t.calls) == 3


def test_non_retryable_error_raises():
    c, _ = client([(404, {"message": "NotFound"})])
    with pytest.raises(RestError):
        c.book("NOPE-USD")


def test_token_bucket_throttles():
    now = [0.0]
    slept = []
    tb = TokenBucket(rate=2.0, burst=2, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    for _ in range(4):
        tb.acquire()
    assert sum(slept) == pytest.approx(1.0)


def test_parse_ts_nanoseconds():
    assert parse_ts("2026-10-05T00:00:00.123456789Z") == pytest.approx(1791158400.123456)
