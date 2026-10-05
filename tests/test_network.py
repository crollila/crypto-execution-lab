"""Live public-endpoint smoke tests. Run with: pytest -m network"""

import asyncio

import pytest

from execlab.adapters.rest import PRODUCTION, SANDBOX, CoinbaseRest
from execlab.adapters.ws import CoinbaseWs
from execlab.types import BookSnapshot

pytestmark = pytest.mark.network


@pytest.mark.parametrize("base", [PRODUCTION, SANDBOX])
def test_rest_book_and_trades(base):
    rest = CoinbaseRest(base)
    b = rest.book("BTC-USD")
    assert b.best_bid < b.best_ask
    assert rest.product("BTC-USD").tick == 0.01
    assert len(rest.trades("BTC-USD", limit=10)) > 0  # sandbox lists only a few products (no ETH-USD)


def test_ws_streams_books():
    ev = []
    asyncio.run(CoinbaseWs(["BTC-USD"], ev.append).run(duration=5))
    assert any(isinstance(e, BookSnapshot) for e in ev)
