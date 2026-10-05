import pathlib
import re

import pytest

import execlab
from execlab import safety
from execlab.adapters.rest import CoinbaseRest, RequestsTransport
from execlab.adapters.ws import CoinbaseWs
from execlab.safety import LiveOrderGateway, LiveTradingDisabledError, assert_public_rest, assert_public_ws

PKG = pathlib.Path(execlab.__file__).parent


def test_live_trading_flag_is_off():
    assert safety.LIVE_TRADING_ENABLED is False


def test_live_gateway_cannot_be_constructed():
    with pytest.raises(LiveTradingDisabledError):
        LiveOrderGateway(api_key="x")


@pytest.mark.parametrize("method", ["POST", "DELETE", "PUT", "PATCH"])
def test_non_get_rejected(method):
    with pytest.raises(LiveTradingDisabledError):
        assert_public_rest("https://api.exchange.coinbase.com/products/BTC-USD/book", method)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.exchange.coinbase.com/orders",
        "https://api.exchange.coinbase.com/accounts",
        "https://api.exchange.coinbase.com/fills",
        "https://api.coinbase.com/api/v3/brokerage/orders",
        "https://evil.example.com/products/BTC-USD/book",
        "http://api.exchange.coinbase.com/products/BTC-USD/book",
    ],
)
def test_private_or_foreign_urls_rejected(url):
    with pytest.raises(LiveTradingDisabledError):
        assert_public_rest(url)


def test_public_urls_allowed():
    assert_public_rest("https://api.exchange.coinbase.com/products/BTC-USD/book")
    assert_public_rest("https://api-public.sandbox.exchange.coinbase.com/products/ETH-USD/trades")


def test_rest_client_refuses_private_paths_before_any_io():
    class Boom:
        def get(self, *a, **k):
            raise AssertionError("transport must not be reached")

    with pytest.raises(LiveTradingDisabledError):
        CoinbaseRest(transport=Boom())._get("/orders")


def test_ws_private_channels_rejected():
    with pytest.raises(LiveTradingDisabledError):
        assert_public_ws("wss://advanced-trade-ws.coinbase.com", ["user"])
    with pytest.raises(LiveTradingDisabledError):
        assert_public_ws("wss://evil.example.com", ["level2"])
    CoinbaseWs(["BTC-USD"], lambda e: None)  # default construction passes the guard


def test_transport_has_no_write_verbs():
    for verb in ("post", "put", "delete", "patch"):
        assert not hasattr(RequestsTransport, verb)


def test_no_signing_or_credential_code_in_package():
    forbidden = re.compile(r"CB-ACCESS|api_secret|private_key|hmac|\.post\(|\.delete\(|jwt", re.I)
    hits = []
    for f in PKG.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{f.name}:{i}: {line.strip()}")
    assert not hits, hits
