import asyncio
import json

import pytest

from execlab.adapters.ws import Backoff, CoinbaseWs, Resync
from execlab.types import BookSnapshot, Side, Status, Trade

TS = "2026-10-05T00:00:00.000000Z"


def l2(seq, typ, updates, pid="BTC-USD"):
    return {
        "channel": "l2_data",
        "timestamp": TS,
        "sequence_num": seq,
        "events": [
            {
                "type": typ,
                "product_id": pid,
                "updates": [
                    {"side": s, "event_time": TS, "price_level": str(p), "new_quantity": str(q)}
                    for s, p, q in updates
                ],
            }
        ],
    }


def trades(seq, side="BUY", price="101"):
    return {
        "channel": "market_trades",
        "timestamp": TS,
        "sequence_num": seq,
        "events": [
            {
                "type": "update",
                "trades": [
                    {"product_id": "BTC-USD", "trade_id": "7", "price": price, "size": "0.5", "time": TS, "side": side}
                ],
            }
        ],
    }


def make(events):
    return CoinbaseWs(["BTC-USD"], events.append, snapshot_interval=0.0)


def test_parse_snapshot_update_and_trades():
    ev = []
    ws = make(ev)
    ws.handle(l2(0, "snapshot", [("bid", 100, 1), ("bid", 99, 2), ("offer", 101, 1)]))
    ws.handle(l2(1, "update", [("bid", 100, 0), ("offer", 100.5, 3)]))
    ws.handle(trades(2, side="BUY"))
    books = [e for e in ev if isinstance(e, BookSnapshot)]
    assert books[-1].bids == ((99.0, 2.0),)
    assert books[-1].asks == ((100.5, 3.0), (101.0, 1.0))
    tr = [e for e in ev if isinstance(e, Trade)]
    assert tr[0].aggressor is Side.SELL  # Coinbase reports the maker side


def test_sequence_gap_raises_resync_and_emits_status():
    ev = []
    ws = make(ev)
    ws.handle(l2(0, "snapshot", [("bid", 100, 1), ("offer", 101, 1)]))
    with pytest.raises(Resync):
        ws.handle(l2(5, "update", [("bid", 100, 2)]))
    assert any(isinstance(e, Status) and e.kind == "gap" for e in ev)


def test_updates_before_snapshot_are_ignored():
    ev = []
    ws = make(ev)
    ws.handle(l2(0, "update", [("bid", 100, 1), ("offer", 101, 1)]))
    assert not ev


class FakeSocket:
    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def recv(self):
        if not self.script:
            await asyncio.sleep(10)  # silent feed -> stale timeout
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)


def test_reconnect_resync_and_backoff():
    sockets = [
        FakeSocket([l2(0, "snapshot", [("bid", 100, 1), ("offer", 101, 1)]), ConnectionError("reset")]),
        FakeSocket([l2(0, "snapshot", [("bid", 200, 1), ("offer", 201, 1)]), trades(1), l2(3, "update", [])]),
        FakeSocket([]),  # stale feed
    ]
    made = []

    def connect(url):
        made.append(url)
        if len(made) <= len(sockets):
            return sockets[len(made) - 1]
        return FakeSocket([ConnectionError("down")])

    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    ev = []
    ws = CoinbaseWs(
        ["BTC-USD"], ev.append, snapshot_interval=0.0, connect=connect, sleep=fake_sleep,
        stale_timeout=0.05, backoff=Backoff(1.0, 8.0, seed=1), max_reconnects=3,
    )
    with pytest.raises(ConnectionError):
        asyncio.run(ws.run())
    kinds = [e.kind for e in ev if isinstance(e, Status)]
    assert kinds.count("disconnect") == 4
    assert kinds.count("reconnect") == 3
    assert "gap" in kinds
    # book discarded and rebuilt from the fresh snapshot after each reconnect
    books = [e for e in ev if isinstance(e, BookSnapshot)]
    assert books[0].best_bid == 100 and books[1].best_bid == 200
    assert {m["channel"] for m in sockets[0].sent} == {"level2", "market_trades", "heartbeats"}
    assert len(sleeps) == 3 and all(0 < s <= 8.0 for s in sleeps)


def test_backoff_caps_and_jitters():
    b = Backoff(0.5, 4.0, seed=0)
    d = [b.next() for _ in range(10)]
    assert max(d) <= 4.0 and d[0] <= 0.5 and d[-1] >= 2.0
