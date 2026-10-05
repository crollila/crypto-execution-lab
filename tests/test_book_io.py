from execlab.book import OrderBook
from execlab.io import read_events, write_events
from execlab.types import BookSnapshot, Side, Status, Trade


def test_order_book_updates():
    b = OrderBook("X")
    b.apply_snapshot([(99.0, 1.0), (98.0, 2.0)], [(101.0, 1.0)])
    b.apply_update(Side.BUY, 100.0, 0.5)
    b.apply_update(Side.BUY, 98.0, 0.0)
    b.apply_update(Side.SELL, 100.5, 3.0)
    top = b.top(1.0, 5)
    assert top.bids == ((100.0, 0.5), (99.0, 1.0))
    assert top.asks == ((100.5, 3.0), (101.0, 1.0))
    assert not b.is_crossed()
    b.apply_update(Side.BUY, 100.6, 1.0)
    assert b.is_crossed()


def test_delta_encoding_roundtrip_is_exact(tmp_path):
    evs = [
        BookSnapshot(0.0, "X", ((99.0, 1.0), (98.0, 2.0)), ((101.0, 1.0),)),
        Trade(0.5, "X", 101.0, 0.3, Side.BUY, "1"),
        BookSnapshot(1.0, "X", ((99.0, 1.5),), ((101.0, 0.7), (102.0, 4.0))),  # level removed + added
        Status(1.5, "X", "disconnect", "x"),
        BookSnapshot(2.0, "X", ((97.0, 1.0),), ((103.0, 1.0),)),
        BookSnapshot(70.0, "X", ((97.0, 2.0),), ((103.0, 1.0),)),  # keyframe after 60s
        BookSnapshot(1.0, "Y", ((5.0, 1.0),), ((6.0, 1.0),)),
    ]
    p = tmp_path / "c.jsonl.gz"
    write_events(p, evs)
    back = read_events(p)
    assert sorted(back, key=lambda e: (e.ts, e.product)) == sorted(evs, key=lambda e: (e.ts, e.product))
    assert [e.product for e in read_events(p, product="Y")] == ["Y"]


def test_replay_orders_trades_before_book_at_equal_ts(tmp_path):
    evs = [BookSnapshot(1.0, "X", ((1.0, 1.0),), ((2.0, 1.0),)), Trade(1.0, "X", 2.0, 1.0, Side.BUY)]
    p = tmp_path / "c.jsonl"
    write_events(p, evs)
    assert isinstance(read_events(p)[0], Trade)
