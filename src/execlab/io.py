"""Compact JSONL(.gz) serialization of normalized events for capture and deterministic replay."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from .types import BookSnapshot, Event, Side, Status, Trade

# Replay ordering on equal timestamps: connectivity first, then prints, then the book
# (the book snapshot already reflects the prints, so trades must be seen first).
_KIND_ORDER = {"S": 0, "T": 1, "B": 2}


def encode(ev: Event) -> dict:
    if isinstance(ev, BookSnapshot):
        return {
            "k": "B",
            "ts": round(ev.ts, 6),
            "p": ev.product,
            "b": [[px, q] for px, q in ev.bids],
            "a": [[px, q] for px, q in ev.asks],
        }
    if isinstance(ev, Trade):
        return {
            "k": "T",
            "ts": round(ev.ts, 6),
            "p": ev.product,
            "px": ev.price,
            "sz": ev.size,
            "ag": ev.aggressor.value,
            "id": ev.trade_id,
        }
    if isinstance(ev, Status):
        return {"k": "S", "ts": round(ev.ts, 6), "p": ev.product, "kind": ev.kind, "d": ev.detail}
    raise TypeError(type(ev))


def decode(d: dict) -> Event:
    k = d["k"]
    if k == "B":
        return BookSnapshot(
            ts=d["ts"],
            product=d["p"],
            bids=tuple((float(p), float(q)) for p, q in d["b"]),
            asks=tuple((float(p), float(q)) for p, q in d["a"]),
        )
    if k == "T":
        return Trade(d["ts"], d["p"], float(d["px"]), float(d["sz"]), Side(d["ag"]), d.get("id", ""))
    if k == "S":
        return Status(d["ts"], d["p"], d["kind"], d.get("d", ""))
    raise ValueError(k)


def _open(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8")


class EventWriter:
    """Writes events; book snapshots are delta-encoded against the previous snapshot of the
    same product ("D" records), with a full keyframe ("B") every `keyframe_s` seconds and
    after any connectivity event. Decoding reconstructs the exact original snapshots."""

    def __init__(self, path: str | Path, keyframe_s: float = 60.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = _open(self.path, "w")
        self.keyframe_s = keyframe_s
        self._last: dict[str, BookSnapshot] = {}
        self._last_key: dict[str, float] = {}
        self.count = 0

    def write(self, ev: Event) -> None:
        d = encode(ev)
        if isinstance(ev, BookSnapshot):
            prev = self._last.get(ev.product)
            if prev is not None and ev.ts - self._last_key[ev.product] < self.keyframe_s:
                d = {"k": "D", "ts": d["ts"], "p": ev.product,
                     "b": _diff(prev.bids, ev.bids), "a": _diff(prev.asks, ev.asks)}
            else:
                self._last_key[ev.product] = ev.ts
            self._last[ev.product] = ev
        elif isinstance(ev, Status):
            self._last.pop(ev.product, None)
        self._fh.write(json.dumps(d, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _diff(old: tuple[tuple[float, float], ...], new: tuple[tuple[float, float], ...]) -> list[list[float]]:
    o, n = dict(old), dict(new)
    out = [[p, q] for p, q in new if o.get(p) != q]
    out += [[p, 0] for p in o if p not in n]
    return out


def _apply(levels: dict[float, float], diff: list[list[float]]) -> None:
    for p, q in diff:
        if q == 0:
            levels.pop(float(p), None)
        else:
            levels[float(p)] = float(q)


def write_events(path: str | Path, events: Iterable[Event]) -> int:
    with EventWriter(path) as w:
        for ev in events:
            w.write(ev)
        return w.count


def sort_key(ev: Event) -> tuple[float, int]:
    k = "B" if isinstance(ev, BookSnapshot) else "T" if isinstance(ev, Trade) else "S"
    return (ev.ts, _KIND_ORDER[k])


def read_events(path: str | Path, product: str | None = None) -> list[Event]:
    """Load and deterministically order a capture (stable sort by ts, then kind)."""
    out: list[Event] = []
    books: dict[str, tuple[dict[float, float], dict[float, float]]] = {}
    with _open(Path(path), "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            k, pid = d["k"], d["p"]
            if k == "B":
                books[pid] = ({float(p): float(q) for p, q in d["b"]}, {float(p): float(q) for p, q in d["a"]})
            elif k == "D":
                if pid not in books:
                    continue  # delta without a keyframe (truncated file) -> skip
                _apply(books[pid][0], d["b"])
                _apply(books[pid][1], d["a"])
            if product is not None and pid != product:
                continue
            if k in ("B", "D"):
                bids, asks = books[pid]
                out.append(BookSnapshot(d["ts"], pid, tuple(sorted(bids.items(), reverse=True)),
                                        tuple(sorted(asks.items()))))
            else:
                out.append(decode(d))
    out.sort(key=sort_key)
    return out


def iter_events(path: str | Path) -> Iterator[Event]:
    yield from read_events(path)
