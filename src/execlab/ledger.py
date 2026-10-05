"""Average-cost position and P&L ledger (quote-currency cash accounting)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Side


@dataclass(slots=True)
class Fill:
    ts: float
    order_id: int
    product: str
    side: Side
    price: float
    qty: float
    liquidity: str  # "maker" | "taker"
    fee: float

    @property
    def notional(self) -> float:
        return self.price * self.qty


@dataclass(slots=True)
class Position:
    qty: float = 0.0
    avg_cost: float = 0.0
    realized: float = 0.0
    fees: float = 0.0
    volume: float = 0.0  # traded notional

    def unrealized(self, mark: float) -> float:
        return (mark - self.avg_cost) * self.qty


@dataclass
class Ledger:
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def position(self, product: str) -> Position:
        return self.positions.setdefault(product, Position())

    def on_fill(self, f: Fill) -> None:
        pos = self.position(f.product)
        signed = f.side.sign * f.qty
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
            new_qty = pos.qty + signed
            pos.avg_cost = (pos.avg_cost * abs(pos.qty) + f.price * f.qty) / abs(new_qty)
            pos.qty = new_qty
        else:
            closing = min(abs(signed), abs(pos.qty))
            pos.realized += (f.price - pos.avg_cost) * closing * (1 if pos.qty > 0 else -1)
            new_qty = pos.qty + signed
            if abs(new_qty) < 1e-12:
                pos.qty, pos.avg_cost = 0.0, 0.0
            elif (new_qty > 0) != (pos.qty > 0):  # flipped through zero
                pos.qty, pos.avg_cost = new_qty, f.price
            else:
                pos.qty = new_qty
        pos.fees += f.fee
        pos.volume += f.notional
        self.cash -= signed * f.price + f.fee
        self.fills.append(f)

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + sum(p.qty * marks[k] for k, p in self.positions.items() if p.qty)

    def pnl(self, marks: dict[str, float]) -> dict[str, float]:
        realized = sum(p.realized for p in self.positions.values())
        unreal = sum(p.unrealized(marks[k]) for k, p in self.positions.items() if p.qty)
        fees = sum(p.fees for p in self.positions.values())
        return {"realized": realized, "unrealized": unreal, "fees": fees, "net": realized + unreal - fees}
