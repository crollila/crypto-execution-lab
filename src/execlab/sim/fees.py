"""Maker/taker fee schedules.

The tier table mirrors the *shape* of Coinbase Advanced Trade's published volume tiers
(30-day USD volume -> maker/taker bps). Fees change; treat these as illustrative
defaults and pass your own `FeeSchedule` for anything that matters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    name: str
    maker_bps: float
    taker_bps: float

    def fee(self, notional: float, liquidity: str) -> float:
        bps = self.maker_bps if liquidity == "maker" else self.taker_bps
        return notional * bps / 1e4


# (min 30d volume USD, maker bps, taker bps)
COINBASE_ADVANCED_TIERS: tuple[tuple[float, float, float], ...] = (
    (0, 40.0, 60.0),
    (10_000, 25.0, 40.0),
    (50_000, 15.0, 25.0),
    (100_000, 10.0, 20.0),
    (1_000_000, 8.0, 18.0),
    (15_000_000, 6.0, 16.0),
    (75_000_000, 3.0, 12.0),
    (250_000_000, 0.0, 8.0),
    (400_000_000, 0.0, 5.0),
)


def coinbase_tier(volume_30d_usd: float) -> FeeSchedule:
    chosen = COINBASE_ADVANCED_TIERS[0]
    for row in COINBASE_ADVANCED_TIERS:
        if volume_30d_usd >= row[0]:
            chosen = row
    return FeeSchedule(f"coinbase>={chosen[0]:,.0f}", chosen[1], chosen[2])


# Named schedules used by the experiments.
RETAIL = coinbase_tier(0)
MID_TIER = coinbase_tier(1_000_000)  # 8 / 18 bps — experiment default
TOP_TIER = coinbase_tier(400_000_000)  # 0 / 5 bps
MAKER_REBATE = FeeSchedule("rebate(-1/+5)", -1.0, 5.0)  # stylized MM-program venue
ZERO = FeeSchedule("zero", 0.0, 0.0)
