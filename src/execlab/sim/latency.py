"""Order-entry latency and connectivity-outage models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..types import Status


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """One-way order-entry latency: lognormal body + occasional spikes (GC, queueing, WAN)."""

    median_ms: float = 40.0
    sigma: float = 0.35
    spike_prob: float = 0.01
    spike_ms: float = 400.0

    def sample(self, rng: np.random.Generator) -> float:
        ms = self.median_ms * math.exp(self.sigma * rng.standard_normal())
        if self.spike_prob > 0 and rng.random() < self.spike_prob:
            ms += rng.exponential(self.spike_ms)
        return ms / 1e3


ZERO_LATENCY = LatencyModel(median_ms=0.0, sigma=0.0, spike_prob=0.0)


@dataclass(frozen=True, slots=True)
class OutageModel:
    """Poisson disconnects with exponential durations, emitted as Status events."""

    rate_per_min: float = 0.0
    mean_duration_s: float = 3.0

    def generate(
        self, product: str, t0: float, t1: float, rng: np.random.Generator
    ) -> list[Status]:
        out: list[Status] = []
        if self.rate_per_min <= 0:
            return out
        t = t0
        lam = self.rate_per_min / 60.0
        while True:
            t += rng.exponential(1.0 / lam)
            if t >= t1:
                break
            dur = rng.exponential(self.mean_duration_s)
            out.append(Status(t, product, "disconnect", "simulated"))
            out.append(Status(t + dur, product, "reconnect", "simulated"))
            t += dur
        return out
