from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float | None = None
    slippage_bps: float | None = None
    spread_bps: float | None = None
    tax_bps: float | None = None

    def grid(self) -> tuple[CostConfig, ...]:
        commission_grid: tuple[float, ...] = (0.0, 1.5, 3.0, 15.0)
        slippage_grid: tuple[float, ...] = (0.0, 5.0, 20.0)
        spread_grid: tuple[float, ...] = (0.0,)
        comm_vals = commission_grid if self.commission_bps is None else (float(self.commission_bps),)
        slip_vals = slippage_grid if self.slippage_bps is None else (float(self.slippage_bps),)
        spread_vals = spread_grid if self.spread_bps is None else (float(self.spread_bps),)
        combos = list(itertools.product(comm_vals, slip_vals, spread_vals))
        return tuple(CostConfig(commission_bps=c, slippage_bps=s, spread_bps=p) for c, s, p in combos)


class CostModel:
    def __init__(self, config: CostConfig) -> None:
        self.config = config

    def charge(self, traded_notional: float) -> float:
        comm = float(self.config.commission_bps or 0.0)
        slip = float(self.config.slippage_bps or 0.0)
        spread = float(self.config.spread_bps or 0.0)
        tax = float(self.config.tax_bps or 0.0)
        bps = comm + slip + spread + tax
        return float(traded_notional) * bps / 10000.0
