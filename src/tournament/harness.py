from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import yaml

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.tournament.distribution import ReturnDistribution
from src.tournament.simulator import TournamentSimulator


def load_participation_grid() -> tuple[float, ...]:
    try:
        with open(Path("configs/universe.yaml"), encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        uni = raw.get("universe", raw) if isinstance(raw, dict) else {}
        grid = uni.get("participation_grid") if isinstance(uni, dict) else None
        if isinstance(grid, list) and grid:
            return tuple(float(x) for x in grid)
    except OSError:
        return (0.01, 0.02, 0.05)
    except (TypeError, ValueError):
        return (0.01, 0.02, 0.05)
    return (0.01, 0.02, 0.05)


def iter_harness_cases(costs: CostConfig, participation_grid: Sequence[float] | None = None) -> Iterator[tuple[CostConfig, float]]:
    parts = tuple(participation_grid) if participation_grid is not None else load_participation_grid()
    for cost in costs.grid():
        for participation in parts:
            yield cost, float(participation)


def run_distribution_eval(
    engine: BacktestEngine,
    simulator: TournamentSimulator,
    model: object,
    panel: object,
    base_config: BacktestConfig,
    *,
    participation: float,
    horizon: int,
    thresholds: Sequence[float],
    tail_weights: dict[float, float],
) -> ReturnDistribution:
    filt = replace(base_config.filters, max_order_to_adv=float(participation))
    config = replace(base_config, filters=filt)
    rolling = simulator.run_rolling(model, panel, config, horizon=horizon)  # type: ignore[arg-type]
    name = getattr(model, "name", "model")
    return ReturnDistribution.summarise(
        name=str(name),
        returns=list(rolling.returns),
        horizon=horizon,
        thresholds=thresholds,
        tail_weights=tail_weights,
        givebacks=list(getattr(rolling, "givebacks", ())),
    )


def harness_case_count(costs: CostConfig, participation_grid: Sequence[float] | None = None) -> int:
    parts = participation_grid if participation_grid is not None else load_participation_grid()
    return len(costs.grid()) * len(parts)
