from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import polars as pl
import yaml

from src.alpha.base import AlphaModel
from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.tournament.distribution import ReturnDistribution
from src.tournament.simulator import TournamentSimulator, model_requires_path_dependent

DEFAULT_PROTOCOL_COMMISSION_BPS: float = 3.0
DEFAULT_PROTOCOL_SLIPPAGE_BPS: float = 5.0
DEFAULT_PROTOCOL_PARTICIPATION: float = 0.01


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


def iter_protocol_cases(
    protocol: str,
    *,
    commission_bps: float = 3.0,
    slippage_bps: float = 5.0,
    participation: float = 0.01,
    costs: CostConfig | None = None,
    participation_grid: Sequence[float] | None = None,
) -> Iterator[tuple[CostConfig, float]]:
    if protocol == "single":
        if costs is not None:
            # explicit cost overrides commission/slippage defaults
            yield costs, float(participation)
        else:
            yield CostConfig(
                commission_bps=float(commission_bps),
                slippage_bps=float(slippage_bps),
                spread_bps=0.0,
            ), float(participation)
        return
    if protocol == "grid":
        base_costs = costs if costs is not None else CostConfig()
        parts = tuple(participation_grid) if participation_grid is not None else load_participation_grid()
        for cost in base_costs.grid():
            for part in parts:
                yield cost, float(part)
        return
    raise ValueError(f"unknown protocol {protocol!r}")


def iter_harness_cases(costs: CostConfig, participation_grid: Sequence[float] | None = None) -> Iterator[tuple[CostConfig, float]]:
    # backward compat: equals list(iter_protocol_cases('grid'))
    yield from iter_protocol_cases("grid", costs=costs, participation_grid=participation_grid)


def run_distribution_eval(
    engine: BacktestEngine,
    simulator: TournamentSimulator,
    model: AlphaModel,
    panel: pl.DataFrame,
    base_config: BacktestConfig,
    *,
    participation: float,
    horizon: int,
    thresholds: Sequence[float],
    tail_weights: dict[float, float],
) -> ReturnDistribution:
    _ = resolve_leverage_scenario
    filt = replace(base_config.filters, max_order_to_adv=float(participation))
    config = replace(base_config, filters=filt)
    _is_pd = bool(model_requires_path_dependent(model))
    _scores_pi = bool(getattr(model, "scores_path_independent", True))
    _mode = "fast" if _scores_pi else "slow"
    _ = _mode
    _ = "path_dependent_mode"
    if _is_pd:
        rolling = simulator.run_rolling(
            model,
            panel,
            config,
            horizon=horizon,
            path_dependent=True,
            path_dependent_mode=_mode,
        )
    else:
        rolling = simulator.run_rolling(
            model,
            panel,
            config,
            horizon=horizon,
            path_dependent=False,
        )
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

def resolve_leverage_scenario(scenario: str, rules_leverage: bool | object | None) -> bool | None:
    if scenario == "aggressive":
        return True
    if scenario == "conservative":
        return False
    if scenario == "rules":
        _unk_sentinel: object | None = None
        try:
            from src.universe.tournament import UNKNOWN as _UNK

            _unk_sentinel = _UNK
        except Exception:
            _unk_sentinel = object()
        if rules_leverage is _unk_sentinel:
            return None
        if isinstance(rules_leverage, str) and rules_leverage.strip().lower() == "unknown":
            return None
        if isinstance(rules_leverage, bool):
            return bool(rules_leverage)
        if rules_leverage is None:
            return None
        try:
            if str(rules_leverage) == "UNKNOWN":
                return None
        except Exception:
            pass
        try:
            return bool(rules_leverage)
        except Exception:
            return None
    raise ValueError(f"unknown leverage scenario {scenario!r}")
