"""Portfolio-family backtest runner (P4). Table dispatch; hooks live in split modules."""

from __future__ import annotations

import logging
from typing import Final

from src.cli.commands.backtest._core import (
    ForensicsHook,
    run_cells,
    run_preflight,
)
from src.cli.commands.backtest._prep import prepare_run
from src.cli.commands.backtest.families.portfolio_exposure import (
    _hook_convexity,
    _hook_lottery_exposure,
    _hook_lottery_rebalance,
)
from src.cli.commands.backtest.families.portfolio_leadership import (
    _hook_leadership_confidence,
    _hook_leadership_policy,
)
from src.cli.commands.backtest.families.portfolio_momentum import (
    _hook_momentum_confidence,
    _hook_momentum_vehicle,
)
from src.cli.context import BacktestContext, BacktestResult

logger = logging.getLogger(__name__)


_PORTFOLIO_FORENSICS: Final[dict[str, ForensicsHook]] = {
    "portfolio.momentum_vehicle": _hook_momentum_vehicle,
    "portfolio.momentum_confidence": _hook_momentum_confidence,
    "portfolio.leadership_policy": _hook_leadership_policy,
    "portfolio.leadership_confidence": _hook_leadership_confidence,
    "portfolio.lottery_exposure": _hook_lottery_exposure,
    "portfolio.lottery_rebalance": _hook_lottery_rebalance,
    "portfolio.convexity_hold": _hook_convexity,
    "portfolio.convexity_rebalance": _hook_convexity,
    "portfolio.convexity_variant": _hook_convexity,
}


def run_portfolio_family(ctx: BacktestContext) -> BacktestResult:
    """Run the portfolio family (P08..P19): shared cells plus family forensics."""
    if not run_preflight(ctx):
        return BacktestResult(return_code=1, cells=(), strategy_id=ctx.strategy_id)
    prep = prepare_run(ctx)
    hook = _PORTFOLIO_FORENSICS.get(ctx.strategy_id)
    cells = run_cells(ctx, prep, hook)
    return BacktestResult(return_code=0, cells=tuple(cells), strategy_id=ctx.strategy_id)


__all__ = ["run_portfolio_family"]
