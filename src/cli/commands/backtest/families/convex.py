"""Convex-family backtest runner (P4)."""

from __future__ import annotations

from typing import Final

from src.cli.commands.backtest._core import ForensicsHook, run_cells, run_preflight
from src.cli.commands.backtest._prep import prepare_run
from src.cli.commands.backtest.families.sticky_house import _hook_equity_group
from src.cli.context import BacktestContext, BacktestResult

_CONVEX_FORENSICS: Final[dict[str, ForensicsHook]] = {
    "convex.lottery_impulse": _hook_equity_group,
}


def run_convex_family(ctx: BacktestContext) -> BacktestResult:
    """Run the convex family (P31 branch, shared with the equity group)."""
    if not run_preflight(ctx):
        return BacktestResult(return_code=1, cells=(), strategy_id=ctx.strategy_id)
    prep = prepare_run(ctx)
    hook = _CONVEX_FORENSICS.get(ctx.strategy_id)
    cells = run_cells(ctx, prep, hook)
    return BacktestResult(return_code=0, cells=tuple(cells), strategy_id=ctx.strategy_id)


__all__ = ["run_convex_family"]
