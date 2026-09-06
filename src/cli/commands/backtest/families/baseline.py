"""Baseline/alpha/champion backtest runner (P4).

Covers the baseline (B0..B5), alpha (M07/M13) and champion families, which
run the common execution path with no family-specific forensics branches.
"""

from __future__ import annotations

from src.cli.commands.backtest._core import run_cells, run_preflight
from src.cli.commands.backtest._prep import prepare_run
from src.cli.context import BacktestContext, BacktestResult


def run_baseline_family(ctx: BacktestContext) -> BacktestResult:
    """Run the baseline family: shared cells, no per-model forensics hooks."""
    if not run_preflight(ctx):
        return BacktestResult(return_code=1, cells=(), strategy_id=ctx.strategy_id)
    prep = prepare_run(ctx)
    cells = run_cells(ctx, prep, None)
    return BacktestResult(return_code=0, cells=tuple(cells), strategy_id=ctx.strategy_id)


__all__ = ["run_baseline_family"]
