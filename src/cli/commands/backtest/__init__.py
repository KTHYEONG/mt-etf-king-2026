"""Backtest command (P4 decomposition of src/cli/_impl.py).

Dispatches to family runners by semantic family prefix instead of the
legacy 34-way strategy-identity if-chain.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping
from typing import Final

from src.cli.commands.backtest.artifacts import attach_backtest_artifacts, emit_backtest_artifacts
from src.cli.commands.backtest.families.baseline import run_baseline_family
from src.cli.commands.backtest.families.convex import run_convex_family
from src.cli.commands.backtest.families.portfolio import run_portfolio_family
from src.cli.commands.backtest.families.sticky import run_sticky_family
from src.cli.context import BacktestContext, BacktestResult, build_backtest_context
from src.strategies.registry import family_of, resolve_strategy_id

logger = logging.getLogger(__name__)

FAMILY_RUNNERS: Final[Mapping[str, Callable[[BacktestContext], BacktestResult]]] = {
    "baseline": run_baseline_family,
    "alpha": run_baseline_family,
    "portfolio": run_portfolio_family,
    "sticky": run_sticky_family,
    "convex": run_convex_family,
    "champion": run_baseline_family,
}


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest: resolve, build context, dispatch by family, emit."""
    try:
        strategy_id = resolve_strategy_id(str(args.model))
        ctx = build_backtest_context(args)
        result = FAMILY_RUNNERS[family_of(strategy_id)](ctx)
        artifacts = attach_backtest_artifacts(ctx, result)
        emit_backtest_artifacts(ctx, result, artifacts)
        return result.return_code
    except Exception as exc:
        logger.error(f"[SYS] backtest status=fail error={exc!r}")
        return 1


__all__ = ["FAMILY_RUNNERS", "cmd_backtest"]
