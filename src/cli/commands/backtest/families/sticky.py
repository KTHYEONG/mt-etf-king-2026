"""Sticky-family backtest runner (P4). Table-driven dispatch, no identity branches."""

from __future__ import annotations

from typing import Final

from src.cli.commands.backtest._core import ForensicsHook, run_cells, run_preflight
from src.cli.commands.backtest._prep import prepare_run
from src.cli.commands.backtest.families.sticky_cash import CASH_HOOKS
from src.cli.commands.backtest.families.sticky_champion import CHAMPION_HOOKS
from src.cli.commands.backtest.families.sticky_core import CORE_HOOKS
from src.cli.commands.backtest.families.sticky_house import EQUITY_GROUP_IDS, HOUSE_HOOKS, _hook_equity_group
from src.cli.commands.backtest.families.sticky_locks import LOCK_HOOKS
from src.cli.commands.backtest.families.sticky_mom60 import MOM60_HOOKS
from src.cli.commands.backtest.families.sticky_split import SPLIT_HOOKS
from src.cli.context import BacktestContext, BacktestResult

_STICKY_FORENSICS: Final[dict[str, ForensicsHook]] = {
    **LOCK_HOOKS,
    **CORE_HOOKS,
    **SPLIT_HOOKS,
    **MOM60_HOOKS,
    **CHAMPION_HOOKS,
    **CASH_HOOKS,
    **HOUSE_HOOKS,
    **{strategy_id: _hook_equity_group for strategy_id in EQUITY_GROUP_IDS if strategy_id.startswith("sticky.")},
}


def run_sticky_family(ctx: BacktestContext) -> BacktestResult:
    """Run the sticky family (P20..P28B/P29/P29V/P30/P33 branches)."""
    if not run_preflight(ctx):
        return BacktestResult(return_code=1, cells=(), strategy_id=ctx.strategy_id)
    prep = prepare_run(ctx)
    hook = _STICKY_FORENSICS.get(ctx.strategy_id)
    cells = run_cells(ctx, prep, hook)
    return BacktestResult(return_code=0, cells=tuple(cells), strategy_id=ctx.strategy_id)


__all__ = ["run_sticky_family"]
