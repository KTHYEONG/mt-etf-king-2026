"""Backtest run context (P4 decomposition of src/cli/_impl.py).

Hoists the panel/calendar/universe/paths setup that cmd_backtest repeated
inside each strategy branch into a single build step.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from src.core.calendar import TradingCalendar, get_calendar
from src.core.paths import DataPaths
from src.core.settings import get_settings
from src.data.panel import BACKTEST_PANEL_COLUMNS, load_backtest_panel
from src.strategies.protocol import StrategyProtocol
from src.strategies.registry import STRATEGIES, resolve_strategy_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestContext:
    strategy_id: str
    strategy: StrategyProtocol
    start: date
    end: date
    paths: DataPaths
    calendar: TradingCalendar
    panel: pl.DataFrame
    leverage_scenario: str
    eval_mode: str
    protocol: str
    commission_bps: float | None
    slippage_bps: float | None
    participation: float | None
    # Spec-gap completion (P4): the contract's field list omits the run flags
    # that gate per-cell trace/forensics behavior. Defaulted so every existing
    # construction keeps working; required for bitwise-equivalent moves (R14).
    trace: bool = False
    forensics: bool = False


@dataclass
class BacktestCellBundle:
    """Per-grid-cell run data produced by the shared cell loop (no I/O)."""

    run_id: str
    meta: dict[str, object]
    summary: dict[str, object]
    rolling: Any = None
    dist: Any = None
    daily: Any = None
    trades: Any = None
    case_config: Any = None
    engine: Any = None
    model: Any = None
    panel: Any = None
    shared_cache: Any = None
    horizon: int = 0
    leverage_allowed: bool | None = None
    inverse_allowed: bool | None = None
    trace_sink: Any = None
    windows_df: Any = None
    forensics_payload: dict[str, object] | None = None
    cost_label: str = ""


@dataclass
class BacktestResult:
    """Aggregate outcome of a family runner: cell bundles plus return code."""

    return_code: int
    cells: tuple[BacktestCellBundle, ...] = ()
    strategy_id: str = ""


def normalize_cli_model_arg(args: argparse.Namespace) -> str:
    """Canonicalize args.model to a semantic id in place; return it."""
    raw = getattr(args, "model", None)
    if raw is None:
        raise ValueError("missing model arg")
    canonical = resolve_strategy_id(str(raw))
    args.model = canonical
    return canonical


def _make_eval_control_model(model_key: str, eval_mode: str) -> Any:
    from src.tournament.eval_mode import resolve_eval_flags

    model = STRATEGIES[model_key]()
    resolve_eval_flags(model, eval_mode)
    return model


def build_backtest_context(args: argparse.Namespace) -> BacktestContext:
    """Assemble shared backtest inputs; raises ValueError on invalid input."""
    model_name = getattr(args, "model", None)
    start_s = getattr(args, "start", None)
    end_s = getattr(args, "end", None)
    if model_name is None or start_s is None or end_s is None:
        raise ValueError("backtest status=fail error=missing --model/--start/--end")
    strategy_id = resolve_strategy_id(str(model_name))
    if strategy_id not in STRATEGIES:
        raise ValueError(f"backtest status=fail error=unknown model {strategy_id}")
    try:
        start = date.fromisoformat(str(start_s))
        end = date.fromisoformat(str(end_s))
    except Exception as exc:
        raise ValueError(f"backtest status=fail error={exc!r}") from exc
    settings = get_settings()
    paths = DataPaths(root=settings.data_root)
    calendar = get_calendar()
    panel = load_backtest_panel(paths, columns=BACKTEST_PANEL_COLUMNS)
    scenario = getattr(args, "leverage_scenario", "aggressive") or "aggressive"
    eval_mode = getattr(args, "eval_mode", "adoption") or "adoption"
    protocol = getattr(args, "protocol", "single") or "single"
    if getattr(args, "stress_grid", False):
        protocol = "grid"
    strategy = STRATEGIES[strategy_id]()
    # Synthetic-panel fallback for the CLI test harness (moved verbatim from
    # cmd_backtest: unit tests run short ranges with no gold/silver panel).
    if panel is None or panel.height == 0:
        import polars as synthetic_pl

        sessions = calendar.sessions(start, end)
        rows = [
            {
                "date": d,
                "ticker": "069500" if ticker == "069500" else ticker,
                "close": 30000.0,
                "open": 30000.0,
                "high": 30100.0,
                "low": 29900.0,
                "is_tradable": True,
                "trading_value": 5_000_000_000,
                "name": "Test",
                "theme": "ThemeA",
                "underlying_index_name": "IndexA",
                "mom_20": 0.01,
                "mom_20_rs": 0.5,
            }
            for d in sessions
            for ticker in ["069500", "451060", "069500", "123456"]
        ]
        uniq = {}
        for r in rows:
            key = (r["date"], r["ticker"])
            uniq[key] = r
        panel = synthetic_pl.DataFrame(list(uniq.values()))
        with contextlib.suppress(Exception):
            panel = panel.with_columns(synthetic_pl.col("date").cast(synthetic_pl.Date))
    return BacktestContext(
        strategy_id=strategy_id,
        strategy=strategy,
        start=start,
        end=end,
        paths=paths,
        calendar=calendar,
        panel=panel,
        leverage_scenario=str(scenario),
        eval_mode=str(eval_mode),
        protocol=str(protocol),
        commission_bps=getattr(args, "commission_bps", None),
        slippage_bps=getattr(args, "slippage_bps", None),
        participation=getattr(args, "participation", None),
        trace=bool(getattr(args, "trace", False)),
        forensics=bool(getattr(args, "forensics", False)),
    )


__all__ = [
    "BacktestCellBundle",
    "BacktestContext",
    "BacktestResult",
    "_make_eval_control_model",
    "build_backtest_context",
    "normalize_cli_model_arg",
]
