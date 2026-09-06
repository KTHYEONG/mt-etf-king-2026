# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.backtest.costs import CostConfig, CostModel
from src.backtest.execution import Fill, NextOpenExecution
from src.backtest.session_cache import build_close_map
from src.backtest.session_grid import resolve_session_grid  # noqa: F401
from src.core.calendar import TradingCalendar
from src.core.config import config_path
from src.core.logging_setup import tagged_log
from src.core.trace import CANDIDATE_CAP, CandidateTrace, GateTrace, NullTraceSink, SessionTrace, TraceSink
from src.features.builder import FeatureBuilder
from src.features.regime import RegimeSnapshot
from pathlib import Path

from src.backtest.pnl import compute_next_open_session_return
from src.execution.ledger import (
    PortfolioLedgerState,
    PortfolioTransitionResult,
    ledger_state_from_weights,
    resolve_session_intent,
    transition_portfolio_state,
)
from src.portfolio.constraints import apply_portfolio_exposure_limits, load_portfolio_exposure_limits, normalize_weights
from src.portfolio.intent import CASH_INTENT, HOLD_INTENT, PortfolioIntent, resolve_portfolio_intent
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.selection import explain_selection_drops
from src.portfolio.sizing import SizingScheme, weights_from_scores
from src.universe.provider import PointInTimeUniverse, UniverseFilters
from src.universe.tournament import TournamentRules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.engine_runner import BacktestEngine  # noqa: F401

# wiring anchors
compute_next_open_session_return(
    weights_before_open={}, weights_after_open={}, prev_closes={}, opens={}, closes={}
)
resolve_portfolio_intent({}, current_weights={}, score_failed=False)

_build_close_map_ref = build_close_map  # noqa: F401




def _append_trades_from_transition(
    trades_records: list[dict[str, object]],
    transition_result: PortfolioTransitionResult,
    decision_date: date,
) -> None:
    for f in transition_result.fills:
        w_after = float(f.target_weight)
        trades_records.append(
            {
                "decision_date": decision_date,
                "execution_date": f.execution_date,
                "ticker": f.ticker,
                "side": "BUY" if w_after >= 0.0 else "SELL",
                "weight_before": 0.0,
                "weight_after": w_after,
                "delta_weight": w_after,
                "weight": w_after,
                "price": float(f.price),
            }
        )


# wiring: PortfolioPolicy.allocate via policy.allocate
_policy_ref = PortfolioPolicy.allocate  # noqa: F401

logger = logging.getLogger("src.backtest.engine")  # pinned name: verbatim-moved from engine.py


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    capital: float
    scheme: SizingScheme
    k: int
    filters: UniverseFilters
    costs: CostConfig


def build_execution_adv(engine: BacktestEngine, tickers: list[str] | set[str] | tuple[str, ...], decision_date: date) -> dict[str, float]:
    from collections.abc import Iterable as _Iterable

    adv: dict[str, float] = {}
    # collect source tickers plus family members
    all_tickers: set[str] = set(str(t) for t in tickers) if tickers else set()
    # also include current_weights family members? Collect via engine.universe.master
    try:
        master = getattr(engine.universe, "master", None)
        if master is not None:
            for t in list(all_tickers):
                try:
                    attr = master.attributes.get(t)  # type: ignore[attr-defined]
                    if attr is not None:
                        fk = getattr(attr, "leverage_family_key", None)
                        if fk:
                            for mt, matr in master.attributes.items():  # type: ignore[attr-defined]
                                if getattr(matr, "leverage_family_key", None) == fk:
                                    all_tickers.add(str(mt))
                except Exception:
                    continue
    except Exception:
        pass
    for ticker in all_tickers:
        try:
            adv_val = engine.universe.adv(str(ticker), decision_date)
            if adv_val is not None:
                adv[str(ticker)] = float(adv_val)
        except Exception:
            continue
    return adv


@dataclass(frozen=True)
class BacktestResult:
    name: str
    daily: pl.DataFrame
    trades: pl.DataFrame
    unfilled: tuple[tuple[date, str], ...]
    config: BacktestConfig
