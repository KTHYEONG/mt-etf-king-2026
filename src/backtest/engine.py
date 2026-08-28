from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.backtest.costs import CostConfig, CostModel
from src.backtest.execution import Fill, NextOpenExecution
from src.backtest.liquidity import cap_target_weights_by_adv
from src.core.calendar import TradingCalendar
from src.features.builder import FeatureBuilder
from src.features.regime import RegimeSnapshot
from src.portfolio.constraints import normalize_weights
from src.portfolio.sizing import SizingScheme, weights_from_scores
from src.universe.provider import PointInTimeUniverse, UniverseFilters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    capital: float
    scheme: SizingScheme
    k: int
    filters: UniverseFilters
    costs: CostConfig


@dataclass(frozen=True)
class BacktestResult:
    name: str
    daily: pl.DataFrame
    trades: pl.DataFrame
    unfilled: tuple[tuple[date, str], ...]
    config: BacktestConfig


class BacktestEngine:
    def __init__(
        self,
        calendar: TradingCalendar,
        universe: PointInTimeUniverse,
        features: FeatureBuilder,
        execution: NextOpenExecution,
        regimes: Mapping[date, RegimeSnapshot] | None = None,
    ) -> None:
        self.calendar = calendar
        self.universe = universe
        self.features = features
        self.execution = execution
        self.regimes: Mapping[date, RegimeSnapshot] | None = regimes

    def run(
        self,
        model: AlphaModel,
        panel: pl.DataFrame,
        config: BacktestConfig,
    ) -> BacktestResult:
        try:
            sessions = self.calendar.sessions(config.start, config.end)
        except Exception:
            sessions = []

        cost_model = CostModel(config.costs)
        unfilled_records: list[tuple[date, str]] = []
        trades_records: list[dict[str, object]] = []
        daily_rows: list[dict[str, object]] = []

        close_map: dict[date, dict[str, float]] = {}
        if panel.height > 0 and {"date", "ticker", "close"} <= set(panel.columns):
            for row in panel.iter_rows(named=True):
                d = row.get("date")
                t = str(row.get("ticker"))
                c = row.get("close")
                if d is None or c is None:
                    continue
                try:
                    cf = float(c)
                except Exception:
                    continue
                close_map.setdefault(d, {})[t] = cf

        equity = float(config.capital)
        prev_equity = equity
        current_weights: dict[str, float] = {}
        pending_execution: tuple[list[Fill], float, dict[str, float]] | None = None
        filt = config.filters

        for idx, decision_date in enumerate(sessions):
            equity_start = prev_equity
            if pending_execution is not None:
                _fills, turnover_weight, new_weights = pending_execution
                traded_notional = turnover_weight * equity_start
                cost = cost_model.charge(traded_notional) if traded_notional else 0.0
                equity_start -= cost
                if new_weights:
                    current_weights = new_weights
                pending_execution = None

            daily_ret = 0.0
            if idx > 0 and current_weights:
                prev_date = sessions[idx - 1]
                cur_closes = close_map.get(decision_date, {})
                prev_closes = close_map.get(prev_date, {})
                for tkr, w in current_weights.items():
                    pc = prev_closes.get(tkr)
                    cc = cur_closes.get(tkr)
                    if pc is None or cc is None or pc == 0:
                        continue
                    daily_ret += float(w) * (cc / pc - 1.0)

            equity = equity_start * (1.0 + daily_ret)
            if equity < 0:
                equity = 0.0
            effective_ret = (equity / prev_equity - 1.0) if prev_equity != 0 else 0.0
            daily_rows.append(
                {
                    "date": decision_date,
                    "ret": float(effective_ret),
                    "equity": float(equity),
                    "return": float(effective_ret),
                }
            )
            prev_equity = equity

            snap_universe = self.universe.get(decision_date, filt)
            try:
                snapshot = self.features.snapshot(panel, snap_universe)
            except Exception:
                snapshot = panel.filter(pl.col("date") == decision_date) if "date" in panel.columns else panel

            from src.universe.tournament import TournamentRules

            try:
                rules = TournamentRules.from_yaml(__import__("pathlib").Path("configs/tournament.yaml"))
            except Exception:
                comm_val = config.costs.commission_bps if config.costs.commission_bps is not None else 0.0
                slip_val = config.costs.slippage_bps if config.costs.slippage_bps is not None else 0.0
                rules = TournamentRules(
                    name="default",
                    start_date=config.start,
                    end_date=config.end,
                    initial_capital=int(config.capital),
                    category="autonomous",
                    leverage_allowed=True,
                    inverse_allowed=True,
                    max_weight=1.0,
                    cash_allowed=True,
                    sponsor_etf_only=False,
                    manifest_path=None,
                    issuer_whitelist=None,
                    commission_bps=float(comm_val),
                    slippage_bps=float(slip_val),
                    max_order_to_adv=filt.max_order_to_adv,
                    stress_grid=(0.01, 0.02, 0.05, 0.10),
                )
            regime_snap = None
            if self.regimes is not None:
                regime_snap = self.regimes.get(decision_date)
            ctx = DecisionContext(
                decision_date=decision_date,
                regime=regime_snap,
                capital=equity_start,
                held=dict(current_weights),
                rules=rules,
            )
            try:
                scores = model.score(snapshot, ctx)
            except Exception:
                scores = {}
            if scores is None:
                scores = {}
            try:
                raw_weights = weights_from_scores(scores, config.scheme, k=config.k)
            except Exception:
                raw_weights = {}
            try:
                target = normalize_weights(raw_weights, max_weight=filt.max_position_weight)
            except Exception:
                target = {}

            try:
                execution_date = self.calendar.next_session(decision_date)
            except Exception:
                execution_date = None
            if execution_date is not None and target:
                adv_map: dict[str, float] = {}
                for ticker in set(target.keys()) | set(current_weights.keys()):
                    adv_val = self.universe.adv(str(ticker), execution_date)
                    if adv_val is not None:
                        adv_map[str(ticker)] = float(adv_val)
                if adv_map:
                    target = cap_target_weights_by_adv(
                        target,
                        current_weights,
                        equity_start,
                        adv_map,
                        filt.max_order_to_adv,
                    )

            fills, unfilled = self.execution.resolve(target, panel, decision_date)
            for tkr in unfilled:
                unfilled_records.append((decision_date, tkr))
            for f in fills:
                trades_records.append(
                    {
                        "decision_date": decision_date,
                        "execution_date": f.execution_date,
                        "ticker": f.ticker,
                        "price": f.price,
                        "weight": f.target_weight,
                    }
                )
            new_weights = {f.ticker: f.target_weight for f in fills}
            all_tickers = set(current_weights.keys()) | set(new_weights.keys())
            turnover_weight = sum(abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0)) for t in all_tickers)
            if fills:
                pending_execution = (fills, turnover_weight, new_weights)

        if daily_rows:
            daily = pl.DataFrame(daily_rows)
            try:
                daily = daily.with_columns(pl.col("date").cast(pl.Date))
            except Exception:
                pass
        else:
            daily = pl.DataFrame({"date": [], "ret": [], "equity": []})
        if trades_records:
            trades = pl.DataFrame(trades_records)
        else:
            trades = pl.DataFrame(
                {
                    "decision_date": [],
                    "execution_date": [],
                    "ticker": [],
                    "price": [],
                    "weight": [],
                }
            )
        return BacktestResult(
            name=getattr(model, "name", "model"),
            daily=daily,
            trades=trades,
            unfilled=tuple(unfilled_records),
            config=config,
        )
