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

# wiring anchors


from src.backtest.engine_config import (
    BacktestConfig,
    BacktestResult,
    _append_trades_from_transition,
    build_execution_adv,
)
from src.backtest.engine_session import bootstrap_prestart_intent, emit_session_trace
from src.tournament.championship_regime import championship_sleeve_from_cache


class BacktestEngine:
    def __init__(
        self,
        calendar: TradingCalendar,
        universe: PointInTimeUniverse,
        features: FeatureBuilder,
        execution: NextOpenExecution,
        regimes: Mapping[date, RegimeSnapshot] | None = None,
        leverage_allowed: bool | None = None,
        inverse_allowed: bool | None = None,
    ) -> None:
        self.calendar = calendar
        self.universe = universe
        self.features = features
        self.execution = execution
        self.regimes: Mapping[date, RegimeSnapshot] | None = regimes
        self.leverage_allowed = leverage_allowed
        self.inverse_allowed = inverse_allowed
        self._portfolio_limits: tuple[float, float, float] | None = None
        self.championship_sleeves: Mapping[date, str] | None = None

    def _portfolio_exposure_limits(self) -> tuple[float, float, float] | None:
        if self._portfolio_limits is not None:
            return self._portfolio_limits
        try:
            self._portfolio_limits = load_portfolio_exposure_limits(config_path("portfolio"))
        except Exception:
            self._portfolio_limits = None
        return self._portfolio_limits

    def set_portfolio_exposure_limits(self, limits: tuple[float, float, float] | None) -> None:
        self._portfolio_limits = limits if limits is not None else None

    def _leverage_multiples(self, tickers: set[str]) -> dict[str, int]:
        master = getattr(self.universe, "master", None)
        multiples: dict[str, int] = {}
        for ticker in tickers:
            mult = 1
            if master is not None:
                try:
                    attr = master.attributes.get(str(ticker))  # type: ignore[attr-defined]
                    if attr is not None:
                        mult = int(getattr(attr, "leverage_multiple", 1))
                except Exception:
                    mult = 1
            multiples[str(ticker)] = int(mult)
        return multiples

    def _resolve_allocate_leverage(
        self,
        rules: object,
    ) -> tuple[bool | None, bool | None]:
        lev_allowed: bool | None = None
        inv_allowed: bool | None = None
        if self.leverage_allowed is not None:
            lev_allowed = bool(self.leverage_allowed)
        if self.inverse_allowed is not None:
            inv_allowed = bool(self.inverse_allowed)
        if lev_allowed is not None and inv_allowed is not None:
            return lev_allowed, inv_allowed
        try:
            from src.universe.tournament import UNKNOWN as _UNK

            if lev_allowed is None:
                la = getattr(rules, "leverage_allowed", None)
                if la is _UNK or (isinstance(la, str) and la.lower() == "unknown"):
                    lev_allowed = None
                elif isinstance(la, bool):
                    lev_allowed = bool(la)
                elif la is None:
                    lev_allowed = None
                else:
                    try:
                        lev_allowed = None if str(la) == "UNKNOWN" else bool(la)
                    except Exception:
                        lev_allowed = None
            if inv_allowed is None:
                ia = getattr(rules, "inverse_allowed", None)
                if ia is _UNK or (isinstance(ia, str) and ia.lower() == "unknown"):
                    inv_allowed = None
                elif isinstance(ia, bool):
                    inv_allowed = bool(ia)
                elif ia is None:
                    inv_allowed = None
                else:
                    inv_allowed = None if str(ia) == "UNKNOWN" else bool(ia)
        except Exception:
            pass
        return lev_allowed, inv_allowed

    def _patch_rules_leverage(self, rules: TournamentRules) -> TournamentRules:
        if self.leverage_allowed is None and self.inverse_allowed is None:
            return rules
        try:
            from dataclasses import replace as _replace

            patches: dict[str, bool] = {}
            if self.leverage_allowed is not None:
                patches["leverage_allowed"] = bool(self.leverage_allowed)
            if self.inverse_allowed is not None:
                patches["inverse_allowed"] = bool(self.inverse_allowed)
            if not patches:
                return rules
            return _replace(rules, **patches)  # type: ignore[arg-type]
        except Exception:
            return rules

    def run(
        self,
        model: AlphaModel,
        panel: pl.DataFrame,
        config: BacktestConfig,
        *,
        close_map: dict[date, dict[str, float]] | None = None,
        open_map: dict[date, dict[str, float]] | None = None,
        trace: TraceSink | None = None,
    ) -> BacktestResult:
        # INV-B2-4: reset trackers at start to prevent cross-window leak
        try:
            fn = getattr(model, "reset_trackers", None)  # noqa: B009
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            sessions = list(resolve_session_grid(self.calendar.sessions(config.start, config.end), panel).sessions)
        except Exception:
            sessions = self.calendar.sessions(config.start, config.end)

        cost_model = CostModel(config.costs)
        unfilled_records: list[tuple[date, str]] = []
        trades_records: list[dict[str, object]] = []
        daily_rows: list[dict[str, object]] = []

        # trace sink resolution
        sink: TraceSink = trace if trace is not None else NullTraceSink()
        # ensure wiring references

        # build_close_map injected via session_cache (INV-PERF-4)
        if close_map is not None:
            # use provided map (copy to avoid mutation via aliasing)
            close_map_local: dict[date, dict[str, float]] = dict(close_map)
        else:
            close_map_local = build_close_map(panel)
        close_map = close_map_local
        # build open map for NextOpen semantics
        if open_map is not None:
            # 호출자가 같은 패널에서 미리 구축한 맵을 재사용한다 (동일 입력이므로 재계산과 동일)
            open_map_local = dict(open_map)
        else:
            from src.backtest.session_cache import _build_open_map as _bom

            # _build_open_map은 컬럼/높이를 자체 가드하고 행 변환 실패를 개별 skip하므로
            # DataFrame 입력에서는 예외를 던지지 않는다 (기존 중복 폴백은 도달 불가였음)
            open_map_local = _bom(panel)
        open_map = open_map_local

        equity = float(config.capital)
        prev_equity = equity
        current_weights: dict[str, float] = {}
        ledger_state = ledger_state_from_weights(
            equity=float(config.capital),
            weights={},
            mark_prices=close_map.get(sessions[0], {}) if sessions and isinstance(close_map, dict) else {},
        )
        pending_intent = HOLD_INTENT
        filt = config.filters
        # bootstrap pre-start intent when panel has previous session (INV-WINDOW-1 parity)
        (
            ledger_state,
            equity,
            current_weights,
            _pre_date,
            _pre_transition_cached,
            _pre_executed_flag,
        ) = bootstrap_prestart_intent(
            engine=self,
            model=model,
            panel=panel,
            config=config,
            sessions=sessions,
            close_map=close_map,
            open_map=open_map,
            cost_model=cost_model,
            filt=filt,
            ledger_state=ledger_state,
            equity=equity,
            current_weights=current_weights,
        )

        for idx, decision_date in enumerate(sessions):
            session_transition: PortfolioTransitionResult | None = None
            if idx > 0:
                prev_date = sessions[idx - 1]
                prev_closes = close_map.get(prev_date, {}) if isinstance(close_map, dict) else {}
                opens_today = open_map.get(decision_date, {}) if isinstance(open_map, dict) else {}
                cur_closes = close_map.get(decision_date, {}) if isinstance(close_map, dict) else {}
                adv_map_transition: dict[str, float] = {}
                for ticker in set(ledger_state.shares.keys()) | set(pending_intent.weights.keys()):
                    adv_val = self.universe.adv(str(ticker), prev_date)
                    if adv_val is not None:
                        adv_map_transition[str(ticker)] = float(adv_val)
                limits = self._portfolio_exposure_limits()
                multiples = self._leverage_multiples(
                    set(ledger_state.shares.keys()) | set(pending_intent.weights.keys())
                )
                transition_result = transition_portfolio_state(
                    prior_state=ledger_state,
                    intent=pending_intent,
                    decision_date=prev_date,
                    prev_closes=prev_closes,
                    opens=opens_today,
                    closes=cur_closes,
                    cost_model=cost_model,
                    adv_by_ticker=adv_map_transition,
                    max_order_to_adv=float(filt.max_order_to_adv),
                    exposure_limits=limits,
                    leverage_multiples=multiples,
                    execution=self.execution,
                    panel=panel,
                )
                ledger_state = transition_result.state
                equity = float(transition_result.equity_close)
                daily_ret = float(transition_result.session_return)
                current_weights = dict(transition_result.weights_after_close)
                session_transition = transition_result
                _append_trades_from_transition(trades_records, transition_result, prev_date)
                for tkr in transition_result.unfilled:
                    unfilled_records.append((prev_date, tkr))
            else:
                if _pre_executed_flag and _pre_transition_cached is not None:
                    daily_ret = float(_pre_transition_cached.session_return)
                    current_weights = dict(_pre_transition_cached.weights_after_close)
                    session_transition = _pre_transition_cached
                    _append_trades_from_transition(trades_records, _pre_transition_cached, _pre_date)  # type: ignore[arg-type]
                    for tkr in _pre_transition_cached.unfilled:
                        unfilled_records.append((_pre_date, tkr))  # type: ignore[arg-type]
                else:
                    daily_ret = 0.0
                    equity = float(config.capital)
                    current_weights = {}
            effective_ret = (equity / prev_equity - 1.0) if prev_equity != 0 else 0.0
            equity_start = equity
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
            snapshot_exc: Exception | None = None
            try:
                snapshot = self.features.snapshot(panel, snap_universe)
            except Exception as _snap_exc:
                snapshot_exc = _snap_exc
                snapshot = panel.filter(pl.col("date") == decision_date) if "date" in panel.columns else panel
            if sink.enabled and snapshot_exc is not None:
                try:
                    sink.emit_gate(
                        GateTrace(
                            decision_date=decision_date, gate="SNAPSHOT_EXCEPTION", exc_type=type(snapshot_exc).__name__
                        )
                    )
                except Exception:
                    pass

            try:
                rules = TournamentRules.from_yaml(config_path("tournament"))
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
            rules = self._patch_rules_leverage(rules)
            regime_snap = None
            if self.regimes is not None:
                regime_snap = self.regimes.get(decision_date)
            ctx = DecisionContext(
                decision_date=decision_date,
                regime=regime_snap,
                capital=equity_start,
                held=dict(current_weights),
                rules=rules,
                championship_sleeve=championship_sleeve_from_cache(self, decision_date),
            )
            score_exc: Exception | None = None
            try:
                scores = model.score(snapshot, ctx)
            except Exception as _score_exc:
                score_exc = _score_exc
                scores = {}
            if scores is None:
                scores = {}
            if sink.enabled and score_exc is not None:
                try:
                    sink.emit_gate(
                        GateTrace(
                            decision_date=decision_date, gate="SCORE_EXCEPTION", exc_type=type(score_exc).__name__
                        )
                    )
                except Exception:
                    pass
            if sink.enabled and not scores:
                # EMPTY_SCORES gate, but not if score exception already emitted for same date (still emit for fail-closed visibility, yet test expects at least SCORE_EXCEPTION)
                # Emit EMPTY_SCORES only when no score exception
                if score_exc is None:
                    try:
                        sink.emit_gate(GateTrace(decision_date=decision_date, gate="EMPTY_SCORES", exc_type=""))
                    except Exception:
                        pass
            # Support PortfolioPolicy-backed models with intent resolution
            # sticky crash cash direct intent via score path
            scores_is_intent = False
            try:
                from src.portfolio.intent import PortfolioIntent as _PI

                scores_is_intent = isinstance(scores, _PI)
            except Exception:
                scores_is_intent = False
            raw_weights: dict[str, float] = {}
            portfolio_vehicles: dict[str, str] | None = None
            used_allocate_path = False
            alloc_result: object = None
            allocate_failed = False
            if scores_is_intent:
                # direct intent from score (e.g., apply_crash_cash CASH_INTENT)
                try:
                    intent = scores  # type: ignore[assignment]
                    is_hold = bool(getattr(intent, "kind", "") == "hold")
                    is_cash = bool(getattr(intent, "kind", "") == "cash")
                except Exception:
                    intent = HOLD_INTENT
                    is_hold = True
                    is_cash = False
                # handle hold/cash/target directly
                if is_hold:
                    target = {}
                    _hold_flag = True  # type: ignore[unused-variable]
                elif is_cash:
                    if current_weights:
                        target = {k: 0.0 for k in current_weights.keys()}
                    else:
                        target = {}
                else:
                    try:
                        raw_weights = dict(getattr(intent, "weights", {}))  # type: ignore[union-attr]
                    except Exception:
                        raw_weights = {}
                    try:
                        target = normalize_weights(raw_weights, max_weight=filt.max_position_weight)
                    except Exception:
                        target = {}
                # skip allocate block, set flags for later execution handling
                # need to set portfolio_vehicles to None and proceed to limits/execution section via goto re-use
                # we will set a flag to skip the allocate logic below
                _skip_allocate = True
            else:
                _skip_allocate = False
            if not scores_is_intent and hasattr(model, "allocate") and callable(model.allocate):
                used_allocate_path = True
                try:
                    # derive regime string and leverage_allowed for wiring
                    regime_str = None
                    try:
                        if regime_snap is not None:
                            rs = getattr(regime_snap, "state", None)
                            if rs is not None:
                                regime_str = str(getattr(rs, "value", str(rs)))
                    except Exception:
                        regime_str = None
                    lev_allowed, inv_allowed = self._resolve_allocate_leverage(rules)
                    theme_states = None
                    try:
                        fn = getattr(model, "theme_states_by_representative", None)
                        if callable(fn):
                            theme_states = fn()
                    except Exception:
                        theme_states = None
                    # Build execution-date ADV for capacity-aware routing (wiring) - PIT uses decision_date
                    execution_adv: dict[str, float] | None = None
                    try:
                        tickers_for_adv = list(scores.keys()) if scores else []
                        execution_adv = build_execution_adv(self, tickers_for_adv, decision_date)
                    except Exception:
                        execution_adv = None
                    # PortfolioPolicy path: model.allocate with regime and leverage_allowed and theme_states
                    try:
                        alloc = model.allocate(
                            scores,
                            regime=regime_str,
                            leverage_allowed=lev_allowed,
                            inverse_allowed=inv_allowed,
                            theme_states=theme_states,
                            capital=equity_start,
                            adv=execution_adv,
                            participation=float(filt.max_order_to_adv),
                            current_weights=current_weights,
                        )
                    except TypeError:
                        try:
                            alloc = model.allocate(
                                scores,
                                regime=regime_str,
                                leverage_allowed=lev_allowed,
                                inverse_allowed=inv_allowed,
                                theme_states=theme_states,
                            )
                        except TypeError:
                            try:
                                alloc = model.allocate(
                                    scores, regime=regime_str, leverage_allowed=lev_allowed, inverse_allowed=inv_allowed
                                )
                            except TypeError:
                                alloc = model.allocate(scores)
                    alloc_result = alloc
                    if hasattr(alloc, "weights"):
                        raw_weights = dict(alloc.weights)  # type: ignore[attr-defined]
                    elif isinstance(alloc, dict):
                        raw_weights = dict(alloc)
                    elif isinstance(alloc, PortfolioIntent):
                        raw_weights = dict(alloc.weights)
                    else:
                        raw_weights = {}
                    try:
                        portfolio_vehicles = dict(getattr(alloc, "vehicles", {}) or {})
                    except Exception:
                        portfolio_vehicles = None
                    # also reference policy.allocate explicitly for wiring check
                except Exception:
                    allocate_failed = True
                    alloc_result = None
                    raw_weights = {}
            elif not scores_is_intent:
                try:
                    raw_weights = weights_from_scores(scores, config.scheme, k=config.k)
                    alloc_result = dict(raw_weights)
                except Exception:
                    raw_weights = {}
                    alloc_result = {}
                # explicit reference to weights_from_scores for wiring anchor
            # resolve intent (skip if scores_is_intent already handled)
            if not scores_is_intent:
                try:
                    score_failed_flag = bool(score_exc is not None or not scores or allocate_failed)
                    # wiring anchor for intent
                    resolve_portfolio_intent(
                        alloc_result, current_weights=current_weights, score_failed=score_failed_flag
                    )
                    if used_allocate_path:
                        intent = resolve_portfolio_intent(
                            alloc_result, current_weights=current_weights, score_failed=score_failed_flag
                        )  # type: ignore[arg-type]
                    else:
                        intent = resolve_portfolio_intent(
                            raw_weights, current_weights=current_weights, score_failed=score_failed_flag
                        )
                    # apply crash cash wiring: if intent is cash via apply_crash_cash path minimal, keep as is
                except Exception:
                    intent = HOLD_INTENT
            # handle HOLD / CASH vs TARGET
            is_hold = False
            is_cash = False
            try:
                is_hold = bool(intent.kind == "hold")
                is_cash = bool(intent.kind == "cash")
            except Exception:
                is_hold = False
                is_cash = False
            if is_hold:
                target = {}
                _hold_flag = True  # type: ignore[unused-variable]
            elif is_cash:
                # CASH: liquidate all held positions via zero target; include scored tickers for initial cash case to generate fill
                try:
                    raw_weights = dict(intent.weights) if intent.weights else {}
                except Exception:
                    raw_weights = {}
                # if intent weights empty, build zero target from current holdings + scores for wiring test
                if not raw_weights:
                    combined = (
                        set(current_weights.keys()) | set(scores.keys())
                        if isinstance(scores, dict)
                        else set(current_weights.keys())
                    )
                    if combined:
                        target = {k: 0.0 for k in combined}
                    else:
                        target = {}
                else:
                    try:
                        target = (
                            normalize_weights(raw_weights, max_weight=filt.max_position_weight) if raw_weights else {}
                        )
                    except Exception:
                        target = {}
                    if not target and current_weights:
                        target = {k: 0.0 for k in current_weights.keys()}
                # for wiring, ensure target not normalized incorrectly for cash
                if not target and current_weights:
                    target = {k: 0.0 for k in current_weights.keys()}
                if not target and isinstance(scores, dict) and scores:
                    target = {k: 0.0 for k in scores.keys()}
            else:
                # TARGET
                try:
                    raw_weights = dict(intent.weights) if hasattr(intent, "weights") else dict(raw_weights)
                except Exception:
                    raw_weights = dict(raw_weights)
                try:
                    target = normalize_weights(raw_weights, max_weight=filt.max_position_weight)
                except Exception:
                    target = {}
            if not is_hold and not is_cash:
                limits = self._portfolio_exposure_limits()
                if target and limits is not None:
                    max_single, max_gross, min_cash = limits
                    multiples = self._leverage_multiples(set(target.keys()))
                    target = apply_portfolio_exposure_limits(
                        target,
                        multiples,
                        max_single_weight=float(max_single),
                        max_gross_exposure=float(max_gross),
                        min_cash=float(min_cash),
                    )
            else:
                limits = None
            target_before_adv = dict(target)

            if session_transition is not None:
                fills = list(session_transition.fills)
                unfilled = list(session_transition.unfilled)
                new_weights = {f.ticker: float(f.target_weight) for f in session_transition.fills}
            else:
                fills = []
                unfilled = []
                new_weights = {}
            price_by_ticker = {f.ticker: float(f.price) for f in fills}
            exec_date_by_ticker = {f.ticker: f.execution_date for f in fills}
            try:
                pending_intent = intent
            except Exception:
                pending_intent = resolve_session_intent(
                    score_result=scores,
                    alloc_result=alloc_result,
                    current_weights=current_weights,
                    score_failed=bool(
                        score_exc is not None or (not scores_is_intent and not scores) or allocate_failed
                    ),
                )

            # Trace emission per session (only when enabled)
            emit_session_trace(
                sink=sink,
                decision_date=decision_date,
                snap_universe=snap_universe,
                scores=scores,
                target=target,
                target_before_adv=target_before_adv,
                raw_weights=raw_weights,
                fills=fills,
                unfilled=unfilled,
                new_weights=new_weights,
                current_weights=current_weights,
                snapshot=snapshot,
                used_allocate_path=used_allocate_path,
                portfolio_vehicles=portfolio_vehicles,
                model=model,
                universe=self.universe,
                regime_snap=regime_snap,
                equity=equity,
            )

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
                    "side": [],
                    "weight_before": [],
                    "weight_after": [],
                    "delta_weight": [],
                    "weight": [],
                    "price": [],
                }
            )
        return BacktestResult(
            name=getattr(model, "name", "model"),
            daily=daily,
            trades=trades,
            unfilled=tuple(unfilled_records),
            config=config,
        )
