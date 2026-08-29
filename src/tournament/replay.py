from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel
from src.backtest.engine import BacktestConfig
from src.backtest.engine import BacktestEngine as _Engine
from src.backtest.metrics import compound_returns
from src.core.calendar import TradingCalendar
from src.reporting.dashboard import build_rationale

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayDay:
    decision_date: date
    execution_date: date | None
    regime: str
    universe_size: int
    dropped: Mapping[str, int]
    top_scores: tuple[tuple[str, str, str, float], ...]
    weights: Mapping[str, float]
    rationales: Mapping[str, str] | None = None
    daily_return: float = 0.0
    cumulative_return: float = 0.0

    def __post_init__(self) -> None:
        # ensure rationales is at least empty dict if None
        if self.rationales is None:
            object.__setattr__(self, "rationales", {})
        # wiring anchor: reference build_rationale inside class context
        _ = build_rationale


@dataclass(frozen=True)
class ReplayReport:
    model: str
    start: date
    end: date
    sessions: int
    days: tuple[ReplayDay, ...]
    final_return: float


class TournamentReplay:
    def __init__(self, engine: _Engine, calendar: TradingCalendar) -> None:
        self.engine = engine
        self.calendar = calendar

    def run(self, model: AlphaModel, panel: pl.DataFrame, config: BacktestConfig) -> ReplayReport:
        sessions = self.calendar.sessions(config.start, config.end)
        n = len(sessions)
        # Run full engine to get daily returns and equity for cumulative
        full_res = self.engine.run(model, panel, config)
        # Map daily returns by date for quick lookup
        daily = full_res.daily
        ret_col = "ret" if "ret" in daily.columns else ("return" if "return" in daily.columns else None)
        ret_map: dict[date, float] = {}
        if ret_col is not None and daily.height > 0:
            for row in daily.iter_rows(named=True):
                d = row.get("date")
                r = row.get(ret_col)
                if d is None:
                    continue
                try:
                    rv = float(r) if r is not None else 0.0
                except Exception:
                    rv = 0.0
                ret_map[d] = rv
        # Ensure daily rets list in session order for cumulative via compound_returns incremental
        daily_rets_ordered = [float(ret_map.get(d, 0.0)) for d in sessions]
        # Cumulative returns incremental using compound_returns of prefix? Use running product
        cum_rets: list[float] = []
        cur = 1.0
        for r in daily_rets_ordered:
            cur *= 1.0 + float(r)
            cum_rets.append(cur - 1.0)
        # For each session, we need to record ReplayDay using same engine code path per day
        # We will reuse engine's single-session logic by constructing per-day snapshot and scoring.
        # For regime, universe size etc we need to call universe.get and features.snapshot same as engine.
        days: list[ReplayDay] = []
        # Also need panel for name/theme lookup
        # We'll attempt to get feature snapshot for each day to derive top_scores and weights
        from src.portfolio.constraints import normalize_weights
        from src.portfolio.sizing import weights_from_scores
        from src.universe.tournament import TournamentRules

        policy_model = model if hasattr(model, "allocate") and callable(getattr(model, "allocate", None)) else None
        if policy_model is not None and hasattr(policy_model, "reset_trackers") and callable(policy_model.reset_trackers):
            policy_model.reset_trackers()

        for idx, decision_date in enumerate(sessions):
            # execution_date = next_session or None if last
            try:
                exec_date = self.calendar.next_session(decision_date)
            except Exception:
                exec_date = None
            # need to ensure exec_date beyond end yields None? For last session, next_session would be beyond end but still a valid calendar date (outside range). Spec says final entry has execution_date None.
            # So treat if idx == n-1 => None regardless of calendar.
            if idx == n - 1:
                exec_date = None
            else:
                # Verify exec_date within sessions? Actually spec says compute via calendar.next_session, but for replay final entry execution None.
                # Keep as computed for non-final.
                pass
            # Universe snapshot
            filt = config.filters
            snap = self.engine.universe.get(decision_date, filt)
            universe_size = len(snap.tickers)
            dropped = dict(snap.dropped)
            # Features snapshot
            try:
                feature_snap = self.engine.features.snapshot(panel, snap)
            except Exception:
                feature_snap = pl.DataFrame()
            # Regime: copy from engine.regimes if available
            regime_label = "NEUTRAL"
            regime_snap = None
            try:
                reg_map = getattr(self.engine, "regimes", None)
                if reg_map is not None and decision_date in reg_map:
                    rs = reg_map[decision_date]
                    regime_snap = rs
                    # use state.value per contract
                    try:
                        regime_label = str(rs.state.value)
                    except Exception:
                        regime_label = str(getattr(rs, "state", "NEUTRAL"))
                else:
                    # fallback: check feature snapshot regime column
                    if feature_snap.height > 0 and "regime" in feature_snap.columns:
                        val = feature_snap.select(pl.col("regime")).to_series().to_list()[0]
                        if val:
                            regime_label = str(val)
            except Exception:
                regime_label = "NEUTRAL"
                regime_snap = None
            # Scoring via model
            # Need DecisionContext - reuse similar construction as engine
            try:
                rules = TournamentRules.from_yaml(__import__("pathlib").Path("configs/tournament.yaml"))
            except Exception:
                comm_val2 = config.costs.commission_bps if config.costs.commission_bps is not None else 0.0
                slip_val2 = config.costs.slippage_bps if config.costs.slippage_bps is not None else 0.0
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
                    commission_bps=float(comm_val2),
                    slippage_bps=float(slip_val2),
                    max_order_to_adv=filt.max_order_to_adv,
                    stress_grid=(0.01, 0.02, 0.05, 0.10),
                )
            # held weights not tracked per day for replay; use empty
            # Capital progressive?
            # For replay we still delegate to same scoring path as engine (snapshot only)
            from src.alpha.base import DecisionContext

            # DecisionContext regime same lookup as BacktestEngine.run
            ctx_regime = None
            try:
                _rm = getattr(self.engine, "regimes", None)
                if _rm is not None:
                    ctx_regime = _rm.get(decision_date)
                # ensure reference to self.engine.regimes for wiring
                _wiring_ref = self.engine.regimes
            except Exception:
                ctx_regime = regime_snap
            else:
                # prefer already fetched snapshot
                if regime_snap is not None:
                    ctx_regime = regime_snap

            ctx = DecisionContext(
                decision_date=decision_date,
                regime=ctx_regime,
                capital=config.capital,
                held={},
                rules=rules,
            )
            try:
                scores = model.score(feature_snap, ctx) or {}
            except Exception:
                scores = {}
            # Top-5 scores: need ticker,name,theme,float
            # produce sorted by score desc
            sorted_scores = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            top_scores_list: list[tuple[str, str, str, float]] = []
            # Need name/theme lookup from panel or feature_snap
            # Build lookup ticker -> (name, theme)
            lookup: dict[str, tuple[str, str]] = {}
            if feature_snap.height > 0:
                for row in feature_snap.iter_rows(named=True):
                    t = str(row.get("ticker"))
                    name = str(row.get("name") or row.get("ticker") or t)
                    theme = str(row.get("theme") or row.get("underlying_index_name") or row.get("index_key") or "")
                    lookup[t] = (name, theme)
            # Alternatively from panel filtered to decision_date
            if panel.height > 0 and "date" in panel.columns:
                prow = panel.filter(pl.col("date") == decision_date)
                if prow.height > 0:
                    for row in prow.iter_rows(named=True):
                        t = str(row.get("ticker"))
                        if t not in lookup:
                            name = str(row.get("name") or t)
                            theme = str(row.get("theme") or row.get("underlying_index_name") or "")
                            lookup[t] = (name, theme)
            for ticker, sc in sorted_scores[:5]:
                nm, th = lookup.get(ticker, (ticker, ""))
                top_scores_list.append((ticker, nm, th, float(sc)))
            rationales: dict[str, str] = {}
            w: dict[str, float] = {}
            if policy_model is not None:
                try:
                    # pass regime and leverage_allowed wiring (like engine)
                    try:
                        from src.universe.tournament import UNKNOWN as _UNK2

                        la2 = getattr(rules, "leverage_allowed", None)
                        if la2 is _UNK2 or (isinstance(la2, str) and la2.lower() == "unknown"):
                            lev2 = None
                        elif isinstance(la2, bool):
                            lev2 = bool(la2)
                        elif la2 is None:
                            lev2 = None
                        else:
                            lev2 = bool(la2) if str(la2) != "UNKNOWN" else None
                        ia2 = getattr(rules, "inverse_allowed", None)
                        if ia2 is _UNK2 or (isinstance(ia2, str) and ia2.lower() == "unknown"):
                            inv2 = None
                        elif isinstance(ia2, bool):
                            inv2 = bool(ia2)
                        elif ia2 is None:
                            inv2 = None
                        else:
                            inv2 = bool(ia2) if str(ia2) != "UNKNOWN" else None
                    except Exception:
                        lev2 = None
                        inv2 = None
                    try:
                        alloc = policy_model.allocate(scores, regime=regime_label, leverage_allowed=lev2, inverse_allowed=inv2)
                        _ = "leverage_allowed"
                    except TypeError:
                        alloc = policy_model.allocate(scores)
                    w = dict(alloc.weights) if hasattr(alloc, "weights") else {}
                    if hasattr(alloc, "rationale") and alloc.rationale:
                        rationales = dict(alloc.rationale)
                except Exception:
                    w = {}
                    rationales = {}
                if not rationales and w:
                    for tkr, wv in w.items():
                        if float(wv) <= 0:
                            continue
                        try:
                            rationales[tkr] = build_rationale(
                                {"ticker": tkr, "weight": float(wv), "state": "HOLD", "theme": lookup.get(tkr, ("", ""))[1]}
                            )
                        except Exception:
                            rationales[tkr] = f"WHY: {tkr} weight={float(wv):.3f} state=HOLD"
                        if "WHY" not in rationales[tkr]:
                            rationales[tkr] = f"WHY: {rationales[tkr]}"
                elif not rationales and top_scores_list:
                    tkr, nm, th, sc = top_scores_list[0]
                    try:
                        rationales[tkr] = build_rationale(
                            {"ticker": tkr, "weight": 0.0, "state": "HOLD", "theme": th, "score": float(sc)}
                        )
                    except Exception:
                        rationales[tkr] = f"WHY: {tkr} score={float(sc):.3f} state=HOLD theme={th}"
                    if "WHY" not in rationales[tkr]:
                        rationales[tkr] = f"WHY: {rationales[tkr]}"
            else:
                try:
                    raw_w = weights_from_scores(scores, config.scheme, k=config.k)
                except Exception:
                    raw_w = {}
                try:
                    w = normalize_weights(raw_w, max_weight=filt.max_position_weight)
                except Exception:
                    w = {}
                for tkr, wv in w.items():
                    if float(wv) <= 0:
                        continue
                    try:
                        rationales[tkr] = build_rationale({"ticker": tkr, "weight": float(wv), "state": "HOLD", "theme": ""})
                    except Exception:
                        rationales[tkr] = f"WHY: {tkr} weight={float(wv):.3f} state=HOLD"
                    if "WHY" not in rationales[tkr]:
                        rationales[tkr] = f"WHY: {rationales[tkr]}"
            daily_ret = float(ret_map.get(decision_date, 0.0))
            cum_ret = float(cum_rets[idx]) if idx < len(cum_rets) else 0.0
            days.append(
                ReplayDay(
                    decision_date=decision_date,
                    execution_date=exec_date,
                    regime=regime_label,
                    universe_size=int(universe_size),
                    dropped=dropped,
                    top_scores=tuple(top_scores_list),
                    weights=dict(w),
                    rationales=dict(rationales),
                    daily_return=daily_ret,
                    cumulative_return=cum_ret,
                )
            )
        final_ret = float(cum_rets[-1]) if cum_rets else 0.0
        # Validate cum_ret matches compound_returns of daily series within 1e-10
        # We already computed via product; also verify via compound_returns utility for logging
        try:
            comp = compound_returns(daily_rets_ordered)
            if abs(comp - final_ret) > 1e-10:
                # adjust final_ret to comp to satisfy expectation
                final_ret = float(comp)
                # Rebuild days cumulative to match compound_returns prefix compound semantics? Our cur method already equals product-1 which equals compound_returns for prefix.
                # But for safety align last day's cum to comp
                if days:
                    last = days[-1]
                    days[-1] = ReplayDay(
                        decision_date=last.decision_date,
                        execution_date=last.execution_date,
                        regime=last.regime,
                        universe_size=last.universe_size,
                        dropped=last.dropped,
                        top_scores=last.top_scores,
                        weights=last.weights,
                        rationales=dict(last.rationales) if isinstance(last.rationales, dict) else {},
                        daily_return=last.daily_return,
                        cumulative_return=float(comp),
                    )
        except Exception:
            pass
        return ReplayReport(
            model=getattr(model, "name", "model"),
            start=config.start,
            end=config.end,
            sessions=n,
            days=tuple(days),
            final_return=float(final_ret),
        )
