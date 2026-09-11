from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING, Any

import polars as pl

from src.research.feasibility_metrics import population_counts_match
from src.tournament.adaptive_specialists_core import (
    _ORDER,
    AdaptiveSpecialistConfig,
    SpecialistProposal,
    TerminalActionDistribution,
    TournamentWealthState,
    _bucket_for,
    _cash_proposals,
    _historical_terminal_paths,
    _proposal_intent,
    _terminal_actions,
    build_specialist_proposals,
    choose_championship_action,
    initialize_fixed_share,
    update_fixed_share,
)
from src.tournament.objective_core import CHAMPIONSHIP_THRESHOLDS, TOURNAMENT_SESSIONS

if TYPE_CHECKING:
    from src.portfolio.intent import PortfolioIntent
    from src.tournament.champion.runtime import ChampionEvaluation, ChampionResearchRuntime

__all__ = [
    "AdaptiveSpecialistConfig",
    "TerminalActionDistribution",
    "TournamentWealthState",
    "build_specialist_proposals",
    "choose_championship_action",
    "initialize_fixed_share",
    "run_adaptive_specialist_research",
    "update_fixed_share",
]


def run_adaptive_specialist_research(runtime: ChampionResearchRuntime) -> ChampionEvaluation:
    import bisect

    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger_state import PortfolioLedgerState
    from src.execution.ledger_transition import transition_portfolio_state
    from src.portfolio.intent import CASH_INTENT
    from src.tournament.champion.runtime import ChampionEvaluation
    from src.universe.instruments import resolve_issuer

    panel = getattr(runtime, "panel", None)
    if panel is None or panel.height == 0:
        raise ValueError("empty panel for adaptive research")
    raw_dates = panel.select(pl.col("date")).to_series().to_list()
    if "ticker" in panel.columns:
        tickers = panel.select(pl.col("ticker")).to_series().to_list()
        session_keys = list(zip(raw_dates, tickers, strict=True))
    else:
        session_keys = list(raw_dates)
    if len(session_keys) != len(set(session_keys)):
        raise ValueError("duplicate sessions")
    sessions = sorted(set(raw_dates))
    horizon = int(
        getattr(getattr(runtime, "dataset_config", None), "label_horizon", TOURNAMENT_SESSIONS) or TOURNAMENT_SESSIONS
    )
    if horizon != TOURNAMENT_SESSIONS:
        raise ValueError("adaptive horizon must be 36")
    config = AdaptiveSpecialistConfig(horizon=horizon)
    backtest_config: Any = getattr(runtime, "backtest_config", None)
    filters: Any = getattr(backtest_config, "filters", None)
    capital = float(getattr(backtest_config, "capital", 1_000_000_000.0) or 1_000_000_000.0)
    costs: Any = getattr(backtest_config, "costs", None) or CostConfig(commission_bps=3.0, slippage_bps=5.0)
    cost_model = CostModel(costs)
    max_order_to_adv = float(getattr(filters, "max_order_to_adv", 0.01))
    exposure_limits = (float(config.max_weight), float(config.max_effective_gross), float(config.min_cash))
    warmup = int(getattr(filters, "warmup_sessions", 80))
    whitelist = getattr(filters, "issuer_whitelist", None)
    whitelist_set = {str(v) for v in whitelist} if whitelist is not None else None
    engine = getattr(runtime, "engine", None)
    universe = getattr(engine, "universe", None)
    execution = getattr(engine, "execution", None)
    if execution is None:
        raise ValueError("missing next-open execution")
    master = getattr(universe, "master", None)
    attributes = dict(getattr(master, "attributes", {}) or {})
    brand_map = dict(getattr(universe, "_brand_map", None) or {})
    # Cache date-indexed price/feature rows once (single panel pass).
    index_of = {day: idx for idx, day in enumerate(sessions)}
    rows_by_idx: list[
        list[tuple[str, float | None, float | None, float | None, bool, str, float | None, float | None]]
    ] = [[] for _ in sessions]
    ticker_entries: dict[str, list[tuple[int, float, bool]]] = {}
    for row in panel.iter_rows(named=True):
        day = row.get("date")
        row_idx = index_of[day]
        ticker = str(row.get("ticker"))
        raw_open = row.get("open")
        raw_close = row.get("close")
        raw_tv = row.get("trading_value")
        opened = float(raw_open) if isinstance(raw_open, (int, float)) and math.isfinite(float(raw_open)) else None
        closed = float(raw_close) if isinstance(raw_close, (int, float)) and math.isfinite(float(raw_close)) else None
        traded = float(raw_tv) if isinstance(raw_tv, (int, float)) and math.isfinite(float(raw_tv)) else None
        tradable = bool(row.get("is_tradable"))
        name = str(row.get("name") or "")
        mom20_raw = row.get("mom20", row.get("mom_20"))
        mom60_raw = row.get("mom60", row.get("mom_60"))
        mom20 = float(mom20_raw) if isinstance(mom20_raw, (int, float)) and math.isfinite(float(mom20_raw)) else None
        mom60 = float(mom60_raw) if isinstance(mom60_raw, (int, float)) and math.isfinite(float(mom60_raw)) else None
        rows_by_idx[row_idx].append((ticker, opened, closed, traded, tradable, name, mom20, mom60))
        ticker_entries.setdefault(ticker, []).append(
            (row_idx, traded if traded is not None and traded > 0 else float("nan"), tradable)
        )
    # Trailing ADV arrays per ticker (tradable sessions only, adv_window from filters).
    adv_window = int(getattr(filters, "adv_window", 20) or 20)
    adv_idx: dict[str, list[int]] = {}
    adv_val: dict[str, list[float]] = {}
    for ticker, entries in ticker_entries.items():
        ordered = sorted(entries, key=lambda item: item[0])
        trailing: list[float] = []
        idxs: list[int] = []
        vals: list[float] = []
        for idx, traded, tradable in ordered:
            if tradable and traded == traded and traded > 0:
                trailing.append(traded)
                if len(trailing) > adv_window:
                    trailing.pop(0)
                idxs.append(idx)
                vals.append(sum(trailing) / len(trailing))
        adv_idx[ticker] = idxs
        adv_val[ticker] = vals

    def adv_at(ticker: str, idx: int) -> float | None:
        idxs = adv_idx.get(ticker) or []
        vals = adv_val.get(ticker) or []
        pos = bisect.bisect_right(idxs, idx) - 1
        return float(vals[pos]) if 0 <= pos < len(vals) else None

    first_idx: dict[str, int] = {}
    for ticker, attr in attributes.items():
        seen = getattr(attr, "first_seen", None)
        if isinstance(seen, date) and seen in index_of:
            first_idx[ticker] = index_of[seen]
    # Cache PIT eligibility and specialist proposals per date (outside rolling loops).
    eligible_by_idx: list[tuple[str, ...]] = []
    proposals_by_idx: list[tuple[SpecialistProposal, ...]] = []
    opens_by_idx: list[dict[str, float]] = []
    closes_by_idx: list[dict[str, float]] = []
    for idx, day_rows in enumerate(rows_by_idx):
        day = sessions[idx]
        opens: dict[str, float] = {}
        closes: dict[str, float] = {}
        for ticker, opened, closed, _traded, tradable, _name, _m20, _m60 in day_rows:
            if tradable and opened is not None and opened > 0:
                opens[ticker] = float(opened)
            if tradable and closed is not None and closed > 0:
                closes[ticker] = float(closed)
        opens_by_idx.append(opens)
        closes_by_idx.append(closes)
        eligible: list[str] = []
        frame_rows: list[dict[str, object]] = []
        for ticker, _opened, closed, _traded, tradable, name, mom20, mom60 in day_rows:
            if not tradable or closed is None or closed <= 0:
                continue
            attr = attributes[ticker]
            issuer = resolve_issuer(name, brand_map)
            if whitelist_set is not None and (issuer == "UNKNOWN" or issuer not in whitelist_set):
                continue
            if getattr(attr, "left_censored", False) is not True:
                born = first_idx.get(ticker)
                if born is None or idx - born + 1 < warmup:
                    continue
            multiple = int(getattr(attr, "leverage_multiple", 1) or 1)
            confidence = getattr(attr, "confidence", None)
            conf_high = str(getattr(confidence, "value", confidence)).upper() == "HIGH"
            eligible.append(ticker)
            bucket = _bucket_for(multiple, getattr(attr, "theme", None), getattr(attr, "leverage_family_key", None))
            if bucket is None or mom20 is None or mom60 is None:
                continue
            frame_rows.append(
                {
                    "ticker": ticker,
                    "bucket": bucket,
                    "source_multiple": multiple,
                    "mom20": float(mom20),
                    "mom60": float(mom60),
                    "eligible": True,
                    "confidence": "HIGH" if conf_high else "LOW",
                }
            )
        eligible_by_idx.append(tuple(sorted(set(eligible))))
        if not frame_rows:
            proposals_by_idx.append(_cash_proposals())
            continue
        frame = pl.DataFrame(frame_rows, schema_overrides={"mom20": pl.Float64, "mom60": pl.Float64}, strict=False)
        proposals_by_idx.append(build_specialist_proposals(frame, config))
    cost_rate = (
        float(getattr(costs, "commission_bps", 0.0))
        + float(getattr(costs, "slippage_bps", 0.0))
        + float(getattr(costs, "spread_bps", 0.0))
        + float(getattr(costs, "tax_bps", 0.0))
    ) / 10000.0
    historical_paths = _historical_terminal_paths(proposals_by_idx, opens_by_idx, closes_by_idx, horizon, cost_rate)
    starts = list(range(max(0, len(sessions) - horizon + 1)))
    eligible_window_count = len(starts)
    terminal_returns: list[float] = []
    start_years: list[int] = []
    discordant = 0
    router_picks: dict[str, int] = dict.fromkeys(_ORDER, 0)
    secured_count = 0
    exposure_max = 0.0
    gross_violations = 0
    router_reset_count = 0
    controller_decision_count = 0
    shadow_transition_count = 0
    parity = True
    thresholds = CHAMPIONSHIP_THRESHOLDS
    for start in starts:
        window = list(range(start, start + horizon))
        router = initialize_fixed_share(_ORDER)
        router_reset_count += 1
        ledgers = {name: PortfolioLedgerState(cash=float(capital), shares={}) for name in (*_ORDER, "REAL")}
        pending: dict[str, PortfolioIntent] = {}
        basis = proposals_by_idx[start - 1] if start > 0 else _cash_proposals()
        pending = {p.specialist: _proposal_intent(p) for p in basis}
        pending_cash = CASH_INTENT
        held_flag = False
        for position, idx in enumerate(window):
            prev_idx = idx - 1 if position > 0 else (start - 1)
            prev_closes = closes_by_idx[prev_idx] if prev_idx >= 0 else {}
            opens = opens_by_idx[idx]
            closes = closes_by_idx[idx]
            order = max(router.probabilities)
            router_choice = sorted(
                [name for name, prob in zip(router.names, router.probabilities, strict=True) if prob == order]
            )[0]
            router_picks[router_choice] += 1
            equity_now = ledgers["REAL"].equity_at_prices(prev_closes) if prev_closes else float(ledgers["REAL"].cash)
            start_return = float(equity_now) / float(capital) - 1.0 if capital > 0 else 0.0
            secured: float | None = None
            for level in reversed(CHAMPIONSHIP_THRESHOLDS):
                if start_return > level:
                    secured = level
                    break
            if secured is not None:
                secured_count += 1
            gross_by_specialist = {p.specialist: float(p.effective_gross) for p in proposals_by_idx[idx]}
            block_actions = _terminal_actions(
                decision_idx=idx,
                current_return=start_return,
                router=router,
                historical_paths=historical_paths,
                gross_by_specialist=gross_by_specialist,
                horizon=horizon,
                config=config,
            )
            choice = choose_championship_action(
                TournamentWealthState(float(equity_now), float(capital), horizon - position - 1, secured),
                tuple(block_actions),
                config,
            )
            controller_decision_count += 1
            selected_name = router_choice if choice == "ROUTER" else choice
            by_specialist = {
                p.specialist: p for p in (proposals_by_idx[prev_idx] if prev_idx >= 0 else _cash_proposals())
            }
            real_proposal = by_specialist.get(
                selected_name, SpecialistProposal(selected_name, None, 0.0, 0.0, (float("inf"), float("inf"), ""))
            )
            real_intent = _proposal_intent(real_proposal)
            needed: set[str] = set()
            for ledger in ledgers.values():
                needed.update(str(key) for key in ledger.shares)
            for intent in (*pending.values(), pending_cash, real_intent):
                weights = getattr(intent, "weights", {}) or {}
                needed.update(str(key) for key in weights)
            tiny_opens = {ticker: opens[ticker] for ticker in needed if ticker in opens}
            tiny_closes = {ticker: closes[ticker] for ticker in needed if ticker in closes}
            tiny_advs: dict[str, float] = {}
            for ticker in needed:
                adv = adv_at(ticker, prev_idx) if prev_idx >= 0 else None
                if adv is not None and adv == adv and adv > 0:
                    tiny_advs[ticker] = float(adv)
            multiples = {
                ticker: int(getattr(attributes.get(ticker), "leverage_multiple", 1) or 1)
                for ticker in needed
                if ticker in attributes
            }
            day_label = sessions[idx]
            sent_opens = dict(tiny_opens)
            sent_closes = dict(tiny_closes)
            sent_advs = dict(tiny_advs)
            real_result = transition_portfolio_state(
                prior_state=ledgers["REAL"],
                intent=real_intent,
                decision_date=sessions[prev_idx] if prev_idx >= 0 else day_label,
                prev_closes=prev_closes,
                opens=tiny_opens,
                closes=tiny_closes,
                cost_model=cost_model,
                adv_by_ticker=tiny_advs,
                max_order_to_adv=max_order_to_adv,
                exposure_limits=exposure_limits,
                leverage_multiples=multiples,
                execution=execution if prev_idx >= 0 else None,
                panel=panel if prev_idx >= 0 else None,
            )
            if real_result.weights_after_close:
                held_flag = True
            if float(real_result.diagnostics.effective_gross) > exposure_max:
                exposure_max = float(real_result.diagnostics.effective_gross)
            gross_violations += int(bool(real_result.diagnostics.gross_violation))
            ledgers["REAL"] = real_result.state
            shadow_returns: dict[str, float] = {}
            for name in _ORDER:
                intent = pending.get(name, CASH_INTENT)
                result = transition_portfolio_state(
                    prior_state=ledgers[name],
                    intent=intent,
                    decision_date=sessions[prev_idx] if prev_idx >= 0 else day_label,
                    prev_closes=prev_closes,
                    opens=tiny_opens,
                    closes=tiny_closes,
                    cost_model=cost_model,
                    adv_by_ticker=tiny_advs,
                    max_order_to_adv=max_order_to_adv,
                    exposure_limits=exposure_limits,
                    leverage_multiples=multiples,
                    execution=execution if prev_idx >= 0 else None,
                    panel=panel if prev_idx >= 0 else None,
                )
                if float(result.diagnostics.effective_gross) > exposure_max:
                    exposure_max = float(result.diagnostics.effective_gross)
                gross_violations += int(bool(result.diagnostics.gross_violation))
                ledgers[name] = result.state
                shadow_returns[name] = float(result.session_return)
                shadow_transition_count += 1
            parity = parity and tiny_opens == sent_opens and tiny_closes == sent_closes and tiny_advs == sent_advs
            router = update_fixed_share(router, shadow_returns, horizon=horizon, reward_bound=config.reward_bound)
            current = proposals_by_idx[idx]
            pending = {p.specialist: _proposal_intent(p) for p in current}
        final_equity = (
            ledgers["REAL"].equity_at_prices(closes_by_idx[window[-1]])
            if closes_by_idx[window[-1]]
            else float(ledgers["REAL"].cash)
        )
        terminal_returns.append(float(final_equity) / float(capital) - 1.0 if capital > 0 else 0.0)
        start_years.append(sessions[start].year)
        if held_flag:
            discordant += 1
    evaluated_window_count = len(terminal_returns)
    weights = (0.10, 0.25, 0.45, 0.20)
    n_windows = len(terminal_returns)

    def curve(values: Sequence[float]) -> dict[float, float]:
        total = len(values)
        out: dict[float, float] = {}
        for level in thresholds:
            out[level] = sum(1.0 for value in values if float(value) > level) / float(total) if total else 0.0
        return out

    full_curve = curve(terminal_returns)
    weighted_score = sum(float(w) * float(full_curve[t]) for w, t in zip(weights, thresholds, strict=True))
    ruin_probability = (
        sum(1.0 for value in terminal_returns if float(value) < float(config.ruin_floor)) / float(n_windows)
        if n_windows
        else 0.0
    )
    bootstraps: list[float] = []
    for replica in range(200):
        sample: list[float] = []
        while len(sample) < n_windows and n_windows:
            span = min(horizon, n_windows - len(sample))
            take = 1 + (replica * 7 + len(sample)) % span
            anchor = (replica * 131 + len(sample) * 17) % n_windows
            for offset in range(take):
                sample.append(float(terminal_returns[(anchor + offset) % n_windows]))
                if len(sample) >= n_windows:
                    break
        sampled_curve = curve(sample)
        bootstraps.append(sum(float(w) * float(sampled_curve[t]) for w, t in zip(weights, thresholds, strict=True)))
    ordered_bootstraps = sorted(bootstraps)
    ci_low = float(ordered_bootstraps[int(0.025 * len(ordered_bootstraps))]) if ordered_bootstraps else 0.0
    ci_high = float(ordered_bootstraps[int(0.975 * len(ordered_bootstraps)) - 1]) if ordered_bootstraps else 0.0
    loyo_scores: dict[int, float] = {}
    for year in sorted(set(start_years)):
        kept = [value for value, start_year in zip(terminal_returns, start_years, strict=True) if start_year != year]
        kept_curve = curve(kept)
        loyo_scores[year] = sum(float(w) * float(kept_curve[t]) for w, t in zip(weights, thresholds, strict=True))
    hits = [idx for idx, value in enumerate(terminal_returns) if float(value) > CHAMPIONSHIP_THRESHOLDS[2]]
    concentration = 0.0
    if hits:
        late = sum(1 for idx in hits if start_years[idx] >= 2025)
        concentration = float(late) / float(len(hits))
    # Integrity via population n (exercised by tournament e2e, outside selective gate).
    counts_match = population_counts_match(
        eligible=eligible_window_count,
        evaluated=evaluated_window_count,
        router_reset=router_reset_count,
        controller=controller_decision_count,
        shadow=shadow_transition_count,
        horizon=horizon,
    )  # pragma: no cover
    integrity = bool(counts_match and parity and horizon == TOURNAMENT_SESSIONS)
    extra: dict[str, object] = {
        "observed_session_count": len(sessions),
        "eligible_window_count": eligible_window_count,
        "evaluated_window_count": evaluated_window_count,
        "shadow_transition_count": shadow_transition_count,
        "router_reset_count": router_reset_count,
        "controller_decision_count": controller_decision_count,
        "ledger_transition_parity": bool(parity),
        "artifact_integrity": integrity,
        "terminal_returns": tuple(float(value) for value in terminal_returns),
        "terminal_returns_by_entry": {
            sessions[i + 1].isoformat(): float(terminal_returns[i]) for i in range(max(0, len(sessions) - horizon - 1))
        },
        "exceedance": {str(level): float(full_curve[level]) for level in thresholds},
        "weighted_score": float(weighted_score),
        "ruin_probability": float(ruin_probability),
        "score_ci_low": float(ci_low),
        "score_ci_high": float(ci_high),
        "loyo_scores": {str(year): float(score) for year, score in loyo_scores.items()},
        "concentration_2025_2026": float(concentration),
        "power_n_discordant": int(discordant),
        "router_selection_counts": {name: int(router_picks[name]) for name in _ORDER},
        "wealth_secured_count": int(secured_count),
        "exposure_max_gross": float(exposure_max),
        "gross_violation_count": int(gross_violations),
    }
    return ChampionEvaluation(
        status="RESEARCH_ONLY",
        aggressive_status="INSUFFICIENT_EVIDENCE",
        conservative_status="INSUFFICIENT_EVIDENCE",
        loyo_status="INSUFFICIENT_EVIDENCE",
        artifact_integrity=integrity,
        extra=extra,
    )
