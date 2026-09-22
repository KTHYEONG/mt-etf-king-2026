"""Live decision pipeline shared with backtest (P27 fix)."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import polars as pl

from src.alpha.base import DecisionContext
from src.portfolio.intent import PortfolioIntent, resolve_portfolio_intent
from src.portfolio.sizing import SizingScheme, weights_from_scores
from src.tournament.championship_regime import (
    ChampionshipSleeve,
    classify_championship_sleeve_from_maps,
    kospi_sleeve_feature_maps,
    select_kospi_headline_series,
)


class StaleSleeveInputError(RuntimeError):
    """Raised when the sleeve index input is missing/stale at decision_date."""


class StalePanelInputError(RuntimeError):
    """Raised when the ETF panel input is missing/stale at decision_date."""

    pass


class StateDiscontinuityError(RuntimeError):
    """Raised when the live sticky ledger cannot supply the exact prior-session entry.

    The post-crash anchor latch makes the prior holding decision-critical: treating an
    unknown prior as CASH turns an operational gap into a SELL_ALL / SWITCH signal.
    Callers must abort the decision (NO_TRADE) and require either a catch-up run or an
    explicit ``--held`` override.
    """


logger = logging.getLogger(__name__)

# P27 adopted run's actual ADV participation cap (0.01) supplied via the
# backtest CLI's --participation flag (verified in the adopted run's own
# meta.json: "participation": 0.01). There is no single config file owning
# this value for a specific adopted run, so it is hardcoded here with provenance.
P27_ADOPTED_MAX_ORDER_TO_ADV: Final[float] = 0.01

# 라이브 스티키 원장 보관 한도: 대회 36세션 + 캐치업 여유분까지 한 번에 담을 수 있는 60세션.
STICKY_STATE_HISTORY_MAX: Final[int] = 60


def assert_panel_input_fresh(panel: pl.DataFrame, *, decision_date: date) -> None:
    if not isinstance(panel, pl.DataFrame):
        raise StalePanelInputError(f"panel inputs stale at {decision_date.isoformat()}: not a DataFrame")
    if panel.height == 0:
        raise StalePanelInputError(f"panel inputs stale at {decision_date.isoformat()}: empty panel")
    if "date" not in panel.columns:
        raise StalePanelInputError(f"panel inputs stale at {decision_date.isoformat()}: missing date column")
    if panel.filter(pl.col("date") == decision_date).height == 0:
        raise StalePanelInputError(f"panel inputs stale at {decision_date.isoformat()}: no row at decision_date")
    return None


def resolve_prior_trading_session(panel: pl.DataFrame, *, decision_date: date) -> date | None:
    if not isinstance(panel, pl.DataFrame):
        return None
    if panel.height == 0:
        return None
    if "date" not in panel.columns:
        return None
    panel_min = panel.get_column("date").min()
    if not isinstance(panel_min, date):
        return None
    if panel_min > decision_date:
        return None
    from src.backtest.session_grid import resolve_session_grid
    from src.core.calendar import get_calendar

    calendar_sessions = get_calendar().sessions(panel_min, decision_date)
    sessions = list(resolve_session_grid(calendar_sessions, panel).sessions)
    prior = [s for s in sessions if s < decision_date]
    if not prior:
        return None
    return max(prior)


def _parse_and_validate_sticky_entry(entry: object) -> tuple[str | None, float, int] | None:
    if not isinstance(entry, dict):
        return None
    held_raw = entry.get("held")
    weight_raw = entry.get("held_weight")
    hold_len_raw = entry.get("hold_len")
    if held_raw is None:
        return (None, 0.0, 0)
    if not isinstance(held_raw, str):
        return None
    if isinstance(weight_raw, bool):
        return None
    try:
        weight = float(weight_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(weight):
        return None
    if isinstance(hold_len_raw, bool):
        return None
    if isinstance(hold_len_raw, float) and not hold_len_raw.is_integer():
        return None
    try:
        hold_len = int(hold_len_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if hold_len < 0:
        return None
    return (held_raw, float(weight), int(hold_len))


def _parse_session_date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _ledger_latest_session(data: Mapping[str, object]) -> date | None:
    history = data.get("history")
    keys: list[object] = list(history) if isinstance(history, dict) else []
    keys.append(data.get("as_of"))
    parsed = [d for d in map(_parse_session_date, keys) if d is not None]
    return max(parsed) if parsed else None


def _state_discontinuity(
    state_path: Path,
    prior_session: date,
    data: Mapping[str, object] | None,
    reason: str,
) -> StateDiscontinuityError:
    latest = _ledger_latest_session(data) if data is not None else None
    latest_label = latest.isoformat() if latest is not None else "none"
    return StateDiscontinuityError(
        f"{reason}: state_path={state_path} prior_session={prior_session.isoformat()} latest_session={latest_label}"
    )


def resolve_prior_sticky_state(
    state_path: Path,
    *,
    prior_session: date | None,
) -> tuple[str | None, float, int]:
    """Return the ledger entry ``(held, held_weight, hold_len)`` recorded for ``prior_session``.

    Args:
        state_path: Live sticky ledger JSON (``data/state/<strategy>_position.json``).
        prior_session: Session immediately before the decision date on the panel session
            grid, or ``None`` when the panel has no earlier session.

    Returns:
        The recorded entry; ``held is None`` means the strategy was genuinely in CASH.
        ``(None, 0.0, 0)`` only when ``prior_session`` is ``None``.

    Raises:
        StateDiscontinuityError: The ledger file is missing, unreadable, not a JSON object,
            has no valid entry for ``prior_session``, or that entry fails validation.
    """
    if prior_session is None:
        return (None, 0.0, 0)

    p = Path(state_path)
    try:
        parsed = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _state_discontinuity(p, prior_session, None, f"sticky ledger unreadable ({exc!r})") from exc
    if not isinstance(parsed, dict):
        raise _state_discontinuity(p, prior_session, None, "sticky ledger is not a JSON object")

    prior_key = prior_session.isoformat()
    history = parsed.get("history")
    if isinstance(history, dict) and prior_key in history:
        entry = _parse_and_validate_sticky_entry(history[prior_key])
        if entry is not None:
            return entry
        raise _state_discontinuity(p, prior_session, parsed, "sticky ledger entry invalid")

    if _parse_session_date(parsed.get("as_of")) == prior_session:
        entry = _parse_and_validate_sticky_entry(parsed)
        if entry is not None:
            return entry
        raise _state_discontinuity(p, prior_session, parsed, "sticky ledger entry invalid")

    raise _state_discontinuity(p, prior_session, parsed, "sticky ledger has no entry for prior session")


def persist_sticky_state(
    state_path: Path,
    *,
    decision_date: date,
    held: str | None,
    held_weight: float,
    hold_len: int,
) -> None:
    """Record the decision for ``decision_date`` in the monotone live sticky ledger.

    The ledger keeps one entry per decided session in ``history``. The top-level
    ``as_of/held/held_weight/hold_len`` mirror the entry of the latest recorded session,
    so re-deciding an older session (manual rerun or catch-up) updates only that
    session's entry and never rewinds the ledger.
    """
    p = Path(state_path)
    history: dict[str, Any] = {}
    if p.exists():
        try:
            old_data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(old_data, dict):
                old_hist = old_data.get("history")
                old_as_of = old_data.get("as_of")
                if isinstance(old_hist, dict):
                    history = dict(old_hist)
                elif isinstance(old_as_of, str):
                    history[old_as_of] = {
                        "held": old_data.get("held"),
                        "held_weight": float(old_data.get("held_weight", 0.0)),
                        "hold_len": int(old_data.get("hold_len", 0)),
                    }
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"[SYS] persist_sticky_state unreadable path={p} error={exc!r}")
            history = {}

    history[decision_date.isoformat()] = {
        "held": held,
        "held_weight": float(held_weight),
        "hold_len": int(hold_len),
    }

    if len(history) > STICKY_STATE_HISTORY_MAX:
        for key in sorted(history)[: len(history) - STICKY_STATE_HISTORY_MAX]:
            del history[key]

    latest_key = max(history)
    latest_entry = history[latest_key]
    payload: dict[str, Any] = {
        "as_of": latest_key,
        "held": latest_entry["held"],
        "held_weight": float(latest_entry["held_weight"]),
        "hold_len": int(latest_entry["hold_len"]),
        "history": history,
    }
    try:
        if p.parent != Path():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"[SYS] persist_sticky_state failed path={p} error={exc!r}")
        return None
    return None


def sticky_state_latest_session(state_path: Path) -> date | None:
    """Return the latest session recorded in the live sticky ledger.

    Returns:
        The maximum ``history`` key, or the legacy top-level ``as_of`` when no history
        exists; ``None`` when the ledger file does not exist.

    Raises:
        StateDiscontinuityError: The file exists but is unreadable, not a JSON object, or
            carries no parseable session date.
    """
    p = Path(state_path)
    try:
        parsed = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if not p.exists():
            return None
        raise StateDiscontinuityError(f"sticky ledger unreadable: state_path={p} error={exc!r}") from exc
    if not isinstance(parsed, dict):
        raise StateDiscontinuityError(f"sticky ledger is not a JSON object: state_path={p}")
    latest = _ledger_latest_session(parsed)
    if latest is None:
        raise StateDiscontinuityError(f"sticky ledger carries no parseable session: state_path={p}")
    return latest


def pending_catchup_sessions(
    state_path: Path,
    panel: pl.DataFrame,
    *,
    target_session: date,
) -> list[date]:
    """List sessions that must be decided, in order, before ``target_session``.

    Sessions come from the same calendar/panel session grid used by
    ``resolve_prior_trading_session`` so each catch-up decision's prior session is exactly
    the previously decided one.

    Returns:
        Ascending sessions ``s`` with ``latest < s < target_session``; empty when the ledger
        does not exist or is already at or beyond ``target_session``.

    Raises:
        StateDiscontinuityError: Propagated from ``sticky_state_latest_session``.
    """
    from src.backtest.session_grid import resolve_session_grid
    from src.core.calendar import get_calendar

    latest = sticky_state_latest_session(state_path)
    panel_min = panel.get_column("date").min() if isinstance(panel, pl.DataFrame) and "date" in panel.columns else None
    if not isinstance(panel_min, date) or latest is None or latest >= target_session or panel_min > target_session:
        return []
    grid = resolve_session_grid(get_calendar().sessions(panel_min, target_session), panel)
    return [s for s in grid.sessions if latest < s < target_session]


def next_hold_len(prior_held: str | None, prior_hold_len: int, new_held: str | None) -> int:
    if new_held is None:
        return 0
    if new_held == prior_held:
        return int(prior_hold_len) + 1
    return 1


def resolve_primary_ticker(weights: Mapping[str, float]) -> str | None:
    if len(weights) == 0:
        return None
    best_w: float | None = None
    for v in weights.values():
        wf = float(v)
        if best_w is None or wf > best_w:
            best_w = wf
    candidates = [str(k) for k, v in weights.items() if float(v) == best_w]
    return min(candidates)


def apply_live_exposure_and_capacity_limits(
    weights: Mapping[str, float],
    panel: pl.DataFrame,
    *,
    held: Mapping[str, float],
    decision_date: date,
    capital: float,
    strategy_id: str,
    adv_window: int = 20,
) -> dict[str, float]:
    if len(weights) == 0:
        return {}
    from src.backtest.liquidity import cap_target_weights_by_adv
    from src.portfolio.constraints import apply_portfolio_exposure_limits, resolve_exposure_limits_for_model
    from src.universe.instruments import resolve_leverage

    tickers = sorted(set(weights.keys()) | set(held.keys()))
    multiples: dict[str, int] = {}
    for ticker in tickers:
        name: str | None = None
        if (
            isinstance(panel, pl.DataFrame)
            and "ticker" in panel.columns
            and "name" in panel.columns
            and "date" in panel.columns
        ):
            frame = panel.filter((pl.col("ticker") == ticker) & (pl.col("date") == decision_date))
            if frame.height > 0:
                for raw in frame.get_column("name").to_list():
                    if raw is not None:
                        name = str(raw)
                        break
        if name is None:
            raise ValueError(f"missing name for ticker {ticker} on {decision_date}: cannot resolve leverage multiplier")
        multiples[str(ticker)] = int(resolve_leverage(name)[0])
    max_single_weight, max_gross_exposure, min_cash = resolve_exposure_limits_for_model(strategy_id)
    exposure_capped = apply_portfolio_exposure_limits(
        weights,
        multiples,
        max_single_weight=max_single_weight,
        max_gross_exposure=max_gross_exposure,
        min_cash=min_cash,
    )
    adv_by_ticker: dict[str, float] = {}
    if (
        isinstance(panel, pl.DataFrame)
        and "ticker" in panel.columns
        and "trading_value" in panel.columns
        and "date" in panel.columns
    ):
        scoped = panel.filter(pl.col("date") <= decision_date)
        if scoped.height > 0:
            dates = sorted({d for d in scoped.get_column("date").to_list() if isinstance(d, date)})
            window_dates = dates[-int(adv_window) :] if int(adv_window) > 0 else dates
            if window_dates:
                scoped = scoped.filter(pl.col("date").is_in(window_dates))
                for ticker in tickers:
                    tframe = scoped.filter(pl.col("ticker") == ticker)
                    vals = [float(v) for v in tframe.get_column("trading_value").to_list() if v is not None]
                    if not vals:
                        continue
                    adv_by_ticker[str(ticker)] = float(sum(vals) / len(vals))
    capped = cap_target_weights_by_adv(
        exposure_capped, dict(held), float(capital), adv_by_ticker, P27_ADOPTED_MAX_ORDER_TO_ADV
    )
    return {k: float(v) for k, v in capped.items() if float(v) > 1e-12}


def assert_sleeve_inputs_fresh(index_daily: pl.DataFrame, *, decision_date: date) -> None:
    sub = select_kospi_headline_series(index_daily)
    if sub.height == 0:
        raise StaleSleeveInputError(f"sleeve inputs missing at {decision_date.isoformat()}: no headline series rows")
    if decision_date not in sub.get_column("date").to_list():
        raise StaleSleeveInputError(f"sleeve inputs stale at {decision_date.isoformat()}: no row at decision_date")
    return None


@dataclass(frozen=True, slots=True)
class LiveOrderEstimate:
    ticker: str
    weight: float
    price_basis_date: date
    price: float
    est_shares: int
    est_krw: float


def resolve_live_championship_sleeve(index_daily: pl.DataFrame, decision_date: date) -> str:
    if not isinstance(index_daily, pl.DataFrame):
        return ChampionshipSleeve.UNCERTAIN.value
    if index_daily.height == 0:
        return ChampionshipSleeve.UNCERTAIN.value
    mom60_by_date, mom20_by_date, rv20_daily_by_date = kospi_sleeve_feature_maps(index_daily)
    mom60_trunc = {d: v for d, v in mom60_by_date.items() if d <= decision_date}
    mom20_trunc = {d: v for d, v in mom20_by_date.items() if d <= decision_date}
    rv_trunc = {d: v for d, v in rv20_daily_by_date.items() if d <= decision_date}
    if not mom60_trunc and not mom20_trunc and not rv_trunc:
        return ChampionshipSleeve.UNCERTAIN.value
    snap = classify_championship_sleeve_from_maps(
        decision_date=decision_date,
        mom60_by_date=mom60_trunc,
        mom20_by_date=mom20_trunc,
        rv20_daily_by_date=rv_trunc,
    )
    return snap.sleeve.value


def build_live_eligible_snapshot(panel: pl.DataFrame, *, decision_date: date) -> pl.DataFrame:
    from src.backtest.session_grid import resolve_session_grid
    from src.core.calendar import get_calendar
    from src.tournament.attainability import market_candidates_by_session

    if not isinstance(panel, pl.DataFrame) or panel.height == 0 or "date" not in panel.columns:
        return panel.head(0) if isinstance(panel, pl.DataFrame) else panel
    # market_candidates_by_session's underlying eligibility check derives history
    # length from session-index arithmetic across the WHOLE session list -- a
    # single-session list always yields (session_idx - first_seen_idx) == 0,
    # silently failing every ticker's history requirement. The full session
    # range up to decision_date must be passed, mirroring how
    # backtest_attainability_payload builds att_sessions.
    earliest_raw = panel.get_column("date").min()
    if not isinstance(earliest_raw, date):
        return panel.head(0)
    if earliest_raw > decision_date:
        return panel.head(0)
    calendar_sessions = get_calendar().sessions(earliest_raw, decision_date)
    sessions = list(resolve_session_grid(calendar_sessions, panel).sessions)
    if not sessions:
        return panel.head(0)
    candidates = market_candidates_by_session(sessions=sessions, panel=panel)
    eligible = candidates.get(decision_date)
    if not eligible:
        return panel.head(0)
    tickers = list(eligible)
    out = panel.filter(pl.col("ticker").is_in(tickers))
    out = out.filter(pl.col("date") == decision_date)
    return out


def compute_live_target_weights(
    model: object,
    snapshot: pl.DataFrame,
    *,
    decision_date: date,
    held: Mapping[str, float],
    capital: float,
    rules: object,
    championship_sleeve: str,
    scheme: SizingScheme = SizingScheme.TOP1,
    k: int = 1,
) -> PortfolioIntent:
    ctx = DecisionContext(
        decision_date=decision_date,
        regime=None,
        capital=float(capital),
        held=dict(held),
        rules=rules,  # type: ignore[arg-type]
        championship_sleeve=championship_sleeve,
    )
    score_result = model.score(snapshot, ctx)  # type: ignore[attr-defined]
    if score_result is None:
        return resolve_portfolio_intent(None, current_weights=held, score_failed=True)
    if isinstance(score_result, PortfolioIntent):
        return score_result
    allocate_fn = getattr(model, "allocate", None)
    if callable(allocate_fn):
        try:
            alloc_result = allocate_fn(
                score_result,
                regime=None,
                leverage_allowed=None,
                inverse_allowed=None,
            )
        except TypeError:
            alloc_result = allocate_fn(score_result)
        return resolve_portfolio_intent(alloc_result, current_weights=held, score_failed=False)
    alloc_weights = weights_from_scores(score_result, scheme, k)
    return resolve_portfolio_intent(alloc_weights, current_weights=held, score_failed=False)


def estimate_live_order_quantities(
    weights: Mapping[str, float],
    panel: pl.DataFrame,
    *,
    decision_date: date,
    capital: float,
) -> dict[str, LiveOrderEstimate]:
    if len(weights) == 0:
        return {}
    try:
        cap = float(capital)
    except (TypeError, ValueError):
        raise ValueError(f"capital must be finite strictly positive, got {capital!r}") from None
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"capital must be finite strictly positive, got {capital!r}")
    out: dict[str, LiveOrderEstimate] = {}
    for ticker, w in weights.items():
        wf = float(w)
        if wf == 0.0:
            continue
        price: float | None = None
        if isinstance(panel, pl.DataFrame) and "ticker" in panel.columns and "close" in panel.columns:
            frame = panel.filter(pl.col("ticker") == ticker)
            if "date" in panel.columns:
                frame = frame.filter(pl.col("date") == decision_date)
            if frame.height > 0:
                for raw in frame.get_column("close").to_list():
                    try:
                        fv = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fv) and fv > 0:
                        price = fv
                        break
        if price is None:
            raise ValueError(f"missing close price for ticker {ticker} on {decision_date}")
        shares = math.floor(cap * wf / price)
        out[ticker] = LiveOrderEstimate(
            ticker=ticker,
            weight=wf,
            price_basis_date=decision_date,
            price=price,
            est_shares=shares,
            est_krw=float(shares * price),
        )
    return out
