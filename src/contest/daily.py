"""Daily shadow evaluation of the leader-mirror contest policy.

Each weekday after the 16:00 KST leaderboard rebuild, the module does four things:

1. Archive same-day closes for the liquid contest ETFs.
2. Identify top-N holders whose daily returns match one exposure over consecutive sessions.
3. Record the mirror recommendation for the next open.
4. Accumulate predeclared validation metrics.

Shadow mode: outputs are informational and never touch the live position state. Fail-closed: missing leaderboard or quotes produce a NO_DATA card with no recommendation.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx

from src.contest.leaderboard import (
    LeaderboardSchemaError,
    LeaderboardSnapshot,
    infer_single_vehicle_holders,
    load_snapshot,
)

logger = logging.getLogger(__name__)

_YAHOO_CHART_URL: Final[str] = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_USER_AGENT: Final[str] = "mt-etf-king-2026 contest-daily/1.0"
_KST: Final[ZoneInfo] = ZoneInfo("Asia/Seoul")


class ShadowAction(StrEnum):
    RECORD = "RECORD"
    NO_DATA = "NO_DATA"
    CONTEST_OVER = "CONTEST_OVER"


class GateVerdict(StrEnum):
    PENDING = "PENDING"
    ADOPT = "ADOPT"
    KEEP_STATIC = "KEEP_STATIC"


@dataclass(frozen=True)
class QuoteBar:
    ticker: str
    prev_close: float
    open: float
    close: float


@dataclass(frozen=True)
class StreakHolder:
    user_name: str
    exposure: str
    weight: float
    streak: int


@dataclass(frozen=True)
class DailyShadowCard:
    session: date
    execution_session: date | None
    action: ShadowAction
    mirror_target: str | None
    long_leader: tuple[str, float] | None
    inverse_leader: tuple[str, float] | None
    our_equity: float | None
    our_equity_source: str | None
    identified_top_share: float | None
    daily_churn: float | None
    review: bool
    gate: GateVerdict
    gate_detail: dict[str, Any]
    holders: tuple[StreakHolder, ...]
    warnings: tuple[str, ...]


def _download_quote(client: httpx.Client, ticker: str, session: date, suffix: str) -> QuoteBar | None:
    response = client.get(
        _YAHOO_CHART_URL.format(symbol=f"{ticker}{suffix}"), params={"interval": "1d", "range": "max"}
    )
    response.raise_for_status()
    payload = response.json()
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    opens = quote["open"]
    closes = quote["close"]
    by_day: dict[date, tuple[float, float]] = {}
    for ts, open_, close in zip(timestamps, opens, closes, strict=False):
        try:
            day = datetime.fromtimestamp(float(ts), tz=UTC).astimezone(_KST).date()
            open_f, close_f = float(open_), float(close)
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        if not (open_f > 0 and close_f > 0 and math.isfinite(open_f) and math.isfinite(close_f)):
            continue
        by_day[day] = (open_f, close_f)
    bar = by_day.get(session)
    if bar is None:
        return None
    earlier = [day for day in by_day if day < session]
    if not earlier:
        return None
    prev_close = by_day[max(earlier)][1]
    open_f, close_f = bar
    return QuoteBar(ticker=ticker, prev_close=prev_close, open=open_f, close=close_f)


def fetch_quotes(tickers: Sequence[str], session: date, suffix: str, timeout_s: float) -> dict[str, QuoteBar]:
    """Download the `session` open/close and previous-session close for each ticker from the Yahoo chart API
    (same endpoint as the reference seed). Bars are keyed by KST trading date.

    A ticker whose bar for `session`, or whose previous bar, is absent or non-positive is omitted (inference
    narrows; the caller decides fail-closed). Network or schema failures for one ticker are logged as
    `[DATA] contest_quotes ticker=<t> status=skip` and do not abort the others.
    """
    bars: dict[str, QuoteBar] = {}
    with httpx.Client(timeout=timeout_s, headers={"User-Agent": _USER_AGENT}) as client:
        for ticker in tickers:
            try:
                bar = _download_quote(client, ticker, session, suffix)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
                bar = None
            if bar is None:
                logger.info(f"[DATA] contest_quotes ticker={ticker} status=skip")
                continue
            bars[ticker] = bar
    return bars


def archive_quotes(root: Path, session: date, bars: Mapping[str, QuoteBar]) -> Path:
    """Write `<root>/<YYYYMMDD>.json` once (atomic tmp+rename). An existing file is kept and returned unchanged.
    The first successful capture is the record, so later reruns cannot rewrite history."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{session.strftime('%Y%m%d')}.json"
    if path.is_file():
        return path
    payload = {
        "session": session.isoformat(),
        "bars": {
            ticker: {"prev_close": bar.prev_close, "open": bar.open, "close": bar.close}
            for ticker, bar in bars.items()
        },
    }
    with tempfile.NamedTemporaryFile(dir=str(root), suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_quotes(root: Path, session: date) -> dict[str, QuoteBar] | None:
    """Read an archived session file, or None when absent."""
    path = root / f"{session.strftime('%Y%m%d')}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        bars = raw["bars"]
        out = {
            str(ticker): QuoteBar(
                ticker=str(ticker),
                prev_close=float(data["prev_close"]),
                open=float(data["open"]),
                close=float(data["close"]),
            )
            for ticker, data in bars.items()
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return out


def infer_streak_holders(
    history: Sequence[tuple[LeaderboardSnapshot, Mapping[str, float], Mapping[str, str]]],
    tol_pct: float,
    min_weight: float,
) -> dict[str, StreakHolder]:
    """Identify each user's exposure at the LAST snapshot in `history` and count how many consecutive trailing
    sessions the same exposure was identified.

    `history` is chronological: (snapshot, changes keyed by wrapper key, exposure_of). Single-session
    identification uses `infer_single_vehicle_holders` with exposure grouping. The streak counts trailing sessions
    whose identification equals the last one; a session where the user is absent or ambiguous breaks it. Users not
    identified at the last session are omitted.
    """
    per_session = [
        infer_single_vehicle_holders(snapshot, changes, tol_pct, min_weight, exposure_of)
        for snapshot, changes, exposure_of in history
    ]
    if not per_session:
        return {}
    last = per_session[-1]
    out: dict[str, StreakHolder] = {}
    for user, (exposure, weight) in last.items():
        streak = 1
        for prev in reversed(per_session[:-1]):
            hit = prev.get(user)
            if hit is None or hit[0] != exposure:
                break
            streak += 1
        out[user] = StreakHolder(user_name=user, exposure=exposure, weight=weight, streak=streak)
    return out


def _is_inverse_side(exposure: str) -> bool:
    return exposure.endswith("I")


def mirror_target(
    snapshot: LeaderboardSnapshot,
    holders: Mapping[str, StreakHolder],
    nickname: str,
    long_alias: str,
    inverse_alias: str,
    min_streak: int,
) -> tuple[str, tuple[str, float] | None, tuple[str, float] | None]:
    """Leader-mirror rule: compare the best visible long-side holder with the best inverse-side holder (streak >=
    min_streak, excluding us), and hold the side opposite to the higher one.

    A side with no identified holder is valued at the lowest visible equity (the top-50 floor): unseen rivals may
    ride it from below. On a tie, choose `long_alias`.

    Returns:
        (target alias, long leader (user, equity) or None, inverse leader or None).
    """
    floor = min((1.0 + entry.total_return_pct / 100.0 for entry in snapshot.entries), default=1.0)
    by_user = {entry.user_name: entry for entry in snapshot.entries}
    long_best: tuple[str, float] | None = None
    inverse_best: tuple[str, float] | None = None
    for user, holder in holders.items():
        if user == nickname or holder.streak < min_streak:
            continue
        entry = by_user.get(user)
        if entry is None:
            continue
        equity = 1.0 + entry.total_return_pct / 100.0
        if _is_inverse_side(holder.exposure):
            if inverse_best is None or equity > inverse_best[1]:
                inverse_best = (user, equity)
        elif long_best is None or equity > long_best[1]:
            long_best = (user, equity)
    long_value = long_best[1] if long_best is not None else floor
    inverse_value = inverse_best[1] if inverse_best is not None else floor
    target = inverse_alias if long_value > inverse_value else long_alias
    return target, long_best, inverse_best


def _metric_session(row: Mapping[str, Any]) -> date | None:
    raw = row.get("session")
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    return None


def evaluate_gates(
    metrics: Sequence[Mapping[str, Any]], session: date, shadow_cfg: Mapping[str, Any]
) -> tuple[GateVerdict, dict[str, Any]]:
    """Predeclared adoption gates over metrics rows with start_session <= row.session <= session.

    G1: in the last `g1_window` RECORD rows, at least `g1_min_days` have identified_top_share >= g1_min_identified_share.
    G2: mean daily_churn over RECORD rows with a value <= g2_max_daily_churn.
    G3: count of NO_DATA rows <= g3_max_no_data_days.

    Returns PENDING before gate_session. Otherwise ADOPT only when G1, G2 and G3 all pass, else KEEP_STATIC.
    Detail carries each gate's value and pass flag. The gate parameters are fixed in config before the shadow
    period and must not be tuned after observing metrics.
    """
    start_session = date.fromisoformat(str(shadow_cfg["start_session"]))
    gate_session = date.fromisoformat(str(shadow_cfg["gate_session"]))
    gates = shadow_cfg.get("gates", {})
    g1_share = float(gates.get("g1_min_identified_share", 0.5))
    g1_window = int(gates.get("g1_window", 5))
    g1_days = int(gates.get("g1_min_days", 3))
    g2_max = float(gates.get("g2_max_daily_churn", 0.15))
    g3_max = int(gates.get("g3_max_no_data_days", 2))
    dated: list[tuple[date, Mapping[str, Any]]] = []
    for row in metrics:
        day = _metric_session(row)
        if day is not None and start_session <= day <= session:
            dated.append((day, row))
    dated.sort(key=lambda item: item[0])
    record_rows = [row for _, row in dated if str(row.get("action")) == ShadowAction.RECORD.value]
    window = record_rows[-g1_window:]
    g1_count = sum(
        1
        for row in window
        if row.get("identified_top_share") is not None
        and float(row["identified_top_share"]) >= g1_share
    )
    g1_pass = g1_count >= g1_days
    churns = [float(row["daily_churn"]) for row in record_rows if row.get("daily_churn") is not None]
    g2_mean = sum(churns) / len(churns) if churns else None
    g2_pass = g2_mean is not None and g2_mean <= g2_max
    g3_count = sum(1 for _, row in dated if str(row.get("action")) == ShadowAction.NO_DATA.value)
    g3_pass = g3_count <= g3_max
    detail = {
        "g1": {"value": g1_count, "pass": g1_pass},
        "g2": {"value": g2_mean, "pass": g2_pass},
        "g3": {"value": g3_count, "pass": g3_pass},
    }
    if session < gate_session:
        return GateVerdict.PENDING, detail
    if g1_pass and g2_pass and g3_pass:
        return GateVerdict.ADOPT, detail
    return GateVerdict.KEEP_STATIC, detail


def _load_or_empty(archive_root: Path, day: date) -> LeaderboardSnapshot:
    try:
        return load_snapshot(archive_root, day)
    except (OSError, LeaderboardSchemaError):
        return LeaderboardSnapshot(base_date=day, requested_at="", entries=(), purchases={})


def _quote_changes(
    bars: Mapping[str, QuoteBar], exposures: Mapping[str, Sequence[str]]
) -> tuple[dict[str, float], dict[str, str]]:
    changes: dict[str, float] = {}
    exposure_of: dict[str, str] = {}
    for exposure, tickers in exposures.items():
        for ticker in tickers:
            bar = bars.get(ticker)
            if bar is None or not (bar.prev_close > 0 and bar.close > 0):
                continue
            if not (math.isfinite(bar.prev_close) and math.isfinite(bar.close)):
                continue
            key = f"{exposure}@{ticker}"
            changes[key] = (bar.close / bar.prev_close - 1.0) * 100.0
            exposure_of[key] = exposure
    return changes, exposure_of


def _metric_row(
    session: date,
    action: ShadowAction,
    mirror: str | None,
    share: float | None,
    churn: float | None,
) -> dict[str, Any]:
    return {
        "session": session.isoformat(),
        "action": action.value,
        "mirror_target": mirror,
        "identified_top_share": share,
        "daily_churn": churn,
    }


def build_daily_card(
    session: date,
    calendar: Any,
    archive_root: Path,
    quotes_root: Path,
    config: Mapping[str, Any],
    state_alias: str | None,
    our_equity_override: tuple[float, str] | None,
    prior_metrics: Sequence[Mapping[str, Any]],
) -> DailyShadowCard:
    """Assemble the shadow card for `session` from archived leaderboards and archived quotes only (no network).

    Uses up to `review_streak + 1` trailing sessions of history. Our equity comes from the leaderboard entry, else
    `our_equity_override` (the same semantics as the weekly card).

    Returns:
        - NO_DATA with warning LEADERBOARD_STALE when the snapshot for `session` is missing.
        - NO_DATA with warning QUOTES_MISSING when fewer than both mirror-side exposures are quoted.
        - CONTEST_OVER after end_date.
        - RECORD otherwise.
    """
    nickname = str(config.get("nickname", ""))
    end_date = date.fromisoformat(str(config["end_date"]))
    inference = config.get("inference", {})
    tol_pct = float(inference.get("match_tol_pct", 0.02))
    min_weight = float(inference.get("min_weight", 0.90))
    shadow = config["shadow"]
    long_alias = str(shadow["long_alias"])
    inverse_alias = str(shadow["inverse_alias"])
    exposures = {str(exp): [str(t) for t in tickers] for exp, tickers in dict(shadow["exposures"]).items()}
    min_streak = int(shadow["min_streak"])
    review_streak = int(shadow["review_streak"])
    top_n = int(shadow["top_n"])

    def gate_with_current(
        action: ShadowAction, mirror: str | None, share: float | None, churn: float | None
    ) -> tuple[GateVerdict, dict[str, Any]]:
        return evaluate_gates(
            [*prior_metrics, _metric_row(session, action, mirror, share, churn)], session, shadow
        )

    def our_equity_from(
        snapshot: LeaderboardSnapshot | None,
    ) -> tuple[float | None, str | None]:
        entry = snapshot.entry_for(nickname) if snapshot is not None and nickname else None
        if entry is not None:
            return 1.0 + entry.total_return_pct / 100.0, "leaderboard"
        if our_equity_override is not None:
            source = "manual" if str(our_equity_override[1]) == "OUR_RETURN_MANUAL" else "estimated"
            return float(our_equity_override[0]), source
        return None, None

    try:
        snapshot = load_snapshot(archive_root, session)
    except (OSError, LeaderboardSchemaError):
        snapshot = None
    if session > end_date:
        our_equity, our_source = our_equity_from(snapshot)
        gate, detail = gate_with_current(ShadowAction.CONTEST_OVER, None, None, None)
        return DailyShadowCard(
            session=session, execution_session=None, action=ShadowAction.CONTEST_OVER,
            mirror_target=None, long_leader=None, inverse_leader=None,
            our_equity=our_equity, our_equity_source=our_source,
            identified_top_share=None, daily_churn=None, review=False,
            gate=gate, gate_detail=detail, holders=(), warnings=(),
        )
    if snapshot is None:
        our_equity, our_source = our_equity_from(None)
        gate, detail = gate_with_current(ShadowAction.NO_DATA, None, None, None)
        return DailyShadowCard(
            session=session, execution_session=None, action=ShadowAction.NO_DATA,
            mirror_target=None, long_leader=None, inverse_leader=None,
            our_equity=our_equity, our_equity_source=our_source,
            identified_top_share=None, daily_churn=None, review=False,
            gate=gate, gate_detail=detail, holders=(), warnings=("LEADERBOARD_STALE",),
        )
    bars = load_quotes(quotes_root, session) or {}
    quoted = {exp for exp, tickers in exposures.items() if any(t in bars for t in tickers)}
    if long_alias not in quoted or inverse_alias not in quoted:
        our_equity, our_source = our_equity_from(snapshot)
        gate, detail = gate_with_current(ShadowAction.NO_DATA, None, None, None)
        return DailyShadowCard(
            session=session, execution_session=None, action=ShadowAction.NO_DATA,
            mirror_target=None, long_leader=None, inverse_leader=None,
            our_equity=our_equity, our_equity_source=our_source,
            identified_top_share=None, daily_churn=None, review=False,
            gate=gate, gate_detail=detail, holders=(), warnings=("QUOTES_MISSING",),
        )

    hist_sessions = [session]
    for _ in range(review_streak):
        try:
            hist_sessions.append(calendar.previous_session(hist_sessions[-1]))
        except ValueError:
            break
    hist_sessions.reverse()
    history: list[tuple[LeaderboardSnapshot, Mapping[str, float], Mapping[str, str]]] = []
    for hist_day in hist_sessions:
        snap = snapshot if hist_day == session else _load_or_empty(archive_root, hist_day)
        hist_bars = bars if hist_day == session else (load_quotes(quotes_root, hist_day) or {})
        changes, exposure_of = _quote_changes(hist_bars, exposures)
        history.append((snap, changes, exposure_of))

    holders = infer_streak_holders(history, tol_pct, min_weight)
    target, long_leader, inverse_leader = mirror_target(
        snapshot, holders, nickname, long_alias, inverse_alias, min_streak
    )
    ranked = sorted(snapshot.entries, key=lambda entry: entry.rank)[:top_n]
    denom = [entry for entry in ranked if entry.user_name != nickname]
    identified = sum(
        1
        for entry in denom
        if (holder := holders.get(entry.user_name)) is not None and holder.streak >= min_streak
    )
    share: float | None = identified / len(denom) if denom else None
    churn: float | None
    if len(history) >= 2:
        prev_snap, prev_changes, prev_exposure = history[-2]
        prev_id = infer_single_vehicle_holders(prev_snap, prev_changes, tol_pct, min_weight, prev_exposure)
        cur_snap, cur_changes, cur_exposure = history[-1]
        cur_id = infer_single_vehicle_holders(cur_snap, cur_changes, tol_pct, min_weight, cur_exposure)
        top_users = {entry.user_name for entry in ranked}
        both = [user for user in top_users if user in cur_id and user in prev_id]
        churn = (
            sum(1 for user in both if cur_id[user][0] != prev_id[user][0]) / len(both) if both else None
        )
    else:
        churn = None
    our_entry = snapshot.entry_for(nickname) if nickname else None
    if our_entry is not None and state_alias is not None:
        ahead = [entry for entry in snapshot.entries if entry.rank < our_entry.rank]
        review = any(
            (holder := holders.get(entry.user_name)) is not None
            and holder.exposure == state_alias
            and holder.streak >= review_streak
            for entry in ahead
        )
    else:
        review = False
    our_equity, our_source = our_equity_from(snapshot)
    gate, detail = gate_with_current(ShadowAction.RECORD, target, share, churn)
    return DailyShadowCard(
        session=session, execution_session=calendar.next_session(session), action=ShadowAction.RECORD,
        mirror_target=target, long_leader=long_leader, inverse_leader=inverse_leader,
        our_equity=our_equity, our_equity_source=our_source,
        identified_top_share=share, daily_churn=churn, review=review,
        gate=gate, gate_detail=detail,
        holders=tuple(sorted(holders.values(), key=lambda holder: holder.user_name)),
        warnings=(),
    )


def card_metric_row(card: DailyShadowCard) -> dict[str, Any]:
    """Metric row persisted to metrics.jsonl (one row per session, upserted)."""
    return {
        "session": card.session.isoformat(),
        "action": card.action.value,
        "mirror_target": card.mirror_target,
        "identified_top_share": card.identified_top_share,
        "daily_churn": card.daily_churn,
        "gate": card.gate.value,
        "review": card.review,
    }


def daily_to_dict(card: DailyShadowCard) -> dict[str, Any]:
    """Serialize a shadow card to JSON-ready primitives."""
    return {
        "session": card.session.isoformat(),
        "execution_session": card.execution_session.isoformat() if card.execution_session else None,
        "action": card.action.value,
        "mirror_target": card.mirror_target,
        "long_leader": list(card.long_leader) if card.long_leader else None,
        "inverse_leader": list(card.inverse_leader) if card.inverse_leader else None,
        "our_equity": card.our_equity,
        "our_equity_source": card.our_equity_source,
        "identified_top_share": card.identified_top_share,
        "daily_churn": card.daily_churn,
        "review": card.review,
        "gate": card.gate.value,
        "gate_detail": {key: dict(value) for key, value in card.gate_detail.items()},
        "holders": [
            {"user_name": holder.user_name, "exposure": holder.exposure,
             "weight": holder.weight, "streak": holder.streak}
            for holder in card.holders
        ],
        "warnings": list(card.warnings),
    }


def _fmt_opt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "없음"


def render_daily_markdown(card: DailyShadowCard) -> str:
    """Korean shadow card: mirror recommendation for the next open, long/inverse leaders, identification share, churn, review flag, and gate status."""
    lines = [f"# 일일 그림자 카드 ({card.session.isoformat()})", ""]
    mirror = card.mirror_target or "없음"
    exec_session = card.execution_session.isoformat() if card.execution_session else "없음"
    lines.append(f"- 그림자 추천 (다음 시가): {mirror} (집행일 {exec_session})")
    if card.long_leader is not None:
        lines.append(f"- 롱 선두: {card.long_leader[0]} (자본 {card.long_leader[1]:.4f})")
    else:
        lines.append("- 롱 선두: 없음")
    if card.inverse_leader is not None:
        lines.append(f"- 인버스 선두: {card.inverse_leader[0]} (자본 {card.inverse_leader[1]:.4f})")
    else:
        lines.append("- 인버스 선두: 없음")
    lines.append(f"- 식별률: {_fmt_opt(card.identified_top_share)}")
    lines.append(f"- 전환율: {_fmt_opt(card.daily_churn)}")
    lines.append(f"- REVIEW: {'참' if card.review else '거짓'}")
    detail = " ".join(
        f"{name}={'통과' if info.get('pass') else '미통과'}({info.get('value')})"
        for name, info in card.gate_detail.items()
    )
    lines.append(f"- 게이트 현황: {card.gate.value} {detail}".rstrip())
    lines.append("")
    lines.append("## 추정 보유")
    if card.holders:
        lines.extend(
            f"- {holder.user_name}: {holder.exposure} (비중 {holder.weight:.3f}, 연속 {holder.streak})"
            for holder in card.holders
        )
    else:
        lines.append("- 없음")
    lines.append("")
    lines.append("## 경고")
    lines.extend(f"- {warning}" for warning in card.warnings)
    if not card.warnings:
        lines.append("- 없음")
    lines.append("")
    lines.append("※ 그림자 모드: 실제 매매 기준은 주간 카드/현재 보유")
    return "\n".join(lines) + "\n"
