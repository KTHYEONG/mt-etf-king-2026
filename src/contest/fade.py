"""Contest fade overlay daily card (HY2_FADEUP6).

Default holding is HY2; the session after an HY2 close-to-close gain of at
least +6.00% holds HY2I intraday. Stateless and fail-closed: missing data
means no fade.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from src.contest.daily import QuoteBar

_LONG_TICKER_DEFAULT: str = "0193T0"
_INVERSE_TICKER_DEFAULT: str = "0197X0"


class FadeAction(StrEnum):
    HOLD_LONG = "HOLD_LONG"
    FADE = "FADE"
    RETURN_LONG = "RETURN_LONG"
    KEEP_INVERSE = "KEEP_INVERSE"
    NO_DATA = "NO_DATA"
    CONTEST_OVER = "CONTEST_OVER"
    INACTIVE = "INACTIVE"


@dataclass(frozen=True)
class FadeSettings:
    long_ticker: str
    inverse_ticker: str
    base_alias: str
    inverse_alias: str
    threshold_pct: float
    borderline_pct: float
    mode: Literal["close_return", "open_only"]
    close_route: Literal["continuous_1515", "auction_plus_afterhours"]
    endgame_sessions: int
    output_dir: Path

    @classmethod
    def load(cls, contest: Mapping[str, Any]) -> FadeSettings:
        """Validate `contest["fade"]`.

        Raises:
            KeyError: the fade block or a required key is missing.
            ValueError: blank or equal base/inverse aliases, out-of-range threshold/borderline,
                non-positive endgame_sessions, or an unknown mode/close_route literal.
        """
        fade = contest["fade"]
        if not isinstance(fade, Mapping):
            raise KeyError("fade")
        long_ticker = str(fade["long_ticker"])
        inverse_ticker = str(fade["inverse_ticker"])
        base_alias = str(fade["base_alias"]).strip()
        inverse_alias = str(fade["inverse_alias"]).strip()
        if not base_alias or not inverse_alias:
            raise ValueError(f"base_alias/inverse_alias must be non-empty, got {base_alias!r}/{inverse_alias!r}")
        if base_alias == inverse_alias:
            raise ValueError(f"base_alias and inverse_alias must differ, got {base_alias!r}")
        threshold_pct = float(fade["threshold_pct"])
        borderline_pct = float(fade["borderline_pct"])
        mode = str(fade["mode"])
        close_route = str(fade["close_route"])
        endgame_raw = fade["endgame_sessions"]
        if isinstance(endgame_raw, bool):
            raise ValueError("endgame_sessions must be a positive int")
        endgame_sessions = int(endgame_raw)
        output_dir = Path(str(fade["output_dir"]))
        if not math.isfinite(threshold_pct) or threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be positive, got {threshold_pct!r}")
        if not math.isfinite(borderline_pct) or not (0 <= borderline_pct < threshold_pct):
            raise ValueError(f"borderline_pct must satisfy 0 <= b < threshold, got {borderline_pct!r}")
        if mode not in ("close_return", "open_only"):
            raise ValueError(f"unknown mode {mode!r}")
        if close_route not in ("continuous_1515", "auction_plus_afterhours"):
            raise ValueError(f"unknown close_route {close_route!r}")
        if endgame_sessions <= 0:
            raise ValueError(f"endgame_sessions must be positive, got {endgame_raw!r}")
        return cls(
            long_ticker=long_ticker,
            inverse_ticker=inverse_ticker,
            base_alias=base_alias,
            inverse_alias=inverse_alias,
            threshold_pct=threshold_pct,
            borderline_pct=borderline_pct,
            mode=cast("Literal['close_return', 'open_only']", mode),
            close_route=cast("Literal['continuous_1515', 'auction_plus_afterhours']", close_route),
            endgame_sessions=endgame_sessions,
            output_dir=output_dir,
        )


@dataclass(frozen=True)
class FadeCard:
    session: date
    execution_session: date | None
    action: FadeAction
    long_change_pct: float | None
    prev_long_change_pct: float | None
    threshold_pct: float
    borderline: bool
    mode: str
    close_route: str
    sessions_remaining_after_execution: int
    endgame: bool
    warnings: tuple[str, ...]
    state_alias: str | None = None


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def decide_fade(
    change_pct: float | None,
    prev_change_pct: float | None,
    settings: FadeSettings,
) -> tuple[FadeAction, bool]:
    """Map the signal session's HY2 close-to-close change to the next session's action.

    The intraday reversal edge measured in the probe follows large prior up-moves, so only a change at or above
    `threshold_pct` triggers a fade. There is no symmetric rule for down-moves: the default holding is already long.

    Args:
        change_pct: HY2 close-to-close change of session T in percent, or None when unavailable.
        prev_change_pct: the same for session T-1 (used only in open_only mode to know whether HY2I is held
            overnight into T+1), or None.
        settings: validated fade settings.

    Returns:
        (action, borderline) where borderline is True when change_pct is within borderline_pct of the threshold.
    """
    change = _finite(change_pct)
    if change is None:
        return FadeAction.NO_DATA, False
    borderline = abs(change - settings.threshold_pct) < settings.borderline_pct
    is_fade = change >= settings.threshold_pct
    if settings.mode == "close_return":
        return (FadeAction.FADE if is_fade else FadeAction.HOLD_LONG), borderline
    prev = _finite(prev_change_pct)
    held_inverse = prev is not None and prev >= settings.threshold_pct
    if held_inverse and is_fade:
        return FadeAction.KEEP_INVERSE, borderline
    if held_inverse:
        return FadeAction.RETURN_LONG, borderline
    return (FadeAction.FADE if is_fade else FadeAction.HOLD_LONG), borderline


def _bar_change(bar: QuoteBar | None) -> float | None:
    if bar is None:
        return None
    if not (bar.prev_close > 0 and bar.close > 0):
        return None
    if not (math.isfinite(bar.prev_close) and math.isfinite(bar.close)):
        return None
    change = (bar.close / bar.prev_close - 1.0) * 100.0
    return change if math.isfinite(change) else None


def build_fade_card(
    session: date,
    calendar: Any,
    bars: Mapping[str, QuoteBar],
    prev_bars: Mapping[str, QuoteBar] | None,
    config: Mapping[str, Any],
    state_alias: str | None,
) -> FadeCard:
    """Assemble the fade card for signal session `session`, gated on the weekly-recommended holding.

    `state_alias` is the caller's read of the weekly decision state (the same value `contest-weekly` writes to
    `data/state/contest_position.json` and `contest-daily` already reads). The fade signal is meaningful only
    while that recommendation equals `settings.base_alias`: the probe's modeled P(rank 1) assumed the account is
    already holding the base vehicle, and a fade trade executed while actually holding a different vehicle would
    be a real-money error, not a benign no-op.

    Args:
        state_alias: the current weekly-recommended alias, or None when it could not be determined. This function
            performs no I/O to obtain it; the caller reads the state file exactly once, as it already does for
            `contest-daily`.

    Returns:
        - CONTEST_OVER when `session >= contest.end_date` (checked first; unchanged from the prior version).
        - INACTIVE when `state_alias != settings.base_alias` (covers None and any other alias, including
          `settings.inverse_alias`). No quote lookups are required to reach this branch.
        - NO_DATA with warning QUOTES_MISSING when the base alias matches but the long-ticker bar is absent or
          non-positive (unchanged from the prior version).
        - Otherwise the `decide_fade` action (unchanged from the prior version).
    """
    settings = FadeSettings.load(config)
    end_date = date.fromisoformat(str(config["end_date"]))
    if session >= end_date:
        return FadeCard(
            session=session,
            execution_session=None,
            action=FadeAction.CONTEST_OVER,
            long_change_pct=None,
            prev_long_change_pct=None,
            threshold_pct=settings.threshold_pct,
            borderline=False,
            mode=settings.mode,
            close_route=settings.close_route,
            sessions_remaining_after_execution=0,
            endgame=False,
            warnings=(),
            state_alias=state_alias,
        )
    try:
        execution = calendar.next_session(session)
    except ValueError:
        execution = None
    remaining_after = 0
    if execution is not None:
        if execution <= end_date:
            try:
                remaining_after = len([s for s in calendar.sessions(execution, end_date) if s > execution])
            except ValueError:  # pragma: no cover - calendar range mismatch
                remaining_after = 0
        else:
            remaining_after = 0
    endgame = remaining_after < settings.endgame_sessions
    if state_alias != settings.base_alias:
        gate_warning = "STATE_UNKNOWN" if state_alias is None else "STATE_NOT_BASE"
        inactive_warnings: list[str] = [gate_warning]
        if endgame:
            inactive_warnings.append("ENDGAME_REVIEW")
        return FadeCard(
            session=session,
            execution_session=execution,
            action=FadeAction.INACTIVE,
            long_change_pct=None,
            prev_long_change_pct=None,
            threshold_pct=settings.threshold_pct,
            borderline=False,
            mode=settings.mode,
            close_route=settings.close_route,
            sessions_remaining_after_execution=remaining_after,
            endgame=endgame,
            warnings=tuple(inactive_warnings),
            state_alias=state_alias,
        )
    bar = bars.get(settings.long_ticker) if bars else None
    bar_valid = (
        bar is not None
        and bar.prev_close > 0
        and bar.close > 0
        and math.isfinite(bar.prev_close)
        and math.isfinite(bar.close)
    )
    change = _bar_change(bar) if bar_valid else None
    prev_change = _bar_change(prev_bars.get(settings.long_ticker) if prev_bars else None)
    warnings: list[str] = []
    if not bar_valid:
        if settings.mode == "open_only" and prev_change is None:
            warnings.append("PREV_UNKNOWN")
        if endgame:
            warnings.append("ENDGAME_REVIEW")
        return FadeCard(
            session=session,
            execution_session=execution,
            action=FadeAction.NO_DATA,
            long_change_pct=None,
            prev_long_change_pct=prev_change,
            threshold_pct=settings.threshold_pct,
            borderline=False,
            mode=settings.mode,
            close_route=settings.close_route,
            sessions_remaining_after_execution=remaining_after,
            endgame=endgame,
            warnings=("QUOTES_MISSING", *warnings),
            state_alias=state_alias,
        )
    action, borderline = decide_fade(change, prev_change, settings)
    if settings.mode == "open_only" and prev_change is None:
        warnings.append("PREV_UNKNOWN")
    if borderline:
        warnings.append("BORDERLINE")
    if endgame:
        warnings.append("ENDGAME_REVIEW")
    return FadeCard(
        session=session,
        execution_session=execution,
        action=action,
        long_change_pct=change,
        prev_long_change_pct=prev_change,
        threshold_pct=settings.threshold_pct,
        borderline=borderline,
        mode=settings.mode,
        close_route=settings.close_route,
        sessions_remaining_after_execution=remaining_after,
        endgame=endgame,
        warnings=tuple(warnings),
        state_alias=state_alias,
    )


def fade_to_dict(card: FadeCard) -> dict[str, Any]:
    """JSON-ready primitives (ISO dates, enum values)."""
    return {
        "session": card.session.isoformat(),
        "execution_session": card.execution_session.isoformat() if card.execution_session else None,
        "action": card.action.value,
        "state_alias": card.state_alias,
        "long_change_pct": card.long_change_pct,
        "prev_long_change_pct": card.prev_long_change_pct,
        "threshold_pct": card.threshold_pct,
        "borderline": card.borderline,
        "mode": card.mode,
        "close_route": card.close_route,
        "sessions_remaining_after_execution": card.sessions_remaining_after_execution,
        "endgame": card.endgame,
        "warnings": list(card.warnings),
    }


def _fmt_change(value: float | None) -> str:
    return f"{value:+.2f}%" if value is not None else "없음"


def render_fade_markdown(card: FadeCard, base_alias: str = "HY2") -> str:
    """Korean operator card: the signal, the weekly gate, the action, and a timed checklist.

    The checklist uses the normative execution semantics of this spec (open and close windows, order types, and the
    order of SELL before BUY). An INACTIVE card carries no timed order windows: only the gate reason and the
    weekly recommendation that governs instead.
    """
    long_code = _LONG_TICKER_DEFAULT
    inverse_code = _INVERSE_TICKER_DEFAULT
    state_display = card.state_alias if card.state_alias is not None else "확인 불가"
    lines = [f"# 페이드 오버레이 카드 ({card.session.isoformat()})", ""]
    exec_session = card.execution_session.isoformat() if card.execution_session else "없음"
    lines.append(f"- 신호 세션: {card.session.isoformat()} (HY2 등락률 {_fmt_change(card.long_change_pct)}, 기준 +{card.threshold_pct:.2f}%)")
    lines.append(f"- 집행 세션: {exec_session}")
    lines.append(f"- 이번 주 보유(주간 카드 기준): {state_display}")
    lines.append(f"- 액션: {card.action.value} (모드 {card.mode}, 종가경로 {card.close_route})")
    lines.append(f"- 집행 후 잔여 세션: {card.sessions_remaining_after_execution}")
    if card.borderline:
        lines.append(f"- 경계값: HTS 전일대비 등락률({long_code})로 최종 판단")
    if "PREV_UNKNOWN" in card.warnings:
        lines.append("- 이전 보유 불명: 현재 보유를 HTS에서 확인 후 집행")
    if "ENDGAME_REVIEW" in card.warnings:
        lines.append("- 엔드게임: 잔여 세션 적음 — 수동 검토 (ENDGAME_REVIEW)")
    lines.append("")
    if card.action is FadeAction.FADE:
        lines.append("## 집행 체크리스트 (페이드)")
        lines.append(f"1. 08:40–08:59 HY2({long_code}) 전량 시장가 매도 (장전)")  # noqa: RUF001
        lines.append(f"2. 09:00–09:02 매도 체결 확인 후 HY2I({inverse_code}) 시장가 매수 (매수가능금액 전액)")  # noqa: RUF001
        if card.close_route == "auction_plus_afterhours":
            lines.append(f"3. 15:20–15:29 HY2I({inverse_code}) 시장가 매도 (15:30 종가 체결)")  # noqa: RUF001
            lines.append(f"4. 15:40–16:00 시간외 종가 HY2({long_code}) 매수 (매수가능금액 전액)")  # noqa: RUF001
        else:
            lines.append(f"3. 15:15–15:19 HY2I({inverse_code}) 전량 시장가 매도 → 체결 확인 후 HY2({long_code}) 시장가 매수 (매수가능금액 전액)")  # noqa: RUF001
            lines.append("4. 15:20 이후 주문 금지")
    elif card.action is FadeAction.RETURN_LONG:
        lines.append("## 집행 체크리스트 (복귀)")
        lines.append(f"1. 08:40–08:59 HY2I({inverse_code}) 전량 시장가 매도 (장전)")  # noqa: RUF001
        lines.append(f"2. 09:00–09:02 매도 체결 확인 후 HY2({long_code}) 시장가 매수 (매수가능금액 전액)")  # noqa: RUF001
    elif card.action is FadeAction.HOLD_LONG:
        lines.append("주문 없음 — HY2 보유 유지")
    elif card.action is FadeAction.KEEP_INVERSE:
        lines.append("주문 없음 — HY2I 보유 유지")
    elif card.action is FadeAction.NO_DATA:
        lines.append("주문 없음 — 현재 보유 유지 (자료 부족, 신규 주문 없음)")
        lines.append(f"HTS 전일대비 등락률({long_code})이 +{card.threshold_pct:.2f}% 이상이면 FADE 체크리스트를 수동 적용")
    elif card.action is FadeAction.INACTIVE:
        lines.append("## 집행 (없음 — 비활성)")
        lines.append("주문 없음 — 이번 카드에서 파생되는 주문 없음 — 주간 카드의 현재 추천이 우선")
        lines.append(f"- 사유: 주간 보유({state_display})가 페이드 기준 보유({base_alias})와 달라 발동하지 않음")
    else:
        lines.append("대회 종료 — 주문 없음")
    lines.append("")
    lines.append("## 경고")
    if card.warnings:
        lines.extend(f"- {warning}" for warning in card.warnings)
    else:
        lines.append("- 없음")
    return "\n".join(lines) + "\n"
