"""Live sticky decision explanation (read-only operator labels)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from src.core.calendar import TradingCalendar

# 추석/설 연휴 + 주말을 덮는 탐색 범위(14일).
NEXT_SESSION_LOOKAHEAD_DAYS: Final[int] = 14


class LiveAction(StrEnum):
    """Operator action implied by moving from the prior ledger holding to the new one.

    Execution is always at the next session's open; ``HOLD`` and ``STAY_CASH`` mean no order.
    """

    BUY = "BUY"
    SELL_ALL = "SELL_ALL"
    SWITCH = "SWITCH"
    HOLD = "HOLD"
    STAY_CASH = "STAY_CASH"


class LiveReason(StrEnum):
    """Machine-readable label for why a live sticky decision was taken."""

    ANCHOR_LATCH_HOLD = "ANCHOR_LATCH_HOLD"
    ANCHOR_PICK = "ANCHOR_PICK"
    ANCHOR_STOP_SWITCH = "ANCHOR_STOP_SWITCH"
    ANCHOR_STOP_CASH = "ANCHOR_STOP_CASH"
    NO_ANCHOR_CASH = "NO_ANCHOR_CASH"
    MOM60_LEADER = "MOM60_LEADER"
    OTHER = "OTHER"


SLEEVE_KO: Final[dict[str, str]] = {
    "CRASH_REBOUND": "급락 후 반등",
    "INACTIVE": "비활성(신규 진입 없음)",
    "LOTTERY_ON": "상승 추세(60일 모멘텀)",
    "UNCERTAIN": "판단 불가",
}

ACTION_KO: Final[dict[str, str]] = {
    "BUY": "신규 매수",
    "SELL_ALL": "전량 매도",
    "SWITCH": "종목 교체(전량 매도 후 매수)",
    "HOLD": "보유 유지(주문 없음)",
    "STAY_CASH": "현금 유지(주문 없음)",
}


@dataclass(frozen=True, slots=True)
class AnchorMonitorRow:
    """Stop distance of one campaign anchor at the decision-date close."""

    ticker: str
    name: str
    close: float
    mom_20: float
    drawdown_20: float
    roll_max_20: float
    stop_close: float
    stopped: bool


@dataclass(frozen=True, slots=True)
class LiveDecisionExplanation:
    """Operator-facing explanation of a live sticky decision."""

    decision_date: date
    execution_date: date | None
    sleeve: str
    prior_held: str | None
    prior_weight: float
    new_held: str | None
    new_weight: float
    action: LiveAction
    reason: LiveReason
    reason_ko: str
    anchors: tuple[AnchorMonitorRow, ...]


def derive_live_action(prior_held: str | None, new_held: str | None) -> LiveAction:
    """Map a ledger transition to the operator action (weight-only changes are HOLD)."""
    if prior_held is None and new_held is None:
        return LiveAction.STAY_CASH
    if prior_held is None:
        return LiveAction.BUY
    if new_held is None:
        return LiveAction.SELL_ALL
    if prior_held == new_held:
        return LiveAction.HOLD
    return LiveAction.SWITCH


def next_trading_session(decision_date: date, calendar: TradingCalendar) -> date | None:
    """Return the first exchange session strictly after ``decision_date`` (the fill session).

    Returns ``None`` when the calendar has no session within the lookahead horizon, so an
    unknown fill date is shown as unknown rather than guessed.
    """
    start = decision_date + timedelta(days=1)
    end = decision_date + timedelta(days=NEXT_SESSION_LOOKAHEAD_DAYS)
    sessions = calendar.sessions(start, end)
    if not sessions:
        return None
    return min(sessions)


def _validate_stop_drawdown(stop_drawdown: float) -> float:
    if isinstance(stop_drawdown, bool):
        raise ValueError(f"stop_drawdown must be in (0, 1), got {stop_drawdown!r}")
    stop = float(stop_drawdown)
    if not math.isfinite(stop) or not 0 < stop < 1:
        raise ValueError(f"stop_drawdown must be in (0, 1), got {stop_drawdown!r}")
    return stop


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def build_anchor_monitor(
    panel: pl.DataFrame,
    *,
    decision_date: date,
    anchor_tickers: Sequence[str],
    stop_drawdown: float,
) -> tuple[AnchorMonitorRow, ...]:
    """Summarize each campaign anchor's stop distance at the decision-date close.

    ``stopped`` uses the same inclusive boundary as ``post_crash_anchor_scores``
    (``drawdown_20 <= -stop_drawdown``) so the monitor never disagrees with the model.
    ``stop_close`` is the close at which that boundary is hit given today's 20-session high.
    """
    stop = _validate_stop_drawdown(stop_drawdown)
    tickers = tuple(t for t in anchor_tickers if isinstance(t, str) and t)
    if not tickers:
        return ()
    if not isinstance(panel, pl.DataFrame) or panel.height == 0:
        return ()
    if "date" not in panel.columns or "ticker" not in panel.columns:
        return ()
    if "close" not in panel.columns or "mom_20" not in panel.columns or "drawdown_20" not in panel.columns:
        return ()
    day = panel.filter(pl.col("date") == decision_date)
    if day.height == 0:
        return ()
    has_name = "name" in day.columns
    rows: list[AnchorMonitorRow] = []
    for ticker in tickers:
        frame = day.filter(pl.col("ticker") == ticker)
        if frame.height == 0:
            continue
        record = frame.row(0, named=True)
        close = _finite_float(record.get("close"))
        mom_20 = _finite_float(record.get("mom_20"))
        drawdown_20 = _finite_float(record.get("drawdown_20"))
        if close is None or mom_20 is None or drawdown_20 is None:
            continue
        if 1.0 + drawdown_20 <= 0:
            continue
        roll_max_20 = close / (1.0 + drawdown_20)
        stop_close = roll_max_20 * (1.0 - stop)
        stopped = drawdown_20 <= -stop
        name = ""
        if has_name:
            raw_name = record.get("name")
            if raw_name is not None:
                name = str(raw_name)
        rows.append(
            AnchorMonitorRow(
                ticker=ticker,
                name=name,
                close=float(close),
                mom_20=float(mom_20),
                drawdown_20=float(drawdown_20),
                roll_max_20=float(roll_max_20),
                stop_close=float(stop_close),
                stopped=bool(stopped),
            )
        )
    return tuple(rows)


def _anchor_by_ticker(anchors: tuple[AnchorMonitorRow, ...], ticker: str | None) -> AnchorMonitorRow | None:
    if ticker is None:
        return None
    for row in anchors:
        if row.ticker == ticker:
            return row
    return None


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _fmt_krw(value: float) -> str:
    return f"{float(value):,.0f}원"


def _reason_ko(
    reason: LiveReason,
    *,
    sleeve: str,
    prior_held: str | None,
    new_held: str | None,
    anchors: tuple[AnchorMonitorRow, ...],
) -> str:
    sleeve_ko = SLEEVE_KO.get(sleeve, sleeve)
    prior_row = _anchor_by_ticker(anchors, prior_held)
    new_row = _anchor_by_ticker(anchors, new_held)
    if reason is LiveReason.ANCHOR_LATCH_HOLD and new_row is not None:
        return (
            f"보유 앵커 {new_row.ticker} 유지({sleeve_ko} 국면, 래치): "
            f"mom20 {_fmt_pct(new_row.mom_20)}, 낙폭 {_fmt_pct(new_row.drawdown_20)}, "
            f"손절가 {_fmt_krw(new_row.stop_close)}"
        )
    if reason is LiveReason.ANCHOR_STOP_SWITCH and prior_row is not None and new_row is not None:
        return (
            f"보유 앵커 {prior_row.ticker} 손절(낙폭 {_fmt_pct(prior_row.drawdown_20)}, "
            f"손절가 {_fmt_krw(prior_row.stop_close)}) 후 앵커 {new_row.ticker} 교체 "
            f"({sleeve_ko} 국면, mom20 {_fmt_pct(new_row.mom_20)})"
        )
    if reason is LiveReason.ANCHOR_PICK and new_row is not None:
        return (
            f"앵커 {new_row.ticker} 신규 편입({sleeve_ko} 국면): "
            f"mom20 {_fmt_pct(new_row.mom_20)}, 낙폭 {_fmt_pct(new_row.drawdown_20)}, "
            f"손절가 {_fmt_krw(new_row.stop_close)}"
        )
    if reason is LiveReason.ANCHOR_STOP_CASH and prior_row is not None:
        return (
            f"보유 앵커 {prior_row.ticker} 손절(낙폭 {_fmt_pct(prior_row.drawdown_20)}, "
            f"손절가 {_fmt_krw(prior_row.stop_close)})로 현금 전환({sleeve_ko} 국면)"
        )
    if reason is LiveReason.NO_ANCHOR_CASH:
        return f"편입 가능한 앵커가 없어 현금 유지·전환({sleeve_ko} 국면)"
    if reason is LiveReason.MOM60_LEADER:
        return f"{sleeve_ko} 국면 60일 모멘텀 리더 {new_held} 편입"
    return f"{sleeve_ko} 국면에서 {new_held if new_held is not None else '현금'} 결정"


def explain_live_decision(
    *,
    decision_date: date,
    execution_date: date | None,
    sleeve: str,
    prior_held: str | None,
    prior_weight: float,
    new_held: str | None,
    new_weight: float,
    anchors: tuple[AnchorMonitorRow, ...],
    anchor_tickers: Sequence[str],
) -> LiveDecisionExplanation:
    """Explain a live sticky decision in operator terms without re-deciding it.

    The reason is reconstructed from the observed transition, the sleeve, and the anchor
    stop monitor. It labels the decision; it never influences it.
    """
    anchor_set = {t for t in anchor_tickers if isinstance(t, str) and t}
    prior_stopped = False
    if prior_held is not None and prior_held in anchor_set:
        prior_row = _anchor_by_ticker(anchors, prior_held)
        prior_stopped = prior_row is not None and prior_row.stopped
    if new_held is not None and new_held in anchor_set and new_held == prior_held:
        reason = LiveReason.ANCHOR_LATCH_HOLD
    elif (
        new_held is not None
        and new_held in anchor_set
        and prior_held is not None
        and prior_held in anchor_set
        and prior_stopped
    ):
        reason = LiveReason.ANCHOR_STOP_SWITCH
    elif new_held is not None and new_held in anchor_set:
        reason = LiveReason.ANCHOR_PICK
    elif new_held is None and prior_held is not None and prior_held in anchor_set and prior_stopped:
        reason = LiveReason.ANCHOR_STOP_CASH
    elif new_held is None:
        reason = LiveReason.NO_ANCHOR_CASH
    elif sleeve == "LOTTERY_ON" and new_held not in anchor_set:
        reason = LiveReason.MOM60_LEADER
    else:
        reason = LiveReason.OTHER
    action = derive_live_action(prior_held, new_held)
    return LiveDecisionExplanation(
        decision_date=decision_date,
        execution_date=execution_date,
        sleeve=sleeve,
        prior_held=prior_held,
        prior_weight=float(prior_weight),
        new_held=new_held,
        new_weight=float(new_weight),
        action=action,
        reason=reason,
        reason_ko=_reason_ko(
            reason,
            sleeve=sleeve,
            prior_held=prior_held,
            new_held=new_held,
            anchors=anchors,
        ),
        anchors=tuple(anchors),
    )


def explanation_to_payload(explanation: LiveDecisionExplanation) -> dict[str, object]:
    """Serialize the explanation into the JSON-safe artifact fields (dates ISO, floats finite)."""
    action_code = explanation.action.value
    return {
        "execution_date": explanation.execution_date.isoformat() if explanation.execution_date is not None else None,
        "action": {
            "code": action_code,
            "ko": ACTION_KO.get(action_code, action_code),
            "from": explanation.prior_held,
            "to": explanation.new_held,
        },
        "regime": {
            "sleeve": explanation.sleeve,
            "ko": SLEEVE_KO.get(explanation.sleeve, explanation.sleeve),
        },
        "reason_code": explanation.reason.value,
        "reason_ko": explanation.reason_ko,
        "anchor_monitor": [
            {
                "ticker": row.ticker,
                "name": row.name,
                "close": float(row.close),
                "mom_20": float(row.mom_20),
                "drawdown_20": float(row.drawdown_20),
                "roll_max_20": float(row.roll_max_20),
                "stop_close": float(row.stop_close),
                "stopped": bool(row.stopped),
            }
            for row in explanation.anchors
        ],
        "quantity_note_ko": (
            "est_shares는 초기자본·결정일 종가 기준 목표 수량이며, "
            "보유 유지(HOLD) 또는 현금 유지(STAY_CASH) 시 주문이 필요 없습니다."
        ),
    }


__all__ = [
    "ACTION_KO",
    "NEXT_SESSION_LOOKAHEAD_DAYS",
    "SLEEVE_KO",
    "AnchorMonitorRow",
    "LiveAction",
    "LiveDecisionExplanation",
    "LiveReason",
    "build_anchor_monitor",
    "derive_live_action",
    "explain_live_decision",
    "explanation_to_payload",
    "next_trading_session",
]
