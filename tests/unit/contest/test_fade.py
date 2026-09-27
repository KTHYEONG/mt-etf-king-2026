"""Invariant guards for the contest fade overlay daily card (HY2_FADEUP6)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.contest.daily import QuoteBar
from src.contest.fade import (
    FadeAction,
    FadeCard,
    FadeSettings,
    build_fade_card,
    decide_fade,
    fade_to_dict,
    render_fade_markdown,
)
from src.core.calendar import get_calendar

CAL = get_calendar()


def _settings(**over: Any) -> FadeSettings:
    base: dict[str, Any] = {
        "long_ticker": "0193T0",
        "inverse_ticker": "0197X0",
        "base_alias": "HY2",
        "inverse_alias": "HY2I",
        "threshold_pct": 6.0,
        "borderline_pct": 0.15,
        "mode": "close_return",
        "close_route": "continuous_1515",
        "endgame_sessions": 5,
        "output_dir": Path("results/contest_fade"),
    }
    base.update(over)
    return FadeSettings.load({"fade": base, "end_date": "2026-11-13"})


def _config(**fade_over: Any) -> dict[str, Any]:
    fade: dict[str, Any] = {
        "enabled": True,
        "long_ticker": "0193T0",
        "inverse_ticker": "0197X0",
        "base_alias": "HY2",
        "inverse_alias": "HY2I",
        "threshold_pct": 6.0,
        "borderline_pct": 0.15,
        "mode": "close_return",
        "close_route": "continuous_1515",
        "endgame_sessions": 5,
        "output_dir": "results/contest_fade",
    }
    fade.update(fade_over)
    return {"nickname": "tester", "start_date": "2026-09-21", "end_date": "2026-11-13", "fade": fade}


def _bar(ticker: str, prev: float, close: float) -> QuoteBar:
    return QuoteBar(ticker=ticker, prev_close=prev, open=prev, close=close)


def test_threshold_inclusive_triggers_fade() -> None:
    """Threshold inclusive triggers fade (change == threshold)."""
    action, borderline = decide_fade(6.0, None, _settings())
    assert action is FadeAction.FADE
    assert borderline is True


def test_below_threshold_holds_long() -> None:
    """Below threshold holds long."""
    action, _ = decide_fade(5.99, None, _settings())
    assert action is FadeAction.HOLD_LONG


def test_large_down_move_never_fades() -> None:
    """Large down move never fades."""
    action, borderline = decide_fade(-12.0, None, _settings())
    assert action is FadeAction.HOLD_LONG
    assert borderline is False


def test_borderline_flag_does_not_change_action() -> None:
    """Borderline flag does not change action."""
    action, borderline = decide_fade(5.90, None, _settings())
    assert action is FadeAction.HOLD_LONG
    assert borderline is True


def test_missing_change_is_fail_closed() -> None:
    """Missing change is fail-closed."""
    action, _ = decide_fade(None, None, _settings())
    assert action is FadeAction.NO_DATA


def test_non_finite_change_is_fail_closed() -> None:
    """Non-finite change is fail-closed."""
    action, _ = decide_fade(float("nan"), None, _settings())
    assert action is FadeAction.NO_DATA


def test_open_only_continuity_matrix() -> None:
    """Open-only continuity matrix (KEEP/RETURN/FADE/HOLD)."""
    settings = _settings(mode="open_only")
    assert decide_fade(7.0, 7.0, settings)[0] is FadeAction.KEEP_INVERSE
    assert decide_fade(2.0, 7.0, settings)[0] is FadeAction.RETURN_LONG
    assert decide_fade(7.0, 2.0, settings)[0] is FadeAction.FADE
    assert decide_fade(2.0, 2.0, settings)[0] is FadeAction.HOLD_LONG


def test_open_only_unknown_previous_assumes_long_with_warning() -> None:
    """Open-only unknown previous assumes long with warning."""
    config = _config(mode="open_only")
    session = date(2026, 9, 23)
    bars = {"0193T0": _bar("0193T0", 100.0, 107.0)}
    card = build_fade_card(session, CAL, bars, None, config, "HY2")
    assert card.action is FadeAction.FADE
    assert "PREV_UNKNOWN" in card.warnings


def test_card_change_computed_from_quote_bar() -> None:
    """Card change computed from quote bar (11550 -> 11865)."""
    config = _config()
    session = date(2026, 9, 23)
    bars = {"0193T0": _bar("0193T0", 11550.0, 11865.0)}
    card = build_fade_card(session, CAL, bars, None, config, "HY2")
    assert card.long_change_pct is not None
    assert abs(card.long_change_pct - 2.727) < 1e-3
    assert card.action is FadeAction.HOLD_LONG
    assert card.execution_session == date(2026, 9, 28)


def test_missing_long_bar_yields_no_data_card() -> None:
    """Missing long bar yields NO_DATA card."""
    config = _config()
    session = date(2026, 9, 23)
    card = build_fade_card(session, CAL, {}, None, config, "HY2")
    assert card.action is FadeAction.NO_DATA
    assert "QUOTES_MISSING" in card.warnings
    assert card.execution_session is not None


def test_contest_end_yields_contest_over() -> None:
    """Contest end yields CONTEST_OVER even when the state gate also mismatches."""
    config = _config()
    card = build_fade_card(date(2026, 11, 13), CAL, {"0193T0": _bar("0193T0", 100.0, 107.0)}, None, config, "SEMI2")
    assert card.action is FadeAction.CONTEST_OVER
    assert card.execution_session is None


def test_endgame_window_flagged() -> None:
    """Endgame window flagged when few sessions remain after execution."""
    config = _config()
    sessions = CAL.sessions(date(2026, 9, 21), date(2026, 11, 13))
    session = sessions[-3]
    card = build_fade_card(session, CAL, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, "HY2")
    assert card.endgame is True
    assert "ENDGAME_REVIEW" in card.warnings


def test_fade_checklist_timing_continuous_route() -> None:
    """Fade checklist timing continuous route lists windows in order."""
    card = FadeCard(
        session=date(2026, 9, 23),
        execution_session=date(2026, 9, 28),
        action=FadeAction.FADE,
        long_change_pct=7.0,
        prev_long_change_pct=None,
        threshold_pct=6.0,
        borderline=False,
        mode="close_return",
        close_route="continuous_1515",
        sessions_remaining_after_execution=20,
        endgame=False,
        warnings=(),
    )
    text = render_fade_markdown(card)
    assert text.index("08:40–08:59") < text.index("09:00–09:02") < text.index("15:15–15:19")  # noqa: RUF001
    assert "15:20 이후 주문 금지" in text
    assert "0193T0" in text and "0197X0" in text


def test_fade_checklist_exact_close_route() -> None:
    """Fade checklist exact-close route uses auction plus after-hours."""
    card = FadeCard(
        session=date(2026, 9, 23),
        execution_session=date(2026, 9, 28),
        action=FadeAction.FADE,
        long_change_pct=7.0,
        prev_long_change_pct=None,
        threshold_pct=6.0,
        borderline=False,
        mode="close_return",
        close_route="auction_plus_afterhours",
        sessions_remaining_after_execution=20,
        endgame=False,
        warnings=(),
    )
    text = render_fade_markdown(card)
    assert "15:20–15:29" in text  # noqa: RUF001
    assert "15:40–16:00" in text  # noqa: RUF001
    assert "15:15–15:19" not in text  # noqa: RUF001


def test_hold_card_has_no_order_windows() -> None:
    """Hold card has no order windows."""
    card = FadeCard(
        session=date(2026, 9, 23),
        execution_session=date(2026, 9, 28),
        action=FadeAction.HOLD_LONG,
        long_change_pct=2.0,
        prev_long_change_pct=None,
        threshold_pct=6.0,
        borderline=False,
        mode="close_return",
        close_route="continuous_1515",
        sessions_remaining_after_execution=20,
        endgame=False,
        warnings=(),
    )
    text = render_fade_markdown(card)
    assert "주문 없음" in text
    assert "08:40" not in text
    assert "15:15" not in text


def test_settings_validation_rejects_bad_literals() -> None:
    """Settings validation rejects bad literals."""
    with pytest.raises(ValueError, match="mode"):
        _settings(mode="weekly")
    with pytest.raises(ValueError, match="borderline"):
        _settings(borderline_pct=6.0)
    with pytest.raises(ValueError, match="close_route"):
        _settings(close_route="at_close")
    with pytest.raises(ValueError, match="threshold"):
        _settings(threshold_pct=0.0)
    with pytest.raises(KeyError, match="fade"):
        FadeSettings.load({"end_date": "2026-11-13"})


def test_fade_round_trip_dict() -> None:
    """fade_to_dict emits JSON-ready primitives."""
    config = _config()
    card = build_fade_card(date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 107.0)}, None, config, "HY2")
    payload = fade_to_dict(card)
    assert payload["session"] == "2026-09-23"
    assert payload["action"] == card.action.value
    assert isinstance(payload["warnings"], list)


def test_settings_rejects_non_mapping_fade_block() -> None:
    """Non-mapping fade block raises KeyError."""
    with pytest.raises(KeyError, match="fade"):
        FadeSettings.load({"fade": ["not-a-mapping"], "end_date": "2026-11-13"})


def test_settings_rejects_bool_and_non_positive_endgame() -> None:
    """Bool or non-positive endgame_sessions raises ValueError."""
    with pytest.raises(ValueError, match="endgame"):
        _settings(endgame_sessions=True)
    with pytest.raises(ValueError, match="endgame"):
        _settings(endgame_sessions=0)


def test_invalid_string_change_is_fail_closed() -> None:
    """Unparseable change string is fail-closed."""
    action, borderline = decide_fade("bad", None, _settings())  # type: ignore[arg-type]
    assert action is FadeAction.NO_DATA
    assert borderline is False


def test_non_positive_bar_yields_no_data() -> None:
    """Non-positive long bar yields NO_DATA."""
    config = _config()
    card = build_fade_card(date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 0.0, 100.0)}, None, config, "HY2")
    assert card.action is FadeAction.NO_DATA
    assert "QUOTES_MISSING" in card.warnings


def test_non_finite_bar_yields_no_data() -> None:
    """Non-finite long bar yields NO_DATA."""
    config = _config()
    card = build_fade_card(date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, float("inf"))}, None, config, "HY2")
    assert card.action is FadeAction.NO_DATA
    assert "QUOTES_MISSING" in card.warnings


class _StubCalendar:
    def __init__(self, nxt: date | None = None, raise_next: bool = False) -> None:
        self._nxt = nxt
        self._raise_next = raise_next

    def next_session(self, day: date) -> date:
        if self._raise_next:
            raise ValueError("no next session")
        assert self._nxt is not None
        return self._nxt

    def sessions(self, start: date, end: date) -> list[date]:
        return CAL.sessions(start, end)


def test_calendar_without_next_session_yields_none_execution() -> None:
    """Calendar without next session yields None execution."""
    config = _config()
    stub = _StubCalendar(raise_next=True)
    card = build_fade_card(date(2026, 9, 23), stub, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, "HY2")
    assert card.execution_session is None
    assert card.sessions_remaining_after_execution == 0
    assert card.endgame is True


def test_execution_beyond_end_date_leaves_no_remaining() -> None:
    """Execution beyond end_date leaves no remaining sessions."""
    config = _config()
    stub = _StubCalendar(nxt=date(2026, 12, 1))
    card = build_fade_card(date(2026, 9, 23), stub, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, "HY2")
    assert card.execution_session == date(2026, 12, 1)
    assert card.sessions_remaining_after_execution == 0
    assert card.endgame is True


def test_borderline_build_adds_warning_and_renders_hts_line() -> None:
    """Borderline build adds BORDERLINE warning and renders the HTS line."""
    config = _config()
    card = build_fade_card(date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 106.05)}, None, config, "HY2")
    assert card.action is FadeAction.FADE
    assert card.borderline is True
    assert "BORDERLINE" in card.warnings
    assert "HTS" in render_fade_markdown(card)


def test_open_only_unknown_previous_renders_hts_check() -> None:
    """Open-only unknown previous renders the HTS holding check."""
    config = _config(mode="open_only")
    card = build_fade_card(date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 107.0)}, None, config, "HY2")
    assert "PREV_UNKNOWN" in card.warnings
    assert "HTS" in render_fade_markdown(card)


def test_endgame_card_renders_review_line() -> None:
    """Endgame card renders the manual review line."""
    config = _config()
    sessions = CAL.sessions(date(2026, 9, 21), date(2026, 11, 13))
    card = build_fade_card(sessions[-3], CAL, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, "HY2")
    assert "ENDGAME_REVIEW" in card.warnings
    assert "ENDGAME_REVIEW" in render_fade_markdown(card)


def test_open_only_missing_bar_assumes_long() -> None:
    """Open-only missing bar assumes long with PREV_UNKNOWN."""
    config = _config(mode="open_only")
    card = build_fade_card(date(2026, 9, 23), CAL, {}, None, config, "HY2")
    assert card.action is FadeAction.NO_DATA
    assert "QUOTES_MISSING" in card.warnings
    assert "PREV_UNKNOWN" in card.warnings


def test_missing_bar_at_endgame_adds_review_warning() -> None:
    """Missing bar at endgame adds ENDGAME_REVIEW."""
    config = _config()
    sessions = CAL.sessions(date(2026, 9, 21), date(2026, 11, 13))
    card = build_fade_card(sessions[-3], CAL, {}, None, config, "HY2")
    assert card.action is FadeAction.NO_DATA
    assert "QUOTES_MISSING" in card.warnings
    assert "ENDGAME_REVIEW" in card.warnings


def test_invalid_prev_bars_yield_none_prev_change() -> None:
    """Invalid prev bars yield None prev change."""
    from src.contest.fade import _bar_change

    assert _bar_change(None) is None
    assert _bar_change(_bar("0193T0", 0.0, 100.0)) is None
    assert _bar_change(_bar("0193T0", 100.0, float("inf"))) is None
    config = _config()
    card = build_fade_card(
        date(2026, 9, 23),
        CAL,
        {"0193T0": _bar("0193T0", 100.0, 101.0)},
        {"0193T0": _bar("0193T0", 0.0, 100.0)},
        config,
        "HY2",
    )
    assert card.prev_long_change_pct is None


def _render_card(action: FadeAction, **over: Any) -> str:
    base: dict[str, Any] = {
        "session": date(2026, 9, 23),
        "execution_session": date(2026, 9, 28),
        "action": action,
        "long_change_pct": 2.0,
        "prev_long_change_pct": None,
        "threshold_pct": 6.0,
        "borderline": False,
        "mode": "close_return",
        "close_route": "continuous_1515",
        "sessions_remaining_after_execution": 20,
        "endgame": False,
        "warnings": (),
    }
    base.update(over)
    return render_fade_markdown(FadeCard(**base))  # type: ignore[arg-type]


def test_return_long_checklist_reverses_tickers() -> None:
    """Return-long checklist reverses tickers with open windows."""
    config = _config(mode="open_only")
    card = build_fade_card(
        date(2026, 9, 23),
        CAL,
        {"0193T0": _bar("0193T0", 100.0, 102.0)},
        {"0193T0": _bar("0193T0", 100.0, 107.0)},
        config,
        "HY2",
    )
    assert card.action is FadeAction.RETURN_LONG
    text = render_fade_markdown(card)
    assert "08:40" in text and "0193T0" in text and "0197X0" in text


def test_keep_inverse_no_data_and_over_cards_render() -> None:
    """Keep-inverse, no-data and contest-over cards render without order windows."""
    keep = _render_card(FadeAction.KEEP_INVERSE)
    assert "주문 없음" in keep and "HY2I" in keep
    nodata = _render_card(FadeAction.NO_DATA, long_change_pct=None)
    assert "주문 없음" in nodata and "HTS" in nodata
    over = _render_card(FadeAction.CONTEST_OVER, execution_session=None)
    assert "주문 없음" in over
    assert "08:40" not in keep and "08:40" not in nodata and "08:40" not in over


def test_state_mismatch_forces_inactive_regardless_of_price() -> None:
    """State mismatch forces INACTIVE regardless of price."""
    config = _config()
    card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 112.0)}, None, config, "SEMI2"
    )
    assert card.action is FadeAction.INACTIVE
    assert card.long_change_pct is None
    assert card.prev_long_change_pct is None
    assert "STATE_NOT_BASE" in card.warnings
    assert "STATE_UNKNOWN" not in card.warnings
    assert card.state_alias == "SEMI2"


def test_unknown_state_forces_inactive() -> None:
    """Unknown state forces INACTIVE."""
    config = _config()
    card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 112.0)}, None, config, None
    )
    assert card.action is FadeAction.INACTIVE
    assert card.long_change_pct is None
    assert "STATE_UNKNOWN" in card.warnings
    assert "STATE_NOT_BASE" not in card.warnings
    assert card.state_alias is None


def test_state_on_inverse_alias_is_still_inactive() -> None:
    """State on inverse alias is still INACTIVE."""
    config = _config()
    card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 112.0)}, None, config, "HY2I"
    )
    assert card.action is FadeAction.INACTIVE
    assert "STATE_NOT_BASE" in card.warnings


def test_state_on_base_alias_preserves_prior_behavior() -> None:
    """State on base alias behaves exactly as before."""
    config = _config()
    fade_card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 106.0)}, None, config, "HY2"
    )
    assert fade_card.action is FadeAction.FADE
    assert fade_card.state_alias == "HY2"
    hold_card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, "HY2"
    )
    assert hold_card.action is FadeAction.HOLD_LONG
    missing_card = build_fade_card(date(2026, 9, 23), CAL, {}, None, config, "HY2")
    assert missing_card.action is FadeAction.NO_DATA
    assert "QUOTES_MISSING" in missing_card.warnings


def test_endgame_flag_still_computed_while_inactive() -> None:
    """Endgame flag still computed while INACTIVE."""
    config = _config()
    sessions = CAL.sessions(date(2026, 9, 21), date(2026, 11, 13))
    card = build_fade_card(
        sessions[-3], CAL, {"0193T0": _bar("0193T0", 100.0, 112.0)}, None, config, "SEMI2"
    )
    assert card.action is FadeAction.INACTIVE
    assert card.endgame is True
    assert "ENDGAME_REVIEW" in card.warnings
    assert "STATE_NOT_BASE" in card.warnings


def test_inactive_card_renders_no_order_checklist() -> None:
    """INACTIVE card renders no order checklist and names both aliases."""
    config = _config()
    card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 112.0)}, None, config, "SEMI2"
    )
    text = render_fade_markdown(card, "HY2")
    assert "08:40" not in text
    assert "15:15" not in text
    assert "09:00" not in text
    assert "SEMI2" in text
    assert "HY2" in text
    assert "주간 카드" in text


def test_inactive_unknown_state_render_names_absence() -> None:
    """INACTIVE card with unknown state renders the absence plainly."""
    config = _config()
    card = build_fade_card(date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, None)
    text = render_fade_markdown(card, "HY2")
    assert "08:40" not in text and "15:20" not in text and "15:40" not in text
    assert "확인 불가" in text
    assert "HY2" in text


def test_settings_validation_rejects_blank_or_equal_aliases() -> None:
    """Settings validation rejects blank or equal aliases."""
    with pytest.raises(ValueError, match=r"base_alias/inverse_alias"):
        _settings(base_alias="")
    with pytest.raises(ValueError, match=r"base_alias/inverse_alias"):
        _settings(inverse_alias="   ")
    with pytest.raises(ValueError, match=r"must differ"):
        _settings(base_alias="HY2", inverse_alias="HY2")


def test_dict_round_trip_carries_state_alias() -> None:
    """Dict round-trip carries state_alias."""
    config = _config()
    card = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 107.0)}, None, config, "HY2"
    )
    payload = fade_to_dict(card)
    assert payload["state_alias"] == "HY2"


def test_every_card_renders_weekly_holding_line() -> None:
    """Every card renders the weekly-recommended alias gate line."""
    config = _config()
    gated = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, "HY2"
    )
    assert "이번 주 보유(주간 카드 기준): HY2" in render_fade_markdown(gated, "HY2")
    unknown = build_fade_card(
        date(2026, 9, 23), CAL, {"0193T0": _bar("0193T0", 100.0, 101.0)}, None, config, None
    )
    assert "확인 불가" in render_fade_markdown(unknown, "HY2")
