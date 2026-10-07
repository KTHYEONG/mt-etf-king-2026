"""Daily segment policy for the contest (overnight long + intraday reversal).

The account holds the long single-stock 2x vehicle across every overnight gap. For the next session's continuous trading
window it either stays long or holds the inverse 2x vehicle, decided by a majority vote of prior-session close-to-close
moves across the vote vehicles: intraday returns of the semiconductor complex reverse the prior day's move, while the
overnight segment carries the positive drift. Being fully invested in every segment is deliberate: the tournament objective
is the probability of finishing at or above the prize line, which needs the variance of a 2x exposure at all times.

Pure functions only. Fail-closed: any missing or non-finite vote input yields NO_DATA, which means "no orders, keep the
long vehicle".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, time
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from src.contest.daily import QuoteBar
from src.contest.leaderboard import LeaderboardSnapshot

# Auction-boundary limits in KST: KRX continuous trading runs 09:00-15:30, but the opening auction runs
# 08:30-09:00 and closing auction 15:20-15:30. ETFs deviate from NAV in auction prints, so orders are restricted
# to continuous windows starting at or after 09:01 and ending at or before 15:19.
_CONTINUOUS_SESSION_START: Final[time] = time(9, 1)
_CONTINUOUS_SESSION_END: Final[time] = time(15, 19)


class IntradaySide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class SegmentAction(StrEnum):
    HOLD_LONG = "HOLD_LONG"
    SWITCH_SHORT = "SWITCH_SHORT"
    NO_DATA = "NO_DATA"
    CONTEST_OVER = "CONTEST_OVER"


class EquitySource(StrEnum):
    USER = "USER"
    LEADERBOARD = "LEADERBOARD"
    LEDGER = "LEDGER"


@dataclass(frozen=True)
class TradingWindow:
    """KST wall-clock window inside the continuous session in which the operator places market orders."""

    start: time
    end: time


def _parse_window(raw: Any) -> TradingWindow:
    if not isinstance(raw, str):
        raise ValueError(f"window must be 'HH:MM-HH:MM' string, got {raw!r}")
    parts = raw.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"malformed window string {raw!r}")
    s_str, e_str = parts[0].strip(), parts[1].strip()
    if len(s_str) != 5 or s_str[2] != ":" or len(e_str) != 5 or e_str[2] != ":":
        raise ValueError(f"malformed window format in {raw!r}, expected HH:MM-HH:MM")
    try:
        start = time.fromisoformat(s_str)
        end = time.fromisoformat(e_str)
    except ValueError as exc:
        raise ValueError(f"malformed window time in {raw!r}") from exc
    if start >= end:
        raise ValueError(f"window start {start} must precede end {end}")
    if start < _CONTINUOUS_SESSION_START or end > _CONTINUOUS_SESSION_END:
        raise ValueError(
            f"window {raw!r} outside continuous session limits ({_CONTINUOUS_SESSION_START}-{_CONTINUOUS_SESSION_END})"
        )
    return TradingWindow(start=start, end=end)


@dataclass(frozen=True)
class SegmentSettings:
    long_ticker: str
    long_alias: str
    inverse_ticker: str
    inverse_alias: str
    vote: tuple[tuple[str, str], ...]
    vote_min_down: int
    open_window: TradingWindow
    close_window: TradingWindow
    cost_bps_round_trip: float
    prize_rank: int
    endgame_sessions: int
    ledger_state_name: str
    output_dir: Path

    @classmethod
    def load(cls, contest: Mapping[str, Any]) -> SegmentSettings:
        """Validate `contest["segment"]`.

        Windows are `"HH:MM-HH:MM"` strings. Both must lie inside the continuous session and outside the opening and
        closing auctions, because auction prints of these ETFs deviate from NAV by percent-level amounts.

        Raises:
            KeyError: the segment block (or a non-mapping block) or a required key is missing.
            ValueError: blank/equal tickers or aliases; empty or duplicate vote tickers; `vote_min_down` not an int in
                [1, len(vote)] (bool rejected); malformed window strings; a window with start >= end; a window
                starting before 09:01 or ending after 15:19; open_window.end > close_window.start; non-finite or
                negative `cost_bps_round_trip` or one >= 100; `prize_rank` / `endgame_sessions` not a positive int
                (bool rejected); blank `ledger_state_name` or `output_dir`.
        """
        if not isinstance(contest, Mapping) or "segment" not in contest:
            raise KeyError("missing or non-mapping 'segment' block")
        seg = contest["segment"]
        if not isinstance(seg, Mapping):
            raise KeyError("non-mapping 'segment' block")

        for k in (
            "long_ticker",
            "long_alias",
            "inverse_ticker",
            "inverse_alias",
            "vote",
            "vote_min_down",
            "open_window",
            "close_window",
            "cost_bps_round_trip",
            "prize_rank",
            "endgame_sessions",
            "ledger_state_name",
            "output_dir",
        ):
            if k not in seg:
                raise KeyError(f"missing required key {k!r}")

        long_ticker, long_alias = str(seg["long_ticker"]).strip(), str(seg["long_alias"]).strip()
        inverse_ticker, inverse_alias = str(seg["inverse_ticker"]).strip(), str(seg["inverse_alias"]).strip()

        if not (long_ticker and long_alias and inverse_ticker and inverse_alias):
            raise ValueError("tickers and aliases must not be blank")
        if long_ticker == inverse_ticker:
            raise ValueError(f"long_ticker and inverse_ticker must differ, got {long_ticker!r}")
        if long_alias == inverse_alias:
            raise ValueError(f"long_alias and inverse_alias must differ, got {long_alias!r}")

        vote_raw = seg["vote"]
        if not isinstance(vote_raw, Mapping) or not vote_raw:
            raise ValueError("vote must be a non-empty mapping of alias to ticker")

        vote_list: list[tuple[str, str]] = []
        for a_raw, t_raw in vote_raw.items():
            a, t = str(a_raw).strip(), str(t_raw).strip()
            if not a or not t:
                raise ValueError("vote aliases and tickers must not be blank")
            vote_list.append((a, t))

        if len(vote_list) != len({t for _, t in vote_list}):
            raise ValueError("duplicate tickers in vote mapping")
        vote_tuple = tuple(vote_list)

        vmd = seg["vote_min_down"]
        if isinstance(vmd, bool) or not isinstance(vmd, int) or not (1 <= vmd <= len(vote_tuple)):
            raise ValueError(f"vote_min_down must be an int in [1, {len(vote_tuple)}], got {vmd!r}")

        open_win, close_win = _parse_window(seg["open_window"]), _parse_window(seg["close_window"])
        if open_win.end > close_win.start:
            raise ValueError(f"open_window.end ({open_win.end}) > close_window.start ({close_win.start})")

        cost_raw = seg["cost_bps_round_trip"]
        if isinstance(cost_raw, bool) or not isinstance(cost_raw, (int, float)):
            raise ValueError("cost_bps_round_trip must be numeric")
        cost = float(cost_raw)
        if not math.isfinite(cost) or cost < 0.0 or cost >= 100.0:
            raise ValueError(f"cost_bps_round_trip must be finite and in [0, 100), got {cost}")

        pr, es = seg["prize_rank"], seg["endgame_sessions"]
        for name, val in (("prize_rank", pr), ("endgame_sessions", es)):
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive int, got {val!r}")

        ledger_name, out_dir_str = str(seg["ledger_state_name"]).strip(), str(seg["output_dir"]).strip()
        if not ledger_name or not out_dir_str:
            raise ValueError("ledger_state_name and output_dir must not be blank")
        output_dir = Path(out_dir_str)

        return cls(
            long_ticker=long_ticker,
            long_alias=long_alias,
            inverse_ticker=inverse_ticker,
            inverse_alias=inverse_alias,
            vote=vote_tuple,
            vote_min_down=vmd,
            open_window=open_win,
            close_window=close_win,
            cost_bps_round_trip=cost,
            prize_rank=pr,
            endgame_sessions=es,
            ledger_state_name=ledger_name,
            output_dir=output_dir,
        )


@dataclass(frozen=True)
class LedgerState:
    """Our account equity (multiple of initial capital) at the close of `as_of`."""

    as_of: date
    equity: float
    source: EquitySource


@dataclass(frozen=True)
class Standing:
    """Where we stand against the prize line at the close of the signal session.

    `required_multiple` = prize-line equity / our equity, i.e. the growth needed from here to reach today's prize line.
    It is set only when both values are dated the signal session.
    """

    snapshot_date: date | None
    our_equity: float | None
    our_source: EquitySource | None
    our_as_of: date | None
    prize_line_equity: float | None
    leader_equity: float | None
    required_multiple: float | None


@dataclass(frozen=True)
class SegmentCard:
    session: date
    execution_session: date | None
    final_session: bool
    action: SegmentAction
    intraday_side: IntradaySide | None
    vote_changes: tuple[tuple[str, float | None], ...]
    down_votes: int
    vote_min_down: int
    sessions_remaining_after_execution: int
    endgame: bool
    standing: Standing
    warnings: tuple[str, ...]


def bar_change_pct(bar: QuoteBar | None) -> float | None:
    """Close-to-close change of a quote bar in percent; None when the bar is absent or any price is non-positive or
    non-finite."""
    if bar is None:
        return None
    for p in (bar.prev_close, bar.open, bar.close):
        if not (math.isfinite(p) and p > 0):
            return None
    change = (bar.close / bar.prev_close - 1.0) * 100.0
    return change if math.isfinite(change) else None


def decide_intraday_side(
    changes_pct: Mapping[str, float | None], settings: SegmentSettings
) -> tuple[IntradaySide | None, int]:
    """Majority-vote reversal rule for the next session's intraday segment.

    A vote vehicle that closed down (change < 0) votes for an intraday long; flat or up votes for an intraday short. The
    side is LONG when the down votes reach `vote_min_down`, else SHORT. Flat counts as not-down to match the validated
    rule; the vote across several semiconductor/market vehicles damps single-name noise in the reversal signal.

    Args:
        changes_pct: close-to-close change in percent keyed by vote ticker (missing keys allowed).
        settings: validated settings.

    Returns:
        (side, down_votes). side is None when any vote ticker is missing or non-finite (fail-closed); down_votes then
        counts only the valid down votes.
    """
    down_votes = 0
    all_valid = True
    for _alias, ticker in settings.vote:
        if ticker not in changes_pct:
            all_valid = False
            continue
        val = changes_pct[ticker]
        if val is None or not math.isfinite(val):
            all_valid = False
            continue
        if val < 0.0:
            down_votes += 1

    if not all_valid:
        return None, down_votes

    side = IntradaySide.LONG if down_votes >= settings.vote_min_down else IntradaySide.SHORT
    return side, down_votes


def build_segment_card(
    session: date,
    calendar: Any,
    bars: Mapping[str, QuoteBar],
    config: Mapping[str, Any],
    standing: Standing,
    extra_warnings: Sequence[str] = (),
) -> SegmentCard:
    """Assemble the card that governs the intraday segment of the session after `session`.

    Args:
        session: signal session; its bars supply the vote.
        calendar: trading calendar exposing `next_session(day)` and `sessions(start, end)`.
        bars: quote bars of `session` keyed by ticker.
        config: the `contest` mapping (needs `end_date` and the `segment` block).
        standing: precomputed standing for `session` (see `compute_standing`).
        extra_warnings: warnings produced upstream (ledger/standing) appended after the card's own warnings.

    Returns:
        CONTEST_OVER when `session >= end_date` or the next session lies after `end_date`; NO_DATA with
        `CALENDAR_UNAVAILABLE` when the calendar cannot supply the next session; NO_DATA with `QUOTES_MISSING` when the
        vote is incomplete; otherwise HOLD_LONG (side LONG) or SWITCH_SHORT (side SHORT).
    """
    settings = SegmentSettings.load(config)
    end_date_val = config["end_date"]
    end_date = date.fromisoformat(end_date_val) if isinstance(end_date_val, str) else end_date_val

    def _make_warnings(card_w: list[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        res: list[str] = []
        for w in card_w + list(extra_warnings):
            if w not in seen:
                seen.add(w)
                res.append(w)
        return tuple(res)

    def _contest_over() -> SegmentCard:
        return SegmentCard(
            session=session,
            execution_session=None,
            final_session=False,
            action=SegmentAction.CONTEST_OVER,
            intraday_side=None,
            vote_changes=tuple((alias, None) for alias, _ in settings.vote),
            down_votes=0,
            vote_min_down=settings.vote_min_down,
            sessions_remaining_after_execution=0,
            endgame=False,
            standing=standing,
            warnings=_make_warnings([]),
        )

    if session >= end_date:
        return _contest_over()

    try:
        next_session = calendar.next_session(session)
        remaining_days = calendar.sessions(next_session, end_date) if next_session < end_date else []
    except Exception:
        vote_data = tuple((alias, bar_change_pct(bars.get(ticker))) for alias, ticker in settings.vote)
        _, dv = decide_intraday_side({t: c for (_, t), (_, c) in zip(settings.vote, vote_data, strict=True)}, settings)
        return SegmentCard(
            session=session,
            execution_session=None,
            final_session=False,
            action=SegmentAction.NO_DATA,
            intraday_side=None,
            vote_changes=vote_data,
            down_votes=dv,
            vote_min_down=settings.vote_min_down,
            sessions_remaining_after_execution=0,
            endgame=False,
            standing=standing,
            warnings=_make_warnings(["CALENDAR_UNAVAILABLE"]),
        )

    if next_session > end_date:
        return _contest_over()

    execution_session = next_session
    final_session = execution_session == end_date

    sessions_remaining = sum(1 for s in remaining_days if s > execution_session)
    endgame = sessions_remaining < settings.endgame_sessions

    vote_changes = tuple((alias, bar_change_pct(bars.get(ticker))) for alias, ticker in settings.vote)
    changes_pct = {t: c for (_, t), (_, c) in zip(settings.vote, vote_changes, strict=True)}
    side, down_votes = decide_intraday_side(changes_pct, settings)

    card_warnings: list[str] = []
    if side is None:
        action = SegmentAction.NO_DATA
        card_warnings.append("QUOTES_MISSING")
    elif side == IntradaySide.LONG:
        action = SegmentAction.HOLD_LONG
    else:
        action = SegmentAction.SWITCH_SHORT

    if final_session:
        card_warnings.append("FINAL_SESSION")
    if endgame:
        card_warnings.append("ENDGAME_REVIEW")

    return SegmentCard(
        session=session,
        execution_session=execution_session,
        final_session=final_session,
        action=action,
        intraday_side=side if action != SegmentAction.NO_DATA else None,
        vote_changes=vote_changes,
        down_votes=down_votes,
        vote_min_down=settings.vote_min_down,
        sessions_remaining_after_execution=sessions_remaining,
        endgame=endgame,
        standing=standing,
        warnings=_make_warnings(card_warnings),
    )


def executed_side_from_card(card: Mapping[str, Any] | None) -> IntradaySide | None:
    """The intraday side actually governed by a persisted card (the card of the previous signal session).

    HOLD_LONG and NO_DATA map to LONG (no orders were due, the long vehicle stayed held); SWITCH_SHORT maps to SHORT.
    CONTEST_OVER, a missing card, or an unparseable action yields None so the ledger refuses to guess.
    """
    if card is None or not isinstance(card, Mapping):
        return None
    action = card.get("action")
    if action in (SegmentAction.HOLD_LONG, SegmentAction.NO_DATA):
        return IntradaySide.LONG
    if action == SegmentAction.SWITCH_SHORT:
        return IntradaySide.SHORT
    return None


def session_policy_multiple(
    bars: Mapping[str, QuoteBar], side: IntradaySide, session_is_final: bool, settings: SegmentSettings
) -> float | None:
    """Estimated equity multiple of one executed session under the segment policy.

    Overnight leg: long vehicle from the previous close to the official open. Intraday leg: the held vehicle from the
    official open to the official close. Real switches happen inside the configured windows, so this is an estimate that
    any leaderboard or user anchor supersedes. Costs: a SHORT session pays two round trips (long sold and rebought, inverse
    bought and sold), one on the final session (the inverse is valued at the close, not sold); LONG pays none.

    Returns:
        The multiple, or None when a needed bar (long always, inverse on SHORT) is absent, non-positive or non-finite.
    """
    long_bar = bars.get(settings.long_ticker)
    if long_bar is None or not all(
        math.isfinite(p) and p > 0 for p in (long_bar.prev_close, long_bar.open, long_bar.close)
    ):
        return None

    overnight_return = long_bar.open / long_bar.prev_close

    if side == IntradaySide.LONG:
        return overnight_return * (long_bar.close / long_bar.open)

    inv_bar = bars.get(settings.inverse_ticker)
    if inv_bar is None or not all(
        math.isfinite(p) and p > 0 for p in (inv_bar.prev_close, inv_bar.open, inv_bar.close)
    ):
        return None

    round_trips = 1 if session_is_final else 2
    cost_factor = 1.0 - (settings.cost_bps_round_trip / 10000.0) * round_trips
    return overnight_return * (inv_bar.close / inv_bar.open) * cost_factor


def advance_ledger(
    prev: LedgerState | None,
    session: date,
    prev_session: date,
    executed_side: IntradaySide | None,
    bars: Mapping[str, QuoteBar],
    session_is_final: bool,
    leaderboard_equity: float | None,
    user_equity: float | None,
    settings: SegmentSettings,
) -> tuple[LedgerState | None, tuple[str, ...]]:
    """Advance our equity estimate to the close of `session`.

    Precedence: a valid user anchor, then our leaderboard row, then compounding the previous state by the session's
    policy multiple. Anchors exist because we sit outside the public top-N and the multiple is only an estimate.

    Returns:
        (state, warnings). Warnings: `USER_EQUITY_INVALID` (non-finite or non-positive user anchor, ignored),
        `LEDGER_AHEAD` (session earlier than the stored state; state returned unchanged), `EQUITY_UNKNOWN` (no anchor
        and no previous state), `LEDGER_STALE` (previous state not dated `prev_session`, side unknown, or bars invalid;
        state returned unchanged).
    """
    if prev is not None and prev.as_of > session:
        return prev, ("LEDGER_AHEAD",)

    user_invalid = False
    valid_user = False
    if user_equity is not None:
        if (
            not isinstance(user_equity, bool)
            and isinstance(user_equity, (int, float))
            and math.isfinite(user_equity)
            and user_equity > 0
        ):
            valid_user = True
        else:
            user_invalid = True

    warn_list: list[str] = []
    if user_invalid:
        warn_list.append("USER_EQUITY_INVALID")

    if valid_user and user_equity is not None:
        return LedgerState(as_of=session, equity=float(user_equity), source=EquitySource.USER), tuple(warn_list)

    if (
        leaderboard_equity is not None
        and not isinstance(leaderboard_equity, bool)
        and math.isfinite(leaderboard_equity)
        and leaderboard_equity > 0
    ):
        state = LedgerState(as_of=session, equity=float(leaderboard_equity), source=EquitySource.LEADERBOARD)
        return state, tuple(warn_list)

    if prev is None:
        warn_list.append("EQUITY_UNKNOWN")
        return None, tuple(warn_list)

    if prev.as_of == session:
        return prev, tuple(warn_list)

    if prev.as_of != prev_session or executed_side is None:
        warn_list.append("LEDGER_STALE")
        return prev, tuple(warn_list)

    mult = session_policy_multiple(bars, executed_side, session_is_final, settings)
    if mult is None:
        warn_list.append("LEDGER_STALE")
        return prev, tuple(warn_list)

    new_equity = prev.equity * mult
    state = LedgerState(as_of=session, equity=new_equity, source=EquitySource.LEDGER)
    return state, tuple(warn_list)


def leaderboard_equity(snapshot: LeaderboardSnapshot | None, session: date, nickname: str) -> float | None:
    """Our equity (1 + total return) from the snapshot, only when it is dated `session` and lists `nickname`."""
    if snapshot is None or snapshot.base_date != session:
        return None
    entry = snapshot.entry_for(nickname)
    if entry is None or not math.isfinite(entry.total_return_pct):
        return None
    equity = 1.0 + entry.total_return_pct / 100.0
    return equity if math.isfinite(equity) and equity > 0 else None


def compute_standing(
    snapshot: LeaderboardSnapshot | None,
    session: date,
    nickname: str,
    ledger: LedgerState | None,
    prize_rank: int,
) -> tuple[Standing, tuple[str, ...]]:
    """Prize-line standing at the close of `session`.

    The prize line is the equity of the `prize_rank`-th best participant other than us. Competitor values come only from a
    snapshot dated `session`; an older snapshot is reported as stale rather than mixed with today's equity.

    Returns:
        (standing, warnings). Warnings: `LEADERBOARD_STALE` (no snapshot or not dated `session`), `EQUITY_UNKNOWN` (no
        ledger), `LEDGER_STALE` (ledger not dated `session`).
    """
    warnings: list[str] = []
    snapshot_date: date | None = None
    prize_line_equity: float | None = None
    leader_equity: float | None = None

    if snapshot is None or snapshot.base_date != session:
        warnings.append("LEADERBOARD_STALE")
        if snapshot is not None:
            snapshot_date = snapshot.base_date
    else:
        snapshot_date = snapshot.base_date
        others = sorted((e.total_return_pct for e in snapshot.entries if e.user_name != nickname), reverse=True)
        # A non-finite competitor cannot be ranked; mis-sorting it would silently shift the prize line.
        if others and all(math.isfinite(r) for r in others):
            leader_equity = 1.0 + others[0] / 100.0
            if len(others) >= prize_rank:
                prize_line_equity = 1.0 + others[prize_rank - 1] / 100.0

    our_equity, our_source, our_as_of = (ledger.equity, ledger.source, ledger.as_of) if ledger else (None, None, None)
    if ledger is None:
        warnings.append("EQUITY_UNKNOWN")
    elif ledger.as_of != session:
        warnings.append("LEDGER_STALE")

    required_multiple: float | None = None
    if (
        snapshot
        and snapshot.base_date == session
        and ledger
        and ledger.as_of == session
        and our_equity
        and our_equity > 0
        and prize_line_equity is not None
    ):
        required_multiple = prize_line_equity / our_equity

    return (
        Standing(
            snapshot_date=snapshot_date,
            our_equity=our_equity,
            our_source=our_source,
            our_as_of=our_as_of,
            prize_line_equity=prize_line_equity,
            leader_equity=leader_equity,
            required_multiple=required_multiple,
        ),
        tuple(warnings),
    )


def segment_to_dict(card: SegmentCard) -> dict[str, Any]:
    """JSON-ready primitives: ISO dates, enum values, `vote_changes` as a list of {alias, change_pct}, nested standing."""
    return {
        "session": card.session.isoformat(),
        "execution_session": card.execution_session.isoformat() if card.execution_session else None,
        "final_session": card.final_session,
        "action": card.action.value,
        "intraday_side": card.intraday_side.value if card.intraday_side else None,
        "vote_changes": [{"alias": alias, "change_pct": chg} for alias, chg in card.vote_changes],
        "down_votes": card.down_votes,
        "vote_min_down": card.vote_min_down,
        "sessions_remaining_after_execution": card.sessions_remaining_after_execution,
        "endgame": card.endgame,
        "standing": {
            "snapshot_date": card.standing.snapshot_date.isoformat() if card.standing.snapshot_date else None,
            "our_equity": card.standing.our_equity,
            "our_source": card.standing.our_source.value if card.standing.our_source else None,
            "our_as_of": card.standing.our_as_of.isoformat() if card.standing.our_as_of else None,
            "prize_line_equity": card.standing.prize_line_equity,
            "leader_equity": card.standing.leader_equity,
            "required_multiple": card.standing.required_multiple,
        },
        "warnings": list(card.warnings),
    }


def ledger_to_dict(state: LedgerState) -> dict[str, Any]:
    """{"as_of": ISO date, "equity": float, "source": enum value}."""
    return {
        "as_of": state.as_of.isoformat(),
        "equity": state.equity,
        "source": state.source.value,
    }


def ledger_from_dict(raw: Any) -> LedgerState | None:
    """Inverse of `ledger_to_dict`; None for malformed input, non-finite or non-positive equity, or an unknown source."""
    if not isinstance(raw, Mapping):
        return None
    try:
        as_of = date.fromisoformat(str(raw["as_of"]))
        equity = float(raw["equity"])
        if isinstance(raw["equity"], bool) or not math.isfinite(equity) or equity <= 0:
            return None
        source = EquitySource(str(raw["source"]))
        return LedgerState(as_of=as_of, equity=equity, source=source)
    except Exception:
        return None


def render_segment_markdown(card: SegmentCard, settings: SegmentSettings) -> str:
    """Korean operator card: vote, decision, standing, and a timed checklist.

    Every card opens the checklist with the 08:50 holdings check (normal state: the long vehicle in full) and the
    mismatch rule: when the account does not hold the long vehicle, sell everything in the open window and buy the
    intraday target (long vehicle for LONG/NO_DATA, inverse for SHORT). SWITCH_SHORT lists the open-window switch to the
    inverse and the close-window switch back; on the final session the close-window step is replaced by holding the
    inverse through the close. All order lines name both alias and ticker, and every switch is sell-all, confirm fill,
    then buy with full buying power. HOLD_LONG, NO_DATA and CONTEST_OVER list no order lines.
    """
    exec_str = (
        f"{card.execution_session.isoformat()}{' (대회 최종일)' if card.final_session else ''}"
        if card.execution_session
        else "없음"
    )

    vote_lines: list[str] = []
    for alias, chg in card.vote_changes:
        chg_str = f"{chg:+.2f}%" if chg is not None else "없음"
        vote_lines.append(f"- {alias}: {chg_str}")

    vote_count_str = f"하락 {card.down_votes}/{len(card.vote_changes)} (기준 {card.vote_min_down})"

    if card.action == SegmentAction.CONTEST_OVER:
        decision_str = "대회 종료"
    elif card.action == SegmentAction.NO_DATA:
        decision_str = "자료 부족"
    elif card.action == SegmentAction.HOLD_LONG:
        decision_str = "장중 롱"
    else:
        decision_str = "장중 숏"

    source_labels = {
        EquitySource.USER: "앱 입력",
        EquitySource.LEADERBOARD: "순위표",
        EquitySource.LEDGER: "추정",
    }
    source_label = source_labels.get(card.standing.our_source, "없음") if card.standing.our_source else "없음"

    our_ret_str = (
        f"{(card.standing.our_equity - 1.0) * 100.0:+.2f}% ({source_label})"
        if card.standing.our_equity is not None
        else "없음"
    )
    prize_ret_str = (
        f"{(card.standing.prize_line_equity - 1.0) * 100.0:+.2f}%"
        if card.standing.prize_line_equity is not None
        else "없음"
    )
    leader_ret_str = (
        f"{(card.standing.leader_equity - 1.0) * 100.0:+.2f}%"
        if card.standing.leader_equity is not None
        else "없음"
    )
    req_mult_str = (
        f"x{card.standing.required_multiple:.2f}"
        if card.standing.required_multiple is not None
        else "없음"
    )

    open_win = f"{settings.open_window.start.strftime('%H:%M')}–{settings.open_window.end.strftime('%H:%M')}"  # noqa: RUF001
    close_win = f"{settings.close_window.start.strftime('%H:%M')}–{settings.close_window.end.strftime('%H:%M')}"  # noqa: RUF001

    long_label = f"{settings.long_alias}({settings.long_ticker})"
    inv_label = f"{settings.inverse_alias}({settings.inverse_ticker})"

    checklist_lines: list[str] = [
        "- 단일가 호가 시간(08:30–09:00, 15:20–15:30)에는 일체 주문을 제출하지 않습니다.",  # noqa: RUF001
    ]

    if card.action != SegmentAction.CONTEST_OVER:
        target_label = inv_label if card.action == SegmentAction.SWITCH_SHORT else long_label
        checklist_lines.append(f"- 08:50 보유 점검: 정상 상태는 {long_label} 전량 보유입니다.")
        checklist_lines.append(
            f"- 불일치 규칙: 계좌가 {long_label} 전량 보유 상태가 아니라면 {open_win} 창에 보유 잔고 전량 매도 체결 확인 후 "
            f"장중 목표 종목 {target_label} 시장가 매수."
        )

        if card.action == SegmentAction.SWITCH_SHORT:
            checklist_lines.append(
                f"- {open_win}: {long_label} 전량 시장가 매도 체결 확인 후, {inv_label} 미수 없는 최대 수량 시장가 매수."
            )
            if card.final_session:
                checklist_lines.append(
                    f"- 장 마감까지 {inv_label} 보유 유지 (대회 최종일: 마감 환매 없이 종가까지 보유)."
                )
            else:
                checklist_lines.append(
                    f"- {close_win}: {inv_label} 전량 시장가 매도 체결 확인 후, {long_label} 미수 없는 최대 수량 시장가 매수."
                )
        elif card.action == SegmentAction.NO_DATA:
            aliases_str = ", ".join(alias for alias, _ in settings.vote)
            checklist_lines.append(
                f"- 수동 폴백 지침: HTS 전일대비 등락률({aliases_str}) 확인 후 {settings.vote_min_down}개 이상 하락이면 "
                "장중 롱(주문 없음 유지), 미만이면 장중 숏 스위칭 절차를 적용하십시오."
            )
        else:
            checklist_lines.append(
                f"- 장중 롱 유지: {long_label} 전량 보유 상태를 지속하며 신규 주문 없음."
            )
    else:
        checklist_lines.append("- 대회 종료: 신규 주문 없음.")

    warning_lines = [f"- {w}" for w in card.warnings] if card.warnings else ["- 없음"]

    sections = [
        f"# 구간 카드 ({card.session.isoformat()})",
        "",
        f"- 신호 세션: {card.session.isoformat()}",
        f"- 실행 세션: {exec_str}",
        "",
        f"## 투표 ({card.session.isoformat()} 종가 기준)",
        *vote_lines,
        f"- 집계: {vote_count_str}",
        f"- 판정: {decision_str}",
        "",
        "## 순위 및 성과",
        f"- 우리 누적수익률: {our_ret_str}",
        f"- 상금권 ({settings.prize_rank}위) 수익률: {prize_ret_str}",
        f"- 1위 수익률: {leader_ret_str}",
        f"- 상금권 진입 필요 배수: {req_mult_str}",
        f"- 잔여 세션: {card.sessions_remaining_after_execution}",
        "",
        "## 실행 체크리스트",
        *checklist_lines,
        "",
        "## 경고",
        *warning_lines,
        "",
    ]
    return "\n".join(sections)
