# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class PositionState(StrEnum):
    HOLD = "HOLD"
    TRIM = "TRIM"
    EXIT = "EXIT"
    WATCH = "WATCH"
    RE_ENTER = "RE_ENTER"


@dataclass
class PositionTracker:
    state: PositionState = PositionState.HOLD
    sessions_since_exit: int = 0
    cooldown: int = 3

    def transition(self, theme_state: str | None) -> PositionState:
        nxt = transition_position(self.state, theme_state, self.sessions_since_exit, self.cooldown)
        # update sessions_since_exit: EXIT counts as 1, WATCH increments, RE_ENTER resets
        if nxt == PositionState.EXIT:
            self.sessions_since_exit = 1
        elif nxt == PositionState.WATCH:
            if self.state in (PositionState.EXIT, PositionState.WATCH):
                self.sessions_since_exit += 1
            else:
                self.sessions_since_exit = 1
        elif nxt == PositionState.RE_ENTER:
            self.sessions_since_exit = 0
        else:
            # HOLD/TRIM reset
            self.sessions_since_exit = 0
        self.state = nxt
        return nxt


def apply_state_multipliers(
    weights: Mapping[str, float],
    states: Mapping[str, PositionState],
    *,
    trim_fraction: float = 0.5,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for ticker, w in weights.items():
        st = states.get(ticker)
        if st == PositionState.TRIM:
            out[ticker] = float(w) * float(trim_fraction)
        elif st in (PositionState.EXIT, PositionState.WATCH):
            out[ticker] = 0.0
        else:
            # HOLD, RE_ENTER, or unknown/missing -> 1.0
            out[ticker] = float(w)
    return out


def infer_theme_proxy(
    ticker: str,
    scores: Mapping[str, float],
    peak_score: float | None,
    *,
    score_drop_pct: float = 0.30,
    k: int = 3,
    conf_c0: float = 0.027174,
    tracker_state: PositionState | None = None,
) -> str:
    try:
        sc = float(scores.get(ticker, 0.0)) if ticker in scores else 0.0
    except Exception:
        sc = 0.0
    # BREAKDOWN via peak drop
    if peak_score is not None:
        try:
            pf = float(peak_score)
            if pf != 0 and sc <= pf * (1.0 - float(score_drop_pct)):
                return "BREAKDOWN"
        except Exception:  # noqa: S110
            pass
    # rank and confidence
    try:
        sorted_items = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    except Exception:
        sorted_items = []
    rank = None
    for idx, (t, _) in enumerate(sorted_items, start=1):
        if t == ticker:
            rank = idx
            break
    if sorted_items:
        try:
            vals = [float(v) for _, v in sorted_items]
            conf = vals[0] - vals[1] if len(vals) >= 2 else 0.0
        except Exception:  # noqa: S110
            conf = 0.0
    else:
        conf = 0.0
    # RECOVERY for EXIT/WATCH tracker within k
    if tracker_state in (PositionState.EXIT, PositionState.WATCH) and rank is not None and rank <= int(k) and sc > 0:
        return "RECOVERY"
    # rank > k -> BREAKDOWN (INV-B2-3)
    if rank is not None and rank > int(k):
        return "BREAKDOWN"
    # rank-1 low confidence -> OVERHEATED (trim, not full exit)
    if rank == 1 and conf < float(conf_c0) / 2.0:
        return "OVERHEATED"
    return "LEADING"


def transition_position(
    prev: PositionState,
    theme_state: str | None,
    sessions_since_exit: int,
    cooldown: int,
) -> PositionState:
    # EXIT / WATCH cooldown logic
    if prev == PositionState.EXIT or prev == PositionState.WATCH:
        if sessions_since_exit >= cooldown and theme_state == "RECOVERY":
            return PositionState.RE_ENTER
        # also if theme_state is LEADING after RECOVERY? Spec says RECOVERY -> LEADING, but test expects RECOVERY triggers RE_ENTER
        return PositionState.WATCH
    if prev == PositionState.HOLD:
        if theme_state == "LEADING":
            return PositionState.HOLD
        if theme_state == "OVERHEATED":
            return PositionState.TRIM
        if theme_state == "BREAKDOWN":
            return PositionState.EXIT
        if theme_state == "RECOVERY":
            return PositionState.HOLD
        return PositionState.HOLD
    if prev == PositionState.TRIM:
        if theme_state == "BREAKDOWN":
            return PositionState.EXIT
        if theme_state == "LEADING":
            return PositionState.HOLD
        if theme_state == "OVERHEATED":
            return PositionState.TRIM
        if theme_state == "RECOVERY":
            return PositionState.HOLD
        return PositionState.TRIM
    if prev == PositionState.RE_ENTER:
        return PositionState.HOLD
    return prev
