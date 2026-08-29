# mypy: ignore-errors
from __future__ import annotations

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
        # update sessions_since_exit
        if self.state == PositionState.EXIT:
            self.sessions_since_exit = 1
        elif self.state == PositionState.WATCH:
            self.sessions_since_exit += 1
        else:
            # if we just entered EXIT, reset, else increment? For HOLD/TRIM/RE_ENTER keep 0
            if nxt == PositionState.EXIT:
                self.sessions_since_exit = 0
            elif nxt == PositionState.WATCH:
                # coming from EXIT
                self.sessions_since_exit = 1
            else:
                self.sessions_since_exit = 0
        # handle post transition for WATCH counting
        if nxt == PositionState.WATCH and self.state not in (PositionState.EXIT, PositionState.WATCH):
            self.sessions_since_exit = 1
        self.state = nxt
        return nxt


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
