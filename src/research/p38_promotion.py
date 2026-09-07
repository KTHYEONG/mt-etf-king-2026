"""P38 promotion artifact loading (date-keyed, no index-slice onto executable population)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.session_grid import resolve_session_grid


class PromotionLoadError(ValueError):
    """Raised when promotion JSON cannot be mapped onto executable entry dates."""


def load_p38_terminal_by_entry_date(
    *,
    promotion_path: Path | None,
    calendar_sessions: Sequence[date],
    panel: pl.DataFrame,
    horizon: int,
) -> dict[date, float] | None:
    if promotion_path is None or not promotion_path.exists():
        return None
    raw = json.loads(promotion_path.read_text(encoding="utf-8"))
    terms_obj = raw.get("terminal_returns")
    if terms_obj is None:
        return None
    if isinstance(terms_obj, Mapping):
        parsed = {date.fromisoformat(str(key)): float(str(value)) for key, value in terms_obj.items()}
        return parsed or None
    if not isinstance(terms_obj, list):
        return None
    terms = [float(str(value)) for value in terms_obj]
    if not terms:
        return None
    by_entry = raw.get("terminal_returns_by_entry")
    if isinstance(by_entry, Mapping):
        parsed = {date.fromisoformat(str(key)): float(str(value)) for key, value in by_entry.items()}
        return parsed or None
    grid = resolve_session_grid(calendar_sessions, panel)
    sessions = list(grid.sessions)
    n_exec = len(sessions) - horizon - 1
    if n_exec <= 0 or len(terms) < n_exec:
        return None
    return {sessions[i + 1]: terms[i] for i in range(n_exec)}
