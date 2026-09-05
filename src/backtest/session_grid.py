# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionGrid:
    sessions: tuple[date, ...]
    phantom: tuple[date, ...]


def resolve_session_grid(sessions: Sequence[date], panel: pl.DataFrame, *, min_rows: int = 1) -> SessionGrid:
    want = list(sessions)
    try:
        need = max(1, int(min_rows))
    except Exception:
        need = 1
    counts: dict[date, int] = {}
    try:
        if panel is not None and isinstance(panel, pl.DataFrame) and "date" in panel.columns and panel.height > 0:
            for d in panel["date"].to_list():
                if isinstance(d, date):
                    counts[d] = counts.get(d, 0) + 1
    except Exception:
        counts = {}
    kept: list[date] = []
    phantom: list[date] = []
    for d in want:
        if counts.get(d, 0) >= need:
            kept.append(d)
        else:
            phantom.append(d)
    for d in phantom:
        logger.info(f"[DATA] gate=phantom_session date={d} status=DROP")
    return SessionGrid(sessions=tuple(kept), phantom=tuple(phantom))


def phantom_session_labels(sessions: Sequence[date], panel: pl.DataFrame) -> list[str]:
    try:
        return [str(d) for d in resolve_session_grid(sessions, panel).phantom]
    except Exception:
        return []
