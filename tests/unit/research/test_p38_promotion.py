from datetime import date
from pathlib import Path

import polars as pl

from src.research.p38_promotion import load_p38_terminal_by_entry_date


def test_load_p38_terminal_list_maps_panel_naive_index_to_executable_entry(tmp_path: Path) -> None:
    sessions = [date(2026, 1, d) for d in (2, 3, 5, 6, 7, 8, 9)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["X"] * len(sessions)})
    promo = tmp_path / "promotion.json"
    promo.write_text('{"terminal_returns": [0.10, 0.20, 0.30, 0.40, 0.50]}', encoding="utf-8")
    loaded = load_p38_terminal_by_entry_date(
        promotion_path=promo,
        calendar_sessions=sessions,
        panel=panel,
        horizon=2,
    )
    assert loaded == {date(2026, 1, 3): 0.10, date(2026, 1, 5): 0.20, date(2026, 1, 6): 0.30, date(2026, 1, 7): 0.40}


def test_load_p38_terminal_prefers_explicit_entry_map(tmp_path: Path) -> None:
    sessions = [date(2026, 1, d) for d in (2, 3, 5, 6, 7)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["X"] * len(sessions)})
    promo = tmp_path / "promotion.json"
    promo.write_text(
        '{"terminal_returns": [0.99], "terminal_returns_by_entry": {"2026-01-05": 0.42}}',
        encoding="utf-8",
    )
    loaded = load_p38_terminal_by_entry_date(
        promotion_path=promo,
        calendar_sessions=sessions,
        panel=panel,
        horizon=2,
    )
    assert loaded == {date(2026, 1, 5): 0.42}
