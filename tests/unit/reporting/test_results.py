from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.paths import DataPaths
from src.reporting.results import make_backtest_run_id, write_backtest_result


def test_SCENARIO_DSR_04_write_and_refuse(tmp_path: Path) -> None:  # noqa: N802
    """SCENARIO_DSR_04: write creates meta+summary; second same run_id raises."""
    paths = DataPaths(root=tmp_path)
    run_id = make_backtest_run_id("B1", date(2026, 1, 1), date(2026, 1, 10), now=datetime(2026, 1, 11, 12, 0, 0, tzinfo=UTC))
    assert "/" not in run_id
    assert "\\" not in run_id
    assert "B1" in run_id
    meta = {"model": "B1", "start": "2026-01-01", "end": "2026-01-10", "horizon": 20}
    summary = {"n_windows": 10, "quantiles": {"0.5": 0.01}, "cvar_05": -0.02, "right_tail_score": 0.1}
    dest = write_backtest_result(paths, run_id=run_id, meta=meta, summary=summary)
    assert dest == tmp_path / "docs/results" / run_id
    assert (dest / "meta.json").exists()
    assert (dest / "summary.json").exists()
    with open(dest / "meta.json", encoding="utf-8") as f:
        m = json.load(f)
        assert "model" in m
    with open(dest / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
        assert "n_windows" in s or "quantiles" in s
    # second call same run_id => FileExistsError
    with pytest.raises(FileExistsError):
        write_backtest_result(paths, run_id=run_id, meta=meta, summary=summary)
    # run_id filesystem-safe token: no path separators
    assert ".." not in run_id
    # ensure indent=2 and ensure_ascii False (human readability)
    raw_meta = (dest / "meta.json").read_text(encoding="utf-8")
    assert "\n  " in raw_meta or '  "' in raw_meta  # indent


def test_write_backtest_result_serializes_oneshot_rows(tmp_path: Path) -> None:
    import json
    from datetime import date

    from src.tournament.distribution import serialize_oneshot_rows

    paths = DataPaths(root=tmp_path)
    run_id = make_backtest_run_id("P27", date(2018, 1, 2), date(2026, 8, 27))
    rows = ((2024, date(2024, 9, 23), 0.431),)
    summary = {
        "oneshot": {
            "starts": ["2024-09-23"],
            "rows": serialize_oneshot_rows(rows),
        }
    }
    dest = write_backtest_result(paths, run_id=run_id, meta={"model": "P27"}, summary=summary)
    with open(dest / "summary.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["oneshot"]["rows"] == [[2024, "2024-09-23", 0.431]]


def test_make_backtest_run_id_safe(tmp_path: Path) -> None:  # noqa: ANN001, ARG001
    ids = make_backtest_run_id("B2", date(2026, 1, 1), date(2026, 2, 1))
    assert "/" not in ids
    assert "\\" not in ids
    assert "B2" in ids
