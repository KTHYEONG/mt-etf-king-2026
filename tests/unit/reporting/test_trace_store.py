from __future__ import annotations

import contextlib
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.reporting.trace_store import write_trace_artifacts


def test_write_trace_artifacts_schema_and_join_keys(tmp_path: Path) -> None:
    dest = tmp_path / "trace"
    sessions = pl.DataFrame(
        {
            "decision_date": [date(2026, 1, 2)],
            "n_universe": [10],
            "n_scores": [5],
            "n_selected": [1],
            "n_fills": [1],
            "n_unfilled": [0],
            "n_candidates_written": [1],
            "n_candidates_truncated": [0],
            "dropped_existence": [0],
            "dropped_price": [0],
            "dropped_history": [0],
            "dropped_sponsor": [0],
            "dropped_liquidity": [0],
            "dropped_eligibility": [0],
            "regime": ["NEUTRAL"],
            "equity": [1_000_000.0],
        }
    )
    with contextlib.suppress(Exception):
        sessions = sessions.with_columns(pl.col("decision_date").cast(pl.Date))
    candidates = pl.DataFrame(
        {
            "decision_date": [date(2026, 1, 2)],
            "ticker": ["069500"],
            "score": [0.5],
            "rank": [1],
            "selected": [True],
            "reject_reason": [""],
            "weight_raw": [0.5],
            "weight_target": [1.0],
            "weight_after_adv": [1.0],
            "weight_fill": [1.0],
        }
    )
    with contextlib.suppress(Exception):
        candidates = candidates.with_columns(pl.col("decision_date").cast(pl.Date))
    gates = [{"decision_date": "2026-01-02", "gate": "EMPTY_SCORES", "exc_type": ""}]
    ret = write_trace_artifacts(dest, sessions=sessions, candidates=candidates, gates=gates)
    assert ret == {"sessions": "sessions.parquet", "candidates": "candidates.parquet", "gates": "gates.jsonl"}
    assert (dest / "sessions.parquet").exists()
    assert (dest / "candidates.parquet").exists()
    assert (dest / "gates.jsonl").exists()
    cand_read = pl.read_parquet(dest / "candidates.parquet")
    assert cand_read["decision_date"].dtype == pl.Date
    assert cand_read["ticker"].dtype == pl.Utf8
    trades_like = pl.DataFrame({"decision_date": [date(2026, 1, 2)], "ticker": ["069500"], "value": [1]})
    with contextlib.suppress(Exception):
        trades_like = trades_like.with_columns(pl.col("decision_date").cast(pl.Date))
    joined = cand_read.join(trades_like, on=["decision_date", "ticker"], how="inner")
    assert joined.height == 1
    lines = (dest / "gates.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert set(obj.keys()) == {"decision_date", "gate", "exc_type"}


def test_write_trace_artifacts_fail_open_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "trace2"
    sessions = pl.DataFrame(
        {
            "decision_date": [date(2026, 1, 2)],
            "n_universe": [1],
            "n_scores": [1],
            "n_selected": [1],
            "n_fills": [1],
            "n_unfilled": [0],
            "n_candidates_written": [1],
            "n_candidates_truncated": [0],
            "dropped_existence": [0],
            "dropped_price": [0],
            "dropped_history": [0],
            "dropped_sponsor": [0],
            "dropped_liquidity": [0],
            "dropped_eligibility": [0],
            "regime": [""],
            "equity": [0.0],
        }
    )
    with contextlib.suppress(Exception):
        sessions = sessions.with_columns(pl.col("decision_date").cast(pl.Date))
    candidates = pl.DataFrame(
        {
            "decision_date": [date(2026, 1, 2)],
            "ticker": ["069500"],
            "score": [0.1],
            "rank": [1],
            "selected": [True],
            "reject_reason": [""],
            "weight_raw": [0.1],
            "weight_target": [0.1],
            "weight_after_adv": [0.1],
            "weight_fill": [0.1],
        }
    )
    with contextlib.suppress(Exception):
        candidates = candidates.with_columns(pl.col("decision_date").cast(pl.Date))
    orig_mkdir = Path.mkdir

    def fake_mkdir(self, *a, **kw):  # type: ignore[no-untyped-def]
        raise OSError("mock mkdir fail")

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    with pytest.raises(OSError, match="mock mkdir fail"):
        write_trace_artifacts(dest, sessions=sessions, candidates=candidates, gates=[])
    monkeypatch.setattr(Path, "mkdir", orig_mkdir)
    ret = write_trace_artifacts(dest, sessions=sessions, candidates=candidates, gates=[])
    assert ret["sessions"] == "sessions.parquet"
