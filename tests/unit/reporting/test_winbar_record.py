from datetime import date

import pytest


def test_summary_record_exposes_winbar_fields() -> None:
    from src.reporting.results import _extract_summary_record

    meta = {
        "model": "P27",
        "strategy_id": "sticky.mom60_raw",
        "start": str(date(2018, 1, 2)),
        "end": str(date(2026, 8, 27)),
        "horizon": 36,
    }
    summary = {
        "n_windows": 2088,
        "n_effective": 58,
        "win_bar_rank": 5,
        "win_bar_oracle_ratio": 0.621,
        "win_bar_exceedance": {"rank": 0.0965, "ratio": 0.1222},
        "win_bar_median": {"rank": 0.2131, "ratio": 0.1994},
        "n_win_bar_windows": {"rank": 1824, "ratio": 1824},
    }

    record = _extract_summary_record("run-winbar", meta, summary)

    assert record["win_bar_rank"] == 5
    assert record["win_bar_oracle_ratio"] == pytest.approx(0.621, abs=1e-12)
    assert record["win_bar_exceedance_rank"] == pytest.approx(0.0965, abs=1e-12)
    assert record["win_bar_exceedance_ratio"] == pytest.approx(0.1222, abs=1e-12)
    assert record["win_bar_median_rank"] == pytest.approx(0.2131, abs=1e-12)
    assert record["win_bar_median_ratio"] == pytest.approx(0.1994, abs=1e-12)
    assert record["n_win_bar_windows_rank"] == 1824
    assert record["n_win_bar_windows_ratio"] == 1824

def test_summary_record_winbar_fields_default_to_none_for_legacy_runs() -> None:
    from src.reporting.results import _extract_summary_record

    # Given: a summary written before the winbar feature existed, plus a non-numeric intruder
    record = _extract_summary_record(
        "run-legacy",
        {"model": "P27", "strategy_id": "sticky.mom60_raw"},
        {"n_windows": 10, "n_effective": 1, "win_bar_exceedance": {"rank": "not-a-number"}},
    )

    # Then: absent keys and unparseable values both degrade to None, never raise
    assert record["win_bar_rank"] is None
    assert record["win_bar_oracle_ratio"] is None
    assert record["win_bar_exceedance_rank"] is None
    assert record["win_bar_exceedance_ratio"] is None
    assert record["win_bar_median_rank"] is None
    assert record["win_bar_median_ratio"] is None
    assert record["n_win_bar_windows_rank"] is None
    assert record["n_win_bar_windows_ratio"] is None
