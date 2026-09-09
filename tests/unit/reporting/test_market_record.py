from datetime import date
import pytest


def test_summary_record_exposes_market_columns() -> None:
    from src.reporting.results import _extract_summary_record

    meta = {
        "model": "P27",
        "strategy_id": "sticky.mom60_raw",
        "start": str(date(2018, 1, 2)),
        "end": str(date(2026, 8, 27)),
        "horizon": 36,
    }
    summary = {
        "n_windows": 2086,
        "n_effective": 58,
        "capture_market": {"0.3": 0.211, "0.4": 0.244, "0.5": 0.301, "0.6": 0.296},
        "attainability_market": {"0.3": 0.419, "0.4": 0.257, "0.5": 0.177, "0.6": 0.125},
        "n_attainable_market": {"0.3": 856, "0.4": 520, "0.5": 369, "0.6": 250},
        "breadth_mean_market": 269.4,
    }

    record = _extract_summary_record("run-market", meta, summary)

    assert record["capture_market_50"] == pytest.approx(0.301)
    assert record["capture_market_30"] == pytest.approx(0.211)
    assert record["attainability_market_50"] == pytest.approx(0.177)
    assert record["attainability_market_60"] == pytest.approx(0.125)
    assert record["attainable_market_50"] == 369
    assert record["attainable_market_30"] == 856
    assert record["breadth_mean_market"] == pytest.approx(269.4)


def test_summary_record_market_columns_default_to_none_for_legacy_runs() -> None:
    from src.reporting.results import _extract_summary_record

    # Given: a summary written before the market fields existed, plus a non-numeric intruder
    record = _extract_summary_record(
        "run-legacy",
        {"model": "P27", "strategy_id": "sticky.mom60_raw"},
        {"n_windows": 10, "n_effective": 1, "capture_market": {"0.5": "not-a-number"}},
    )

    # Then: absent keys and unparseable values both degrade to None, never raise (R9)
    assert record["capture_market_50"] is None
    assert record["capture_market_40"] is None
    assert record["attainability_market_50"] is None
    assert record["attainable_market_50"] is None
    assert record["breadth_mean_market"] is None
