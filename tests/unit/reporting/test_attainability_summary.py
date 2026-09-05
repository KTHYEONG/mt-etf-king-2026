from datetime import date


def test_summary_payload_exposes_attainability_and_capture() -> None:
    from src.reporting.results import _extract_summary_record

    meta = {
        "model": "P36",
        "strategy_id": "sticky.mom60_vol_hysteresis",
        "start": str(date(2018, 1, 2)),
        "end": str(date(2026, 8, 27)),
        "horizon": 36,
    }
    summary = {
        "n_windows": 2088,
        "n_effective": 58,
        "exceedance": {"0.3": 0.086, "0.4": 0.057, "0.5": 0.045},
        "attainability": {"0.4": 0.090, "0.5": 0.061},
        "capture": {"0.4": 0.601, "0.5": None},
        "n_attainable": {"0.4": 188, "0.5": 12},
        "breadth_mean": 1.62,
    }

    record = _extract_summary_record("run-1", meta, summary)

    assert record["p_gt_40"] == 0.057
    assert record["capture_40"] == 0.601
    assert record["capture_50"] is None
    assert record["attainable_40"] == 188
    assert record["breadth_mean"] == 1.62


def test_summary_capture_rejects_non_numeric_values() -> None:
    from src.reporting.results import _extract_summary_record

    record = _extract_summary_record(
        "run-2",
        {"model": "P36", "strategy_id": "sticky.mom60_vol_hysteresis"},
        {
            "n_windows": 1,
            "n_effective": 1,
            "capture": {"0.4": "not-a-number"},
            "attainability": {"0.4": "bad"},
            "n_attainable": {"0.4": 1},
        },
    )
    assert record["capture_40"] is None
    assert record["attainability_40"] is None
