def test_summarise_tail_miss_windows_labels_near_miss() -> None:
    from datetime import date

    import polars as pl

    from src.reporting.tail_forensics import summarise_tail_miss_windows

    windows = pl.DataFrame(
        {
            "window_start": [date(2024, 1, 2), date(2024, 1, 5)],
            "window_end": [date(2024, 1, 4), date(2024, 1, 8)],
            "terminal_return": [0.35, 0.10],
            "gt_40": [False, False],
        }
    )
    candidates = pl.DataFrame(
        {
            "decision_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "selected": [True, True],
            "lottery_active": [False, False],
            "multiple": [2, 2],
            "weight_fill": [1.0, 1.0],
        }
    )
    sessions = pl.DataFrame({"decision_date": [date(2024, 1, 2)], "regime": ["RISK_ON"]})
    report = summarise_tail_miss_windows(windows, candidates, sessions, threshold=0.40, near_miss_lo=0.20)
    assert report.n_windows == 2
    assert report.n_near_miss == 1
    assert report.label_counts.get("LOTTERY_INACTIVE", 0) >= 1
    assert len(report.top_windows) >= 1

def test_summarise_tail_miss_windows_fail_closed_empty() -> None:
    import polars as pl

    from src.reporting.tail_forensics import summarise_tail_miss_windows

    empty = pl.DataFrame(
        {
            "window_start": [],
            "window_end": [],
            "terminal_return": [],
            "gt_40": [],
        }
    )
    report = summarise_tail_miss_windows(empty, pl.DataFrame(), pl.DataFrame(), threshold=0.40)
    assert report.n_windows == 0
    assert report.n_near_miss == 0
    assert report.label_counts == {}
    assert report.top_windows == ()

def test_write_tail_miss_report_json(tmp_path) -> None:
    import json

    from src.reporting.tail_forensics import TailMissReport, write_tail_miss_report

    report = TailMissReport(threshold=0.40, n_windows=5, n_near_miss=2, label_counts={"LOTTERY_INACTIVE": 2}, top_windows=({"window_start": "2024-01-02", "label": "LOTTERY_INACTIVE"},))
    out = write_tail_miss_report(tmp_path, report)
    assert out.endswith("tail_miss_report.json")
    data = json.loads((tmp_path / "tail_miss_report.json").read_text(encoding="utf-8"))
    assert data["threshold"] == 0.40
    assert data["label_counts"]["LOTTERY_INACTIVE"] == 2
