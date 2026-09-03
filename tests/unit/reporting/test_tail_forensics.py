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


def test_compound_close_return_basic_and_missing() -> None:
    from datetime import date

    import polars as pl

    from src.reporting.tail_forensics import compound_close_return

    panel = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "ticker": ["A", "A", "A"],
            "close": [100.0, 110.0, 125.0],
        }
    )
    assert abs(float(compound_close_return(panel, "A", date(2024, 1, 2), date(2024, 1, 4)) or 0.0) - 0.25) < 1e-12
    assert compound_close_return(panel, "A", date(2024, 1, 1), date(2024, 1, 4)) is None
    assert compound_close_return(panel, "B", date(2024, 1, 2), date(2024, 1, 4)) is None


def test_select_attribution_windows_top_q_and_near_miss() -> None:
    from datetime import date

    import polars as pl

    from src.reporting.tail_forensics import select_attribution_windows

    windows = pl.DataFrame(
        {
            "window_start": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "window_end": [date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 3), date(2024, 2, 4)],
            "terminal_return": [1.0, 0.40, 0.10, -0.05],
            "giveback": [0.1, 0.2, 0.0, 0.0],
        }
    )
    out = select_attribution_windows(windows, top_q=0.75, near_miss_lo=0.20, near_miss_hi=0.50)
    starts = set(out["window_start"].to_list())
    assert date(2024, 1, 1) in starts
    assert date(2024, 1, 2) in starts
    assert date(2024, 1, 3) not in starts
    assert date(2024, 1, 4) not in starts


def test_attribute_window_selection_loss_dominates() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.reporting.tail_forensics import attribute_window

    ws, we = date(2024, 1, 2), date(2024, 1, 5)
    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    panel = pl.DataFrame(
        {
            "date": sessions * 2,
            "ticker": ["BEST"] * 4 + ["HELD"] * 4,
            "close": [100.0, 110.0, 120.0, 200.0, 100.0, 101.0, 102.0, 110.0],
        }
    )
    trades = pl.DataFrame(
        {
            "decision_date": sessions,
            "execution_date": sessions,
            "ticker": ["HELD", "HELD", "HELD", "HELD"],
            "side": ["BUY", "BUY", "BUY", "BUY"],
            "weight_before": [0.0, 0.95, 0.95, 0.95],
            "weight_after": [0.95, 0.95, 0.95, 0.95],
            "delta_weight": [0.95, 0.0, 0.0, 0.0],
            "weight": [0.95, 0.95, 0.95, 0.95],
            "price": [100.0, 101.0, 102.0, 110.0],
        }
    )
    master = SimpleNamespace(
        attributes={
            "BEST": SimpleNamespace(leverage_multiple=2, leverage_family_key="FAM_BEST"),
            "HELD": SimpleNamespace(leverage_multiple=2, leverage_family_key="FAM_HELD"),
        }
    )
    attr = attribute_window(
        window_start=ws,
        window_end=we,
        realized_return=0.10,
        giveback=0.01,
        trades=trades,
        panel=panel,
        master=master,
        sessions=sessions,
    )
    assert attr.best_family == "FAM_BEST"
    assert attr.actual_family == "FAM_HELD"
    assert attr.selection_loss > attr.entry_timing_loss + attr.exit_timing_loss
    assert attr.dominant_bucket == "selection"


def test_attribute_window_entry_timing_loss() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.reporting.tail_forensics import attribute_window

    ws, we = date(2024, 1, 2), date(2024, 1, 5)
    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    panel = pl.DataFrame(
        {
            "date": sessions,
            "ticker": ["T"] * 4,
            "close": [100.0, 150.0, 160.0, 170.0],
        }
    )
    trades = pl.DataFrame(
        {
            "decision_date": [date(2024, 1, 4), date(2024, 1, 5)],
            "execution_date": [date(2024, 1, 4), date(2024, 1, 5)],
            "ticker": ["T", "T"],
            "side": ["BUY", "BUY"],
            "weight_before": [0.0, 0.95],
            "weight_after": [0.95, 0.95],
            "delta_weight": [0.95, 0.0],
            "weight": [0.95, 0.95],
            "price": [160.0, 170.0],
        }
    )
    master = SimpleNamespace(
        attributes={"T": SimpleNamespace(leverage_multiple=2, leverage_family_key="FAM_T")}
    )
    attr = attribute_window(
        window_start=ws,
        window_end=we,
        realized_return=0.05,
        giveback=0.0,
        trades=trades,
        panel=panel,
        master=master,
        sessions=sessions,
    )
    assert attr.actual_family == "FAM_T"
    assert attr.entry_timing_loss > 0.0
    assert attr.selection_loss == 0.0


def test_summarise_tail_attribution_sets_primary_gap() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.reporting.tail_forensics import summarise_tail_attribution, write_tail_attribution_report

    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    windows = pl.DataFrame(
        {
            "window_start": [date(2024, 1, 2)],
            "window_end": [date(2024, 1, 5)],
            "terminal_return": [1.0],
            "giveback": [0.02],
        }
    )
    panel = pl.DataFrame(
        {
            "date": sessions * 2,
            "ticker": ["BEST"] * 4 + ["HELD"] * 4,
            "close": [100.0, 110.0, 120.0, 200.0, 100.0, 101.0, 102.0, 110.0],
        }
    )
    trades = pl.DataFrame(
        {
            "decision_date": sessions,
            "execution_date": sessions,
            "ticker": ["HELD"] * 4,
            "side": ["BUY"] * 4,
            "weight_before": [0.0, 0.95, 0.95, 0.95],
            "weight_after": [0.95, 0.95, 0.95, 0.95],
            "delta_weight": [0.95, 0.0, 0.0, 0.0],
            "weight": [0.95] * 4,
            "price": [100.0, 101.0, 102.0, 110.0],
        }
    )
    master = SimpleNamespace(
        attributes={
            "BEST": SimpleNamespace(leverage_multiple=2, leverage_family_key="FAM_BEST"),
            "HELD": SimpleNamespace(leverage_multiple=2, leverage_family_key="FAM_HELD"),
        }
    )
    summary = summarise_tail_attribution(
        windows=windows,
        trades=trades,
        panel=panel,
        master=master,
        sessions=sessions,
        top_q=0.0,
        near_miss_lo=0.20,
        near_miss_hi=0.50,
    )
    assert summary.n_analyzed >= 1
    assert summary.selection_dominates_timing is True
    assert summary.primary_gap == "selection"
    from pathlib import Path

    out = write_tail_attribution_report(Path("tmp/tail_attr_test"), summary)
    assert out.endswith("tail_attribution_report.json")


def test_attribute_window_unknown_without_trades() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.reporting.tail_forensics import attribute_window

    ws, we = date(2024, 1, 2), date(2024, 1, 3)
    sessions = [ws, we]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A", "A"], "close": [100.0, 110.0]})
    trades = pl.DataFrame(
        {
            "decision_date": [],
            "execution_date": [],
            "ticker": [],
            "side": [],
            "weight_before": [],
            "weight_after": [],
            "delta_weight": [],
            "weight": [],
            "price": [],
        }
    )
    master = SimpleNamespace(
        attributes={"A": SimpleNamespace(leverage_multiple=2, leverage_family_key="FAM_A")}
    )
    attr = attribute_window(
        window_start=ws,
        window_end=we,
        realized_return=0.0,
        giveback=0.0,
        trades=trades,
        panel=panel,
        master=master,
        sessions=sessions,
    )
    assert attr.actual_family is None
    assert attr.dominant_bucket == "UNKNOWN"
    assert attr.selection_loss == 0.0
