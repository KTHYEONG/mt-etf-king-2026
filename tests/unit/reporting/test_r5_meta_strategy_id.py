def test_r5_backtest_meta_includes_strategy_id(tmp_path) -> None:
    import json
    from datetime import date

    import polars as pl

    from src.core.paths import DataPaths
    from src.reporting.results import write_backtest_result
    from src.strategies.ids import STICKY_MOM60_RAW

    paths = DataPaths(root=tmp_path)
    daily = pl.DataFrame({"date": [date(2020, 1, 2)], "return": [0.01]})
    trades = pl.DataFrame({"date": [date(2020, 1, 2)], "ticker": ["X"], "weight": [1.0]})
    dest = write_backtest_result(
        paths,
        run_id="test_run",
        meta={"strategy_id": STICKY_MOM60_RAW, "legacy_model_id": "P27"},
        summary={"ok": True},
        daily=daily,
        trades=trades,
    )
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    assert meta["strategy_id"] == STICKY_MOM60_RAW
    assert meta["legacy_model_id"] == "P27"
