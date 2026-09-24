# ruff: noqa
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse
import polars as pl

import pytest


@pytest.fixture(autouse=True)
def _champion_decide_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.cli.commands.pipeline._contest_mode_active", lambda: False)


def _write_silver_dates(root, name: str, days: list) -> None:
    silver_dir = root / "normalized"
    silver_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": days}, schema={"date": pl.Date}).write_parquet(silver_dir / f"{name}.parquet")


def test_cmd_daily_refresh_happy_path_without_decide(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        {"date": [date(2018, 1, 2), date(2026, 8, 20), date(2026, 8, 27)]},
        schema={"date": pl.Date},
    ).write_parquet(silver_dir / "etf_daily.parquet")
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 20), date(2026, 8, 27)])

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 25), date(2026, 8, 26), date(2026, 8, 27)]

    ingest_mock = MagicMock(return_value=0)
    normalize_mock = MagicMock(return_value=0)
    features_mock = MagicMock(return_value=0)
    decide_mock = MagicMock(side_effect=AssertionError("decide must not be called when args.decide is False"))

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", ingest_mock),
        patch("src.cli.commands.data.cmd_normalize", normalize_mock),
        patch("src.cli.commands.features.cmd_features", features_mock),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 0
    ingest_kwargs = next(c.args[0] for c in ingest_mock.call_args_list if c.args[0].dataset == "etf_daily")
    assert ingest_kwargs.dataset == "etf_daily"
    assert ingest_kwargs.start == "2026-08-22"
    assert ingest_kwargs.end == "2026-08-27"
    assert ingest_kwargs.dry_run is False

    normalize_kwargs = next(c.args[0] for c in normalize_mock.call_args_list if c.args[0].dataset == "etf_daily")
    assert normalize_kwargs.dataset == "etf_daily"
    assert normalize_kwargs.mode == "incremental"

    features_kwargs = features_mock.call_args.args[0]
    assert features_kwargs.start == "2018-01-02"
    assert features_kwargs.end == "2026-08-27"

    decide_mock.assert_not_called()


from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse


def test_cmd_daily_refresh_ingest_failure_short_circuits(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=1)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.features.cmd_features", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse


def test_cmd_daily_refresh_normalize_failure_short_circuits(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=1)),
        patch("src.cli.commands.features.cmd_features", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse


def test_cmd_daily_refresh_missing_silver_panel_fails_closed(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse
import polars as pl


def test_cmd_daily_refresh_with_decide_computes_and_persists_recommendation(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh
    from src.cli.constants import CHAMPION_STRATEGY

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame({"date": [date(2018, 1, 2), date(2026, 8, 27)]}, schema={"date": pl.Date}).write_parquet(silver_dir / "etf_daily.parquet")
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    decide_mock = MagicMock(return_value=0)

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=True, output_dir="results/decide_daily")

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 0
    decide_kwargs = decide_mock.call_args.args[0]
    assert decide_kwargs.model == CHAMPION_STRATEGY
    assert decide_kwargs.date == "2026-08-27"
    assert decide_kwargs.panel is None
    assert decide_kwargs.capital is None
    assert decide_kwargs.output == "results/decide_daily/2026-08-27.json"
    assert decide_kwargs.trace is False


from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse
import polars as pl


def test_cmd_daily_refresh_decide_failure_fails_closed(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame({"date": [date(2018, 1, 2), date(2026, 8, 27)]}, schema={"date": pl.Date}).write_parquet(silver_dir / "etf_daily.parquet")
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=True, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(return_value=1)),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


from unittest.mock import MagicMock, patch
import argparse


def test_cmd_daily_refresh_invalid_as_of_fails_closed() -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    args = argparse.Namespace(dataset=None, as_of="not-a-date", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.features.cmd_features", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("must not be called"))),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


from unittest.mock import MagicMock, patch
import argparse


def test_cmd_daily_refresh_no_session_in_lookback_window_fails_closed() -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = []

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.features.cmd_features", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("must not be called"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


import argparse
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl


def test_cmd_daily_refresh_ingests_and_normalizes_the_index_dataset(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame({"date": [date(2018, 1, 2), date(2026, 8, 27)]}, schema={"date": pl.Date}).write_parquet(
        silver_dir / "etf_daily.parquet"
    )
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    ingest_mock = MagicMock(return_value=0)
    normalize_mock = MagicMock(return_value=0)

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", ingest_mock),
        patch("src.cli.commands.data.cmd_normalize", normalize_mock),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 0
    ingest_datasets = [c.args[0].dataset for c in ingest_mock.call_args_list]
    normalize_datasets = [c.args[0].mode and c.args[0].dataset for c in normalize_mock.call_args_list]
    # Then: bronze alias is 'kospi_index', silver alias is 'index_daily' (R8)
    assert "kospi_index" in ingest_datasets
    assert "index_daily" in normalize_datasets
    # Then: the index ingest reuses the SAME window as the etf ingest (R10)
    index_call = next(c.args[0] for c in ingest_mock.call_args_list if c.args[0].dataset == "kospi_index")
    etf_call = next(c.args[0] for c in ingest_mock.call_args_list if c.args[0].dataset == "etf_daily")
    assert index_call.start == etf_call.start
    assert index_call.end == etf_call.end


import argparse
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl


def test_cmd_daily_refresh_index_stage_failure_short_circuits(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame({"date": [date(2018, 1, 2), date(2026, 8, 27)]}, schema={"date": pl.Date}).write_parquet(
        silver_dir / "etf_daily.parquet"
    )

    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]

    def _ingest(ns):
        return 1 if ns.dataset == "kospi_index" else 0

    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=True, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(side_effect=_ingest)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(side_effect=AssertionError("must not run"))),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("must not run"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


def test_resolve_pipeline_target_session_pre_market_weekday() -> None:
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    from unittest.mock import MagicMock
    from src.cli.commands.pipeline import resolve_pipeline_target_session

    kst = ZoneInfo("Asia/Seoul")
    now_dt = datetime(2026, 9, 15, 8, 30, tzinfo=kst)
    cal_mock = MagicMock()
    cal_mock.is_session.side_effect = lambda d: d in (date(2026, 9, 11), date(2026, 9, 14), date(2026, 9, 15))
    cal_mock.sessions.side_effect = lambda s, e: [d for d in [date(2026, 9, 11), date(2026, 9, 14), date(2026, 9, 15)] if s <= d <= e]

    target = resolve_pipeline_target_session(now=now_dt, calendar=cal_mock)
    assert target == date(2026, 9, 14)

    target_explicit = resolve_pipeline_target_session(as_of=date(2026, 9, 15), now=now_dt, calendar=cal_mock)
    assert target_explicit == date(2026, 9, 14)


def test_resolve_pipeline_target_session_post_market_weekday() -> None:
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    from unittest.mock import MagicMock
    from src.cli.commands.pipeline import resolve_pipeline_target_session

    kst = ZoneInfo("Asia/Seoul")
    now_dt = datetime(2026, 9, 15, 16, 0, tzinfo=kst)
    cal_mock = MagicMock()
    cal_mock.is_session.side_effect = lambda d: d in (date(2026, 9, 11), date(2026, 9, 14), date(2026, 9, 15))
    cal_mock.sessions.side_effect = lambda s, e: [d for d in [date(2026, 9, 11), date(2026, 9, 14), date(2026, 9, 15)] if s <= d <= e]

    target = resolve_pipeline_target_session(now=now_dt, calendar=cal_mock)
    assert target == date(2026, 9, 15)


def test_resolve_pipeline_target_session_explicit_past_and_future() -> None:
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    from unittest.mock import MagicMock
    from src.cli.commands.pipeline import resolve_pipeline_target_session

    kst = ZoneInfo("Asia/Seoul")
    now_dt = datetime(2026, 9, 15, 8, 30, tzinfo=kst)
    cal_mock = MagicMock()
    cal_mock.is_session.side_effect = lambda d: d in (date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 14), date(2026, 9, 15))
    cal_mock.sessions.side_effect = lambda s, e: [d for d in [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 14), date(2026, 9, 15)] if s <= d <= e]

    target_past = resolve_pipeline_target_session(as_of=date(2026, 9, 10), now=now_dt, calendar=cal_mock)
    assert target_past == date(2026, 9, 10)

    target_future = resolve_pipeline_target_session(as_of=date(2026, 9, 20), now=now_dt, calendar=cal_mock)
    assert target_future is None


def test_cmd_daily_refresh_wiring_uses_time_aware_target_session(tmp_path) -> None:
    import argparse
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import polars as pl
    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame({"date": [date(2018, 1, 2), date(2026, 9, 14)]}, schema={"date": pl.Date}).write_parquet(silver_dir / "etf_daily.parquet")
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 9, 14)])

    args = argparse.Namespace(dataset=None, as_of="2026-09-15", lookback_days=5, decide=True, output_dir=None)
    mock_resolve = MagicMock(return_value=date(2026, 9, 14))
    decide_mock = MagicMock(return_value=0)
    ingest_mock = MagicMock(return_value=0)

    with (
        patch("src.cli.commands.pipeline.resolve_pipeline_target_session", mock_resolve),
        patch("src.cli.commands.data.cmd_ingest", ingest_mock),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 0
    mock_resolve.assert_called_once_with(as_of=date(2026, 9, 15))
    decide_call = decide_mock.call_args.args[0]
    assert decide_call.date == "2026-09-14"


def _catchup_fixture(tmp_path, ledger_text: str) -> list[date]:
    """준비: 은색 패널과 챔피언 스티키 원장을 만든다."""
    grid = [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21)]
    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": grid}, schema={"date": pl.Date}).write_parquet(silver_dir / "etf_daily.parquet")
    _write_silver_dates(tmp_path, "index_daily", grid)

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sticky_mom60_post_crash_anchor_position.json").write_text(ledger_text, encoding="utf-8")
    return grid


_CATCHUP_LEDGER = (
    '{"as_of": "2026-09-16", "held": "122630", "held_weight": 0.95, "hold_len": 1, '
    '"history": {"2026-09-16": {"held": "122630", "held_weight": 0.95, "hold_len": 1}}}'
)


def _catchup_args() -> argparse.Namespace:
    return argparse.Namespace(dataset=None, as_of="2026-09-21", lookback_days=5, decide=True, output_dir=None)


def test_cmd_daily_refresh_decides_catchup_sessions_before_target(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    grid = _catchup_fixture(tmp_path, _CATCHUP_LEDGER)
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = grid
    decide_mock = MagicMock(return_value=0)

    with (
        patch("src.cli.commands.pipeline.resolve_pipeline_target_session", MagicMock(return_value=date(2026, 9, 21))),
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(_catchup_args())

    assert rc == 0
    dates = [c.args[0].date for c in decide_mock.call_args_list]
    outputs = [c.args[0].output for c in decide_mock.call_args_list]
    assert dates == ["2026-09-17", "2026-09-18", "2026-09-21"]
    assert outputs == [
        "results/decide_daily/2026-09-17.json",
        "results/decide_daily/2026-09-18.json",
        "results/decide_daily/2026-09-21.json",
    ]


def test_cmd_daily_refresh_aborts_target_when_a_catchup_decision_fails(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    grid = _catchup_fixture(tmp_path, _CATCHUP_LEDGER)
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = grid
    decide_mock = MagicMock(return_value=1)

    with (
        patch("src.cli.commands.pipeline.resolve_pipeline_target_session", MagicMock(return_value=date(2026, 9, 21))),
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(_catchup_args())

    assert rc == 1
    assert decide_mock.call_count == 1
    assert decide_mock.call_args.args[0].date == "2026-09-17"


def test_cmd_daily_refresh_aborts_on_unreadable_ledger(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    grid = _catchup_fixture(tmp_path, "{not valid json")
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = grid
    decide_mock = MagicMock(side_effect=AssertionError("decide must not run when the ledger is unreadable"))

    with (
        patch("src.cli.commands.pipeline.resolve_pipeline_target_session", MagicMock(return_value=date(2026, 9, 21))),
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(_catchup_args())

    assert rc == 1
    decide_mock.assert_not_called()


import os
import time


def test_heal_window_reaches_earliest_missing_session(tmp_path, caplog) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27), date(2026, 8, 31), date(2026, 9, 1)])
    _write_silver_dates(
        tmp_path, "index_daily", [date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31), date(2026, 9, 1)]
    )
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31), date(2026, 9, 1)]
    ingest_mock = MagicMock(return_value=0)
    features_mock = MagicMock(side_effect=AssertionError("features must not run past a stale gap"))
    args = argparse.Namespace(dataset=None, as_of="2026-09-01", lookback_days=1, decide=False, output_dir=None)

    with caplog.at_level("INFO"):
        with (
            patch("src.cli.commands.data.cmd_ingest", ingest_mock),
            patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
            patch("src.cli.commands.features.cmd_features", features_mock),
            patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("decide must not run"))),
            patch("src.core.calendar.get_calendar", return_value=cal_stub),
            patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
        ):
            rc = cmd_daily_refresh(args)

    assert rc == 1
    etf_call = next(c.args[0] for c in ingest_mock.call_args_list if c.args[0].dataset == "etf_daily")
    index_call = next(c.args[0] for c in ingest_mock.call_args_list if c.args[0].dataset == "kospi_index")
    assert (etf_call.start, etf_call.end) == ("2026-08-28", "2026-09-01")
    assert (index_call.start, index_call.end) == ("2026-08-28", "2026-09-01")
    features_mock.assert_not_called()
    assert any(
        "heal_start=2026-08-28" in record.message
        and "lookback_start=2026-08-31" in record.message
        and "missing_sessions=1" in record.message
        for record in caplog.records
    )


def test_resolve_ingest_start_never_shortens_lookback() -> None:
    from src.cli.commands.pipeline import resolve_ingest_start

    calendar = SimpleNamespace(sessions=lambda s, e: [date(2026, 8, 27)])
    start = resolve_ingest_start(
        date(2026, 8, 27),
        5,
        etf_dates=[date(2026, 8, 27)],
        index_dates=[date(2026, 8, 27)],
        calendar=calendar,
    )
    assert start == date(2026, 8, 22)


def test_stale_etf_gap_fails_closed_before_features(tmp_path, caplog) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27), date(2026, 8, 31)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)])
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)]
    features_mock = MagicMock(side_effect=AssertionError("features must not run past a stale gap"))
    args = argparse.Namespace(dataset=None, as_of="2026-08-31", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", features_mock),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("decide must not run"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1
    features_mock.assert_not_called()
    assert any(
        record.levelname == "ERROR" and "gate=etf_index_session_gap status=fail" in record.message
        for record in caplog.records
    )


def test_pending_latest_session_continues(tmp_path, caplog) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 9, 22)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 9, 22), date(2026, 9, 23)])
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 9, 22), date(2026, 9, 23)]
    features_mock = MagicMock(return_value=0)
    args = argparse.Namespace(dataset=None, as_of="2026-09-23", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", features_mock),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("decide must not run"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 0
    assert any(
        record.levelname == "WARNING" and "gate=etf_index_session_gap status=pending" in record.message
        for record in caplog.records
    )
    features_kwargs = features_mock.call_args.args[0]
    assert (features_kwargs.start, features_kwargs.end) == ("2026-09-22", "2026-09-23")


def test_missing_index_silver_fails_closed(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27)])
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    features_mock = MagicMock(side_effect=AssertionError("features must not run without index silver"))
    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", features_mock),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("decide must not run"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1
    features_mock.assert_not_called()


@pytest.fixture
def _old_repo_feature_inputs():
    from src.core.config import project_root

    root = project_root()
    targets = [root / "configs" / "features.yaml", *sorted((root / "src" / "features").rglob("*.py"))]
    saved = [(path, path.stat().st_mtime_ns) for path in targets]
    old_ns = time.time_ns() - 10_000_000_000
    for path in targets:
        os.utime(path, ns=(old_ns, old_ns))
    yield targets
    for path, mtime_ns in saved:
        os.utime(path, ns=(mtime_ns, mtime_ns))


def _write_gold(root) -> object:
    gold_dir = root / "features"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = gold_dir / "etf_features.parquet"
    pl.DataFrame({"date": [date(2026, 8, 27)]}, schema={"date": pl.Date}).write_parquet(gold_path)
    return gold_path


def _stale_pass_patches(cal_stub, data_root, features_mock):
    return (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", features_mock),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("decide must not run"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=data_root)),
    )


def test_fresh_gold_skips_rebuild(tmp_path, caplog, _old_repo_feature_inputs) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])
    old_ns = time.time_ns() - 10_000_000_000
    os.utime(tmp_path / "normalized" / "etf_daily.parquet", ns=(old_ns, old_ns))
    os.utime(tmp_path / "normalized" / "index_daily.parquet", ns=(old_ns, old_ns))
    _write_gold(tmp_path)
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    features_mock = MagicMock(side_effect=AssertionError("fresh gold must skip the rebuild"))
    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)
    patches = _stale_pass_patches(cal_stub, tmp_path, features_mock)

    with caplog.at_level("INFO"):
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            rc = cmd_daily_refresh(args)

    assert rc == 0
    features_mock.assert_not_called()
    assert any("features status=unchanged" in record.message for record in caplog.records)


def test_newer_silver_triggers_rebuild(tmp_path, _old_repo_feature_inputs) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])
    gold_path = _write_gold(tmp_path)
    old_ns = time.time_ns() - 10_000_000_000
    os.utime(gold_path, ns=(old_ns, old_ns))
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    features_mock = MagicMock(return_value=0)
    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)
    patches = _stale_pass_patches(cal_stub, tmp_path, features_mock)

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        rc = cmd_daily_refresh(args)

    assert rc == 0
    features_mock.assert_called_once()
    features_kwargs = features_mock.call_args.args[0]
    assert (features_kwargs.start, features_kwargs.end) == ("2026-08-27", "2026-08-27")


def test_force_rebuild_rebuilds_fresh_gold(tmp_path, _old_repo_feature_inputs) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])
    old_ns = time.time_ns() - 10_000_000_000
    os.utime(tmp_path / "normalized" / "etf_daily.parquet", ns=(old_ns, old_ns))
    os.utime(tmp_path / "normalized" / "index_daily.parquet", ns=(old_ns, old_ns))
    _write_gold(tmp_path)
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    features_mock = MagicMock(return_value=0)
    args = argparse.Namespace(
        dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None, force_rebuild=True
    )
    patches = _stale_pass_patches(cal_stub, tmp_path, features_mock)

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        rc = cmd_daily_refresh(args)

    assert rc == 0
    features_mock.assert_called_once()


def test_features_are_stale_tolerates_stat_race(tmp_path) -> None:
    from unittest.mock import MagicMock

    from src.cli.commands.pipeline import features_are_stale

    gold_path = tmp_path / "etf_features.parquet"
    gold_path.write_bytes(b"gold")
    flaky = MagicMock()
    flaky.is_file.return_value = True
    flaky.stat.side_effect = OSError("vanished")
    assert features_are_stale(gold_path, [flaky]) is False


def test_features_failure_fails_closed(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27)])
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27)]
    args = argparse.Namespace(dataset=None, as_of="2026-08-27", lookback_days=5, decide=False, output_dir=None)

    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=1)),
        patch("src.cli.commands.decide.cmd_decide", MagicMock(side_effect=AssertionError("decide must not run"))),
        patch("src.core.calendar.get_calendar", return_value=cal_stub),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
    ):
        rc = cmd_daily_refresh(args)

    assert rc == 1


def test_holiday_noop_returns_zero(tmp_path, _old_repo_feature_inputs) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    _write_silver_dates(tmp_path, "etf_daily", [date(2026, 8, 27), date(2026, 8, 28)])
    _write_silver_dates(tmp_path, "index_daily", [date(2026, 8, 27), date(2026, 8, 28)])
    old_ns = time.time_ns() - 10_000_000_000
    os.utime(tmp_path / "normalized" / "etf_daily.parquet", ns=(old_ns, old_ns))
    os.utime(tmp_path / "normalized" / "index_daily.parquet", ns=(old_ns, old_ns))
    _write_gold(tmp_path)
    cal_stub = MagicMock()
    cal_stub.sessions.return_value = [date(2026, 8, 27), date(2026, 8, 28)]
    features_mock = MagicMock(side_effect=AssertionError("holiday no-op must skip features"))
    args = argparse.Namespace(dataset=None, as_of="2026-08-29", lookback_days=5, decide=False, output_dir=None)
    patches = _stale_pass_patches(cal_stub, tmp_path, features_mock)

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        rc = cmd_daily_refresh(args)

    assert rc == 0
    features_mock.assert_not_called()
