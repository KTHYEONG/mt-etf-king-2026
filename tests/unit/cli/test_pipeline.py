# ruff: noqa
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import argparse
import polars as pl


def test_cmd_daily_refresh_happy_path_without_decide(tmp_path) -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        {"date": [date(2018, 1, 2), date(2026, 8, 20), date(2026, 8, 27)]},
        schema={"date": pl.Date},
    ).write_parquet(silver_dir / "etf_daily.parquet")

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
    ingest_kwargs = ingest_mock.call_args.args[0]
    assert ingest_kwargs.dataset == "etf_daily"
    assert ingest_kwargs.start == "2026-08-22"
    assert ingest_kwargs.end == "2026-08-27"
    assert ingest_kwargs.dry_run is False

    normalize_kwargs = normalize_mock.call_args.args[0]
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

