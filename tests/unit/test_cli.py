from __future__ import annotations

import logging

import pytest

from src.cli import SUBCOMMANDS, main
from src.core.settings import clear_settings_caches


def test_cli_calendar_and_unknown(caplog: pytest.LogCaptureFixture) -> None:
    """SCENARIO-01-07 SCENARIO-06-12: cli calendar 및 알 수 없는 명령어 처리."""
    assert "config-check" in SUBCOMMANDS
    assert "calendar" in SUBCOMMANDS

    caplog.set_level(logging.INFO)
    ret = main(["calendar", "--start", "2026-09-21", "--end", "2026-11-13"])
    assert ret == 0
    # logs should contain session_count=36
    assert any("session_count=36" in m for m in caplog.messages)

    # unknown command returns non-zero without raising
    ret2 = main(["no-such-command"])
    assert ret2 != 0


def test_cli_config_check_hides_secret(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """SCENARIO-01-08: config-check 비밀값 비노출."""
    monkeypatch.setenv("KRX_OPENAPI_KEY", "SECRET123")
    clear_settings_caches()
    caplog.set_level(logging.INFO)
    caplog.clear()
    ret = main(["config-check"])
    assert ret == 0
    combined = "\n".join(caplog.messages)
    assert "krx_openapi_key=True" in combined
    assert "SECRET123" not in combined
    clear_settings_caches()


def test_scenario_02_10_ingest_dry_run(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """SCENARIO-02-10."""
    import httpx

    monkeypatch.setenv("KRX_OPENAPI_KEY", "TESTKEY123")
    clear_settings_caches()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"OutBlock_1": []})

    transport = httpx.MockTransport(handler)
    # Patch httpx.AsyncClient to use mock, but dry-run should not call it anyway
    original_client = httpx.AsyncClient

    class PatchedClient(httpx.AsyncClient):  # type: ignore[misc]
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            kw.setdefault("transport", transport)
            super().__init__(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)

    assert "ingest" in SUBCOMMANDS

    caplog.set_level(logging.INFO)
    ret = main(["ingest", "--dataset", "etf_daily", "--start", "2026-08-13", "--end", "2026-08-20", "--dry-run"])
    assert ret == 0
    assert len(calls) == 0
    clear_settings_caches()


def test_ingest_wires_rate_limiter(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """ingest 실제 실행 시 RateLimiter가 provider에 전달되는지 확인."""
    import httpx

    from src.data.providers.krx import KRXOpenAPIProvider
    from src.data.providers.ratelimit import RateLimiter

    monkeypatch.setenv("KRX_OPENAPI_KEY", "TESTKEY123")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("REQUESTS_PER_SECOND", "2.5")
    clear_settings_caches()

    limiter_args: list[float] = []
    provider_kwargs: list[dict[str, object]] = []

    class SpyRateLimiter(RateLimiter):
        def __init__(self, requests_per_second: float, **kwargs: object) -> None:
            limiter_args.append(requests_per_second)
            super().__init__(requests_per_second, **kwargs)

    original_init = KRXOpenAPIProvider.__init__

    def capture_provider_init(self, *args: object, **kwargs: object) -> None:
        provider_kwargs.append(dict(kwargs))
        original_init(self, *args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"OutBlock_1": [{"BAS_DD": "20260813"}]})

    transport = httpx.MockTransport(handler)

    class PatchedClient(httpx.AsyncClient):  # type: ignore[misc]
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            kw.setdefault("transport", transport)
            super().__init__(*a, **kw)

    monkeypatch.setattr("src.data.providers.ratelimit.RateLimiter", SpyRateLimiter)
    monkeypatch.setattr(KRXOpenAPIProvider, "__init__", capture_provider_init)
    monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)

    ret = main(["ingest", "--dataset", "etf_daily", "--start", "2026-08-13", "--end", "2026-08-13"])
    assert ret == 0
    assert limiter_args == [2.5]
    assert provider_kwargs
    assert isinstance(provider_kwargs[0].get("limiter"), SpyRateLimiter)
    clear_settings_caches()


def test_scenario_03_09_normalize_cli(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """SCENARIO-03-09"""
    from datetime import date, datetime, UTC

    from src.core.paths import DataPaths
    from src.data.bronze import BronzeRecord, BronzeStore

    monkeypatch.setenv("KRX_OPENAPI_KEY", "TESTKEY123")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    clear_settings_caches()
    assert "normalize" in SUBCOMMANDS
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)

    def _row(ticker: str, price: int, dstr: str) -> dict[str, str]:
        return {
            "BAS_DD": dstr,
            "ISU_CD": ticker,
            "ISU_NM": "Test",
            "TDD_CLSPRC": str(price),
            "NAV": str(price),
            "TDD_OPNPRC": str(price),
            "TDD_HGPRC": str(price + 1),
            "TDD_LWPRC": str(price - 1),
            "ACC_TRDVOL": "1000",
            "ACC_TRDVAL": "1000000",
            "MKTCAP": str(price * 1000),
            "INVSTASST_NETASST_TOTAMT": str(price * 1000),
            "LIST_SHRS": "1000",
            "IDX_IND_NM": "Index",
            "OBJ_STKPRC_IDX": "1000",
        }

    for d in [date(2026, 8, 14), date(2026, 8, 18)]:
        rows = [_row("451060", 35000, d.strftime("%Y%m%d")), _row("069500", 30000, d.strftime("%Y%m%d"))]
        rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=2, rows=rows)
        store.write(rec)
    ret = main(["normalize", "--dataset", "etf_daily", "--mode", "full"])
    assert ret == 0
    # fatal case
    from unittest import mock

    from src.data.validation import Severity, ValidationIssue, ValidationReport

    def fatal_validate(self, dataset: str, frame):  # type: ignore[no-untyped-def]
        return ValidationReport(dataset=dataset, rows=frame.height, sessions=0, issues=(ValidationIssue(gate="V2", severity=Severity.CRITICAL, count=1, detail="fatal"),))

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "fatal"))
    clear_settings_caches()
    paths2 = DataPaths(root=tmp_path / "fatal")
    store2 = BronzeStore(paths2)
    d = date(2026, 8, 14)
    rows = [_row("451060", 35000, d.strftime("%Y%m%d"))]
    rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=1, rows=rows)
    store2.write(rec)
    with mock.patch("src.data.validation.PanelValidator.validate", fatal_validate):
        ret2 = main(["normalize", "--dataset", "etf_daily", "--mode", "full"])
        assert ret2 != 0
    clear_settings_caches()


def test_scenario_04_11_universe_cli(monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    """SCENARIO-04-11"""
    from datetime import date

    import polars as pl

    from src.core.paths import DataPaths
    from src.core.settings import clear_settings_caches

    assert "universe" in SUBCOMMANDS
    monkeypatch.setenv("KRX_OPENAPI_KEY", "TESTKEY123")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    clear_settings_caches()

    paths = DataPaths(root=tmp_path)
    silver_path = paths.silver("etf_daily")
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": [date(2026, 8, 27)],
            "ticker": ["069500"],
            "name": ["KODEX 200"],
            "underlying_index_name": ["코스피 200"],
            "is_tradable": [True],
            "close": [30000.0],
            "trading_value": [2e12],
        }
    ).write_parquet(silver_path)

    caplog.set_level(logging.INFO)
    ret = main(["universe", "--date", "2026-08-27", "--mode", "deployment", "--max-order-to-adv", "0.05"])
    assert ret == 0
    combined = "\n".join(caplog.messages)
    assert "[DATA]" in combined
    assert "mode=deployment" in combined
    assert "existence=" in combined
    assert "price=" in combined
    assert "history=" in combined
    assert "sponsor=" in combined
    assert "liquidity=" in combined
    assert "eligibility=" in combined
    clear_settings_caches()


def test_scenario_05_11_features_cli(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """SCENARIO-05-11"""
    from datetime import date

    import polars as pl

    from src.core.paths import DataPaths
    from src.core.settings import clear_settings_caches

    assert "features" in SUBCOMMANDS
    monkeypatch.setenv("KRX_OPENAPI_KEY", "TESTKEY123")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    clear_settings_caches()

    paths = DataPaths(root=tmp_path)
    silver_path = paths.silver("etf_daily")
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    sessions = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
    rows = [
        {
            "date": d,
            "ticker": ticker,
            "name": "Test ETF",
            "close": 30_000.0,
            "open": 29_900.0,
            "high": 30_100.0,
            "low": 29_800.0,
            "nav": 29_950.0,
            "shares_outstanding": 1_000_000,
            "net_assets": 30_000_000_000,
            "trading_value": 2_000_000_000,
            "underlying_index_name": "코스피 200",
            "is_tradable": True,
        }
        for d in sessions
        for ticker in ["069500", "451060"]
    ]
    pl.DataFrame(rows).write_parquet(silver_path)

    ret = main(["features", "--start", "2026-01-02", "--end", "2026-08-27"])
    assert ret == 0
    gold_path = paths.gold("etf_features")
    assert gold_path.exists()
    clear_settings_caches()


def test_scenario_06_12_backtest_and_replay_cli(monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    """SCENARIO-06-12"""
    from datetime import date

    import polars as pl

    from src.core.paths import DataPaths

    assert "backtest" in SUBCOMMANDS
    assert "replay" in SUBCOMMANDS

    monkeypatch.setenv("KRX_OPENAPI_KEY", "TESTKEY123")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    clear_settings_caches()

    paths = DataPaths(root=tmp_path)
    sessions = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    rows = [
        {
            "date": d,
            "ticker": ticker,
            "name": "Test ETF",
            "close": 30_000.0,
            "open": 29_900.0,
            "high": 30_100.0,
            "low": 29_800.0,
            "is_tradable": True,
            "trading_value": 5_000_000_000,
            "underlying_index_name": "IndexA",
            "mom_20": 0.01,
        }
        for d in sessions
        for ticker in ["069500", "451060"]
    ]
    gold_path = paths.gold("etf_features")
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(gold_path)

    caplog.set_level(logging.INFO)
    ret = main(["backtest", "--model", "B1", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert ret == 0
    combined = "\n".join(caplog.messages)
    assert "n_effective=" in combined
    assert "participation=" in combined
    assert "commission_bps=" in combined
    assert "[EVAL]" in combined

    ret_bad = main(["backtest", "--model", "B9", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert ret_bad != 0
    clear_settings_caches()
