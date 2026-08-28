from __future__ import annotations

import logging

import pytest

from src.cli import SUBCOMMANDS, main
from src.core.settings import clear_settings_caches


def test_cli_calendar_and_unknown(caplog: pytest.LogCaptureFixture) -> None:
    """SCENARIO-01-07: cli calendar 및 알 수 없는 명령어 처리."""
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
