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
