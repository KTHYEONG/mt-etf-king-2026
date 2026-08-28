from __future__ import annotations

import pytest
from pydantic import ValidationError

from pathlib import Path

from src.core.settings import Settings, clear_settings_caches, get_settings


def test_missing_key_raises_and_secret_hidden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-01-04: Settings 비밀키 fail-closed 및 노출 방지."""
    missing_enc = tmp_path / "missing.env.enc"
    monkeypatch.setenv("MT_ETF_ENV_ENC", str(missing_enc))
    # Ensure key absent
    monkeypatch.delenv("KRX_OPENAPI_KEY", raising=False)
    monkeypatch.delenv("KRX_OPENAPI_KEY ", raising=False)
    clear_settings_caches()
    with pytest.raises(ValidationError):
        Settings()

    # With key set
    monkeypatch.setenv("KRX_OPENAPI_KEY", "SECRET123")
    clear_settings_caches()
    s = Settings()
    assert s.krx_openapi_key.get_secret_value() == "SECRET123"
    assert "SECRET123" not in repr(s)
    assert "SECRET123" not in str(s)
    assert "SECRET" not in repr(s)
    assert "SECRET" not in str(s)
    # get_settings memoised
    clear_settings_caches()
    a = get_settings()
    b = get_settings()
    assert a is b
    clear_settings_caches()
