from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.settings import Settings, get_settings


def test_missing_key_raises_and_secret_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-01-04: Settings 비밀키 fail-closed 및 노출 방지."""
    # Ensure key absent
    monkeypatch.delenv("KRX_OPENAPI_KEY", raising=False)
    monkeypatch.delenv("KRX_OPENAPI_KEY ", raising=False)
    # Clear memoised cache
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings()

    # With key set
    monkeypatch.setenv("KRX_OPENAPI_KEY", "SECRET123")
    get_settings.cache_clear()
    s = Settings()
    assert s.krx_openapi_key.get_secret_value() == "SECRET123"
    assert "SECRET123" not in repr(s)
    assert "SECRET123" not in str(s)
    assert "SECRET" not in repr(s)
    assert "SECRET" not in str(s)
    # get_settings memoised
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()
