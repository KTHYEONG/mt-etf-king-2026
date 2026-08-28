from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import Settings, clear_settings_caches, get_settings
from src.core.sops_env import (
    SopsDecryptError,
    SopsDotEnvSettingsSource,
    decrypt_env_enc,
    resolve_env_enc_path,
)


def test_resolve_env_enc_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MT_ETF_ENV_ENC", raising=False)
    assert resolve_env_enc_path() == Path(".env.enc")


def test_decrypt_env_enc_missing_file(tmp_path: Path) -> None:
    decrypt_env_enc.cache_clear()
    with pytest.raises(SopsDecryptError, match="encrypted env file not found"):
        decrypt_env_enc(tmp_path / "missing.env.enc")


def test_sops_dotenv_settings_source_maps_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_decrypt(_path: Path) -> str:
        return 'KRX_OPENAPI_KEY="SECRET123"\n'

    monkeypatch.setattr("src.core.sops_env.decrypt_env_enc", fake_decrypt)
    source = SopsDotEnvSettingsSource(Settings, env_enc_path=Path(".env.enc"))
    assert source() == {"krx_openapi_key": "SECRET123"}


def test_settings_loads_from_sops_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    enc_path = tmp_path / "test.env.enc"
    enc_path.write_text("placeholder", encoding="utf-8")

    def fake_decrypt(_path: Path) -> str:
        return 'KRX_OPENAPI_KEY="SECRET123"\n'

    monkeypatch.setenv("MT_ETF_ENV_ENC", str(enc_path))
    monkeypatch.delenv("KRX_OPENAPI_KEY", raising=False)
    clear_settings_caches()
    monkeypatch.setattr("src.core.sops_env.decrypt_env_enc", fake_decrypt)
    settings = Settings()
    assert settings.krx_openapi_key.get_secret_value() == "SECRET123"
    get_settings.cache_clear()
