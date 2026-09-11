from __future__ import annotations

import functools
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.config import clear_config_caches

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    krx_openapi_key: SecretStr
    fred_api: SecretStr | None = None
    ecos_api: SecretStr | None = None
    opendart_api_key: SecretStr | None = None
    krx_base_url: str = "https://data-dbg.krx.co.kr/svc/apis"
    data_root: Path = Path("data")
    log_root: Path = Path("logs")
    request_timeout_s: float = 30.0
    max_concurrency: int = 6
    requests_per_second: float = 5.0
    daily_call_quota: int = 8000
    calendar_name: str = "XKRX"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def clear_settings_caches() -> None:
    get_settings.cache_clear()
    clear_config_caches()
