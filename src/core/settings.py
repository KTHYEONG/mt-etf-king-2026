from __future__ import annotations

import functools
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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
    daily_call_quota: int = 8000
    calendar_name: str = "XKRX"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
