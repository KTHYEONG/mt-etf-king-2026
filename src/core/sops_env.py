from __future__ import annotations

import functools
import os
import shutil
import subprocess
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values
from pydantic_settings import BaseSettings
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource
from pydantic_settings.sources.utils import parse_env_vars

ENV_ENC_PATH_VAR = "MT_ETF_ENV_ENC"
DEFAULT_ENV_ENC_PATH = Path(".env.enc")


class SopsDecryptError(RuntimeError):
    """Raised when sops cannot decrypt the encrypted env file."""


def resolve_env_enc_path() -> Path:
    return Path(os.environ.get(ENV_ENC_PATH_VAR, DEFAULT_ENV_ENC_PATH))


@functools.lru_cache(maxsize=4)
def decrypt_env_enc(path: Path) -> str:
    if not path.is_file():
        raise SopsDecryptError(f"encrypted env file not found: {path}")
    sops_bin = shutil.which("sops")
    if sops_bin is None:
        raise SopsDecryptError("sops executable not found on PATH")
    result = subprocess.run(  # noqa: S603
        [
            sops_bin,
            "-d",
            "--input-type",
            "dotenv",
            "--output-type",
            "dotenv",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "unknown sops error").strip()
        raise SopsDecryptError(f"sops decrypt failed: {msg}")
    return result.stdout


class SopsDotEnvSettingsSource(DotEnvSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], env_enc_path: Path | None = None) -> None:
        self._env_enc_path = env_enc_path or resolve_env_enc_path()
        super().__init__(settings_cls, env_file=None)

    def _read_env_files(self) -> Mapping[str, str | None]:
        try:
            content = decrypt_env_enc(self._env_enc_path)
        except SopsDecryptError:
            return {}
        raw = dotenv_values(stream=StringIO(content))
        file_vars = {k: v for k, v in raw.items() if k is not None and v is not None}
        return parse_env_vars(
            file_vars,
            case_sensitive=self.case_sensitive,
            ignore_empty=self.env_ignore_empty,
            parse_none_str=self.env_parse_none_str,
        )


def clear_env_caches() -> None:
    decrypt_env_enc.cache_clear()
