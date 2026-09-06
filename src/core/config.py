"""Single source of truth for YAML config loading (fail-closed, CWD-independent)."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a config file is missing, unreadable, malformed, or mistyped."""


@functools.lru_cache(maxsize=1)
def project_root() -> Path:
    """Walk up from this file until the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    while True:
        if (current / "pyproject.toml").is_file():
            return current
        parent = current.parent
        if parent == current:
            raise ConfigError("pyproject.toml not found walking up from src/core/config.py")
        current = parent


def config_path(name: str) -> Path:
    """Resolve `configs/<name>.yaml` under the project root."""
    if "/" in name or "\\" in name or ".." in name or not name:
        raise ConfigError(f"invalid config name: {name!r}")
    return project_root() / "configs" / f"{name}.yaml"


@functools.lru_cache
def load_config(name: str) -> Mapping[str, Any]:
    """Parse a config file once per process; fail closed on any problem."""
    path = config_path(name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"config file missing or unreadable: {path}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file does not parse: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file must parse to a mapping: {path}")
    return data


def _check_value_type(value: Any, default: object, location: str) -> None:
    if isinstance(default, bool) or isinstance(value, bool):
        if type(value) is not type(default):
            raise ConfigError(f"config value at {location} has wrong type: {type(value).__name__}")
        return
    if isinstance(default, float) and isinstance(value, (int, float)):
        return
    if isinstance(value, type(default)):
        return
    raise ConfigError(f"config value at {location} has wrong type: {type(value).__name__}")


def config_value(name: str, *keys: str, default: object | None = None, required: bool = False) -> Any:
    """Nested lookup under a config file with fail-closed mistype semantics."""
    data: Any = load_config(name)
    location = name
    for key in keys:
        location = f"{location}.{key}"
        if not isinstance(data, Mapping) or key not in data:
            if required:
                raise ConfigError(f"required config key missing: {location}")
            return default
        data = data[key]
    if data is None and default is not None:
        if required:
            raise ConfigError(f"required config key missing: {location}")
        return default
    if default is not None:
        _check_value_type(data, default, location)
    return data


def clear_config_caches() -> None:
    """Clear the project_root and load_config caches (tests + settings)."""
    project_root.cache_clear()
    load_config.cache_clear()
