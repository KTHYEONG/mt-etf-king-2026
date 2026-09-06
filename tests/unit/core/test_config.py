import os
from pathlib import Path

import pytest


def test_load_config_is_fail_closed() -> None:
    from src.core.config import ConfigError, load_config

    with pytest.raises(ConfigError):
        load_config("definitely_not_a_real_config_name")


def test_config_value_wrong_type_raises_but_absent_key_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core import config as cfg

    cfg.clear_config_caches()
    monkeypatch.setattr(cfg, "load_config", lambda name: {"portfolio": {"family_peak_lock": {"lock_level": "not-a-number"}}})

    # Present but wrong type -> fail closed
    with pytest.raises(cfg.ConfigError):
        cfg.config_value("strategies", "portfolio", "family_peak_lock", "lock_level", default=0.5)

    # Absent key -> documented default
    assert cfg.config_value("strategies", "portfolio", "family_peak_lock", "missing_key", default=0.5) == 0.5

    # Absent key + required -> fail closed
    with pytest.raises(cfg.ConfigError):
        cfg.config_value("strategies", "portfolio", "family_peak_lock", "missing_key", required=True)


def test_config_is_cwd_independent_and_parsed_once(tmp_path: Path) -> None:
    from src.core.config import clear_config_caches, config_path, load_config, project_root

    assert (project_root() / "pyproject.toml").exists()

    clear_config_caches()
    first = load_config("strategies")
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        second = load_config("strategies")
    finally:
        os.chdir(cwd)

    assert second is first, "load_config must be cached, not re-parsed per call"
    assert config_path("strategies").is_absolute()
    assert isinstance(first, dict) and first
