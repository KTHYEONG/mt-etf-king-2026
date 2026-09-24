"""Daily-refresh registration in the main dispatch registry."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli.main  # noqa: F401

main_module = sys.modules["src.cli.main"]


def test_main_registry_routes_daily_refresh() -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh
    from src.cli.main import SUBCOMMANDS

    assert SUBCOMMANDS["daily-refresh"] is cmd_daily_refresh


def test_main_and_parser_registries_agree_on_daily_refresh() -> None:
    from src.cli.main import SUBCOMMANDS as MAIN_SUBCOMMANDS
    from src.cli.parser import SUBCOMMANDS

    assert "daily-refresh" in SUBCOMMANDS
    assert "daily-refresh" in MAIN_SUBCOMMANDS


def test_data_writers_run_under_exclusive_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """daily-refresh dispatches inside one exclusive acquisition on the pipeline lock."""
    seen: dict[str, Any] = {"calls": [], "acquisitions": []}

    @contextmanager
    def _record(lock_path: Path, *, shared: bool, timeout_s: float) -> Iterator[None]:
        seen["acquisitions"].append((lock_path, shared, timeout_s))
        yield

    def _stub(args: argparse.Namespace) -> int:
        seen["calls"].append(args)
        return 0

    monkeypatch.setattr(main_module, "data_lock", _record)
    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setitem(main_module.SUBCOMMANDS, "daily-refresh", _stub)
    assert main_module.main(["daily-refresh"]) == 0
    assert seen["calls"] != []
    assert seen["acquisitions"] == [(tmp_path / "state" / "pipeline.lock", False, 900.0)]


def test_timer_readers_share_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """contest-weekly dispatches inside one shared acquisition on the pipeline lock."""
    seen: dict[str, Any] = {"acquisitions": []}

    @contextmanager
    def _record(lock_path: Path, *, shared: bool, timeout_s: float) -> Iterator[None]:
        seen["acquisitions"].append((lock_path, shared))
        yield

    monkeypatch.setattr(main_module, "data_lock", _record)
    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setitem(main_module.SUBCOMMANDS, "contest-weekly", lambda args: 0)
    assert main_module.main(["contest-weekly"]) == 0
    assert seen["acquisitions"] == [(tmp_path / "state" / "pipeline.lock", True)]


def test_lock_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lock timeout returns 1 without invoking the handler."""
    from src.core.locks import DataLockTimeout

    @contextmanager
    def _timeout(lock_path: Path, *, shared: bool, timeout_s: float) -> Iterator[None]:
        raise DataLockTimeout("held")
        yield  # pragma: no cover

    calls: list[argparse.Namespace] = []

    def _stub(args: argparse.Namespace) -> int:
        calls.append(args)
        return 0  # pragma: no cover

    monkeypatch.setattr(main_module, "data_lock", _timeout)
    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setitem(main_module.SUBCOMMANDS, "normalize", _stub)
    assert main_module.main(["normalize", "--dataset", "etf_daily"]) == 1
    assert calls == []


def test_unlocked_commands_stay_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """calendar dispatches without touching the data lock."""
    acquisitions: list[tuple[Path, bool]] = []

    def _boom(*args: Any, **kwargs: Any) -> Any:
        acquisitions.append((args[0], kwargs.get("shared", False)))
        raise AssertionError("data_lock must not be called")

    monkeypatch.setattr(main_module, "data_lock", _boom)
    assert main_module.main(["calendar", "--start", "2026-08-01", "--end", "2026-08-27"]) == 0
    assert acquisitions == []
