"""Invariant guards for atomic file publication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.core.atomic_io import atomic_write_bytes, atomic_write_parquet, atomic_write_text


def test_replace_publishes_complete_content(tmp_path: Path) -> None:
    """A text write replaces the file exactly and leaves no temp file behind."""
    target = tmp_path / "card.json"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new\n")
    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_write_leaves_original_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An encoder failure propagates with the original bytes and no temp file."""
    target = tmp_path / "panel.parquet"
    frame = pl.DataFrame({"a": [1, 2]})
    frame.write_parquet(target, compression="zstd")
    before = target.read_bytes()

    def _boom(self: object, *args: Any, **kwargs: Any) -> None:
        raise ValueError("encoder down")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _boom)
    with pytest.raises(ValueError, match="encoder down"):
        atomic_write_parquet(frame, target)
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["panel.parquet"]


def test_temp_names_invisible_to_data_globs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted publish leaves only a hidden temp file data globs never match."""
    target = tmp_path / "20260827.json.gz"
    target.write_bytes(b"canonical")

    def _boom(src: object, dst: object) -> None:
        raise OSError("replace down")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="replace down"):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"canonical"
    assert [p.name for p in tmp_path.glob("*.json.gz")] == ["20260827.json.gz"]
    leftovers = [p for p in tmp_path.iterdir() if p.name != target.name]
    assert leftovers == []
    assert [p.name for p in tmp_path.glob("*.parquet")] == []
