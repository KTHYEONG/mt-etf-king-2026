"""Atomic file publication for timer-written data artifacts."""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path

import polars as pl


def _fresh_temp(path: Path) -> Path:
    token = secrets.token_hex(8)
    return path.parent / f".{path.name}.tmp-{os.getpid()}-{token}"


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace `path` with `data`.

    Readers and crash recovery observe either the previous file or the complete new one, never a torn write. The
    temp file lives in the destination directory so the final `os.replace` stays on one filesystem.

    Raises:
        OSError: the write, fsync, or replace failed; the destination is left untouched and the temp file removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _fresh_temp(path)
    try:
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    except Exception:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """UTF-8 variant of `atomic_write_bytes` for JSON/Markdown artifacts and state files."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Atomically replace `path` with `frame` as zstd parquet (pyarrow encoder, as the existing panel writers use).

    Raises:
        OSError: as `atomic_write_bytes`; encoder errors propagate after the temp file is removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _fresh_temp(path)
    try:
        frame.write_parquet(str(temp), compression="zstd", use_pyarrow=True)
        fd = os.open(temp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
        _fsync_dir(path.parent)
    except Exception:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise
