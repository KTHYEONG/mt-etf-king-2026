"""Invariant guards for the advisory data-plane lock."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.core.locks import DataLockTimeout, data_lock


def test_exclusive_excludes_exclusive(tmp_path: Path) -> None:
    """A second exclusive acquisition on a held lock times out."""
    lock = tmp_path / "pipeline.lock"
    with data_lock(lock, shared=False, timeout_s=5):  # noqa: SIM117
        with pytest.raises(DataLockTimeout):
            with data_lock(lock, shared=False, timeout_s=0.2):
                pass  # pragma: no cover


def test_shared_coexists_with_shared(tmp_path: Path) -> None:
    """Two shared holders coexist without waiting."""
    lock = tmp_path / "pipeline.lock"
    with data_lock(lock, shared=True, timeout_s=5), data_lock(lock, shared=True, timeout_s=5):
        pass


def test_shared_blocks_exclusive(tmp_path: Path) -> None:
    """An exclusive acquisition on a shared-held lock times out."""
    lock = tmp_path / "pipeline.lock"
    with data_lock(lock, shared=True, timeout_s=5):  # noqa: SIM117
        with pytest.raises(DataLockTimeout):
            with data_lock(lock, shared=False, timeout_s=0.2):
                pass  # pragma: no cover


def test_release_on_exception(tmp_path: Path) -> None:
    """The lock is released when the block raises; the lock file is never deleted."""
    lock = tmp_path / "pipeline.lock"
    with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117
        with data_lock(lock, shared=False, timeout_s=5):
            raise RuntimeError("boom")
    with data_lock(lock, shared=False, timeout_s=5):
        pass
    assert lock.exists()


def test_wait_logs_info_once_contended(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An acquisition that had to wait logs one INFO line with the wait time."""
    import time

    lock = tmp_path / "pipeline.lock"
    acquired = threading.Event()

    def _holder() -> None:
        with data_lock(lock, shared=False, timeout_s=5):
            acquired.set()
            time.sleep(0.5)

    thread = threading.Thread(target=_holder)
    thread.start()
    try:
        assert acquired.wait(timeout=10)
        caplog.set_level("INFO", logger="src.core.locks")
        with data_lock(lock, shared=False, timeout_s=10):
            pass
    finally:
        thread.join(timeout=10)
    assert any(
        record.levelname == "INFO" and "data_lock mode=exclusive waited_s=" in record.message
        for record in caplog.records
    )
