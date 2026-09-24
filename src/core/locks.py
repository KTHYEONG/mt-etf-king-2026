"""Advisory data-plane lock serializing timer writers and readers."""

from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.05


class DataLockTimeout(RuntimeError):  # noqa: N818
    """Raised when the data lock could not be acquired within the configured timeout."""


@contextmanager
def data_lock(lock_path: Path, *, shared: bool, timeout_s: float) -> Iterator[None]:
    """Hold the advisory data-plane lock for the duration of the block.

    Writers of bronze/silver/gold take it exclusively; timer jobs that only read silver take it shared, so a reader
    never sees silver and gold from two different pipeline runs and two writers never interleave.

    Args:
        lock_path: Lock file (created if missing, never deleted).
        shared: True for `LOCK_SH`, False for `LOCK_EX`.
        timeout_s: Maximum wait in seconds before giving up.

    Raises:
        DataLockTimeout: the lock is still held by another process after `timeout_s`.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    mode_name = "shared" if shared else "exclusive"
    handle = open(lock_path, "a")  # noqa: PTH123, SIM115
    start = time.monotonic()
    contended = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
                break
            except OSError:
                contended = True
                if time.monotonic() - start >= timeout_s:
                    logger.error(
                        f"[SYS] data_lock status=timeout mode={mode_name} path={lock_path} timeout_s={timeout_s}"
                    )
                    raise DataLockTimeout(
                        f"data lock timeout mode={mode_name} path={lock_path} timeout_s={timeout_s}"
                    ) from None
                time.sleep(_POLL_INTERVAL_S)
        if contended:
            logger.info(f"[SYS] data_lock mode={mode_name} waited_s={time.monotonic() - start:.3f}")
        yield
    finally:
        handle.close()
