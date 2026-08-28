from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

from src.core.logging_setup import LOG_TAGS, configure_logging


def test_configure_logging_handlers_and_format(tmp_path: Path) -> None:
    """SCENARIO-01-06: configure_logging 핸들러 수 및 포맷 검증."""
    # Ensure clean root logger
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    configure_logging(log_root=tmp_path)
    configure_logging(log_root=tmp_path)
    assert len(root.handlers) == 2

    # Verify LOG_TAGS
    assert frozenset({"SYS", "DATA", "ALGO", "EVAL"}) == LOG_TAGS

    # Emit debug record
    log = logging.getLogger("test_logging")
    # Ensure level allows debug to file handler (file handler is DEBUG)
    # Root handlers include file handler at DEBUG, console may be INFO but file will capture
    log.debug("[SYS] stage=probe elapsed_ms=12")

    # Force flush
    for h in root.handlers:
        h.flush()

    content = (tmp_path / "sys.log").read_text(encoding="utf-8")
    # Find line containing [SYS]
    found = False
    pattern = re.compile(r"^\[(SYS|DATA|ALGO|EVAL)\]( \w+=\S+)+$")
    for line in content.splitlines():
        if "[SYS] stage=probe elapsed_ms=12" in line:
            # Strip formatter prefix: find '['
            idx = line.find("[")
            stripped = line[idx:]
            assert pattern.match(stripped), f"pattern mismatch: {stripped}"
            found = True
            break
    assert found, f"log file missing expected record: {content!r}"

    # Cleanup: remove handlers to not pollute other tests
    for h in list(root.handlers):
        root.removeHandler(h)
        with contextlib.suppress(Exception):
            h.close()
