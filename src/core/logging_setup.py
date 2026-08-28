from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

LOG_TAGS: Final[frozenset[str]] = frozenset({"SYS", "DATA", "ALGO", "EVAL"})

_LOGGER_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _format_value(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def tagged_log(logger: logging.Logger, tag: str, **kwargs: object) -> None:
    if tag not in LOG_TAGS:
        raise ValueError(f"invalid LOG_TAGS {tag}")
    parts = [f"{k}={_format_value(v)}" for k, v in kwargs.items()]
    msg = f"[{tag}] " + " ".join(parts)
    logger.debug(msg)


def configure_logging(
    level: str = "INFO",
    log_root: Path | None = None,
    stream_name: str = "sys",
) -> None:
    if log_root is None:
        log_root = Path("logs")
    root_logger = logging.getLogger()
    # Parse level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(min(numeric_level, logging.DEBUG))

    # Remove previously added handlers to avoid duplication
    # Keep only handlers we manage: identify by type/name approach - remove all and re-add
    # But to satisfy idempotency (exactly 2 handlers after repeated calls), clear all
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    formatter = logging.Formatter(_LOGGER_FORMAT)

    # Console handler at requested level
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler at DEBUG
    try:
        log_root.mkdir(parents=True, exist_ok=True)
        file_path = log_root / f"{stream_name}.log"
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError:
        # Log directory creation failure -> continue with console only
        pass
