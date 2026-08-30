from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

LOG_TAGS: Final[frozenset[str]] = frozenset({"SYS", "DATA", "ALGO", "EVAL"})

_LOGGER_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _format_value(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        if len(v) <= 5:
            # format each element with _format_value for floats, else str
            parts = []
            for item in v:
                if isinstance(item, float):
                    parts.append(f"{item:.3f}")
                else:
                    parts.append(str(item))
            return "[" + ", ".join(parts) + "]"
        else:
            shown = v[:5]
            remainder = len(v) - 5
            parts = []
            for item in shown:
                if isinstance(item, float):
                    parts.append(f"{item:.3f}")
                else:
                    parts.append(str(item))
            return "[" + ", ".join(parts) + f", truncated={remainder}]"
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
    # Preserve pytest caplog handlers so tests can capture logs
    caplog_handlers = [h for h in root_logger.handlers if "LogCapture" in type(h).__name__ or "CapLog" in type(h).__name__]
    for h in list(root_logger.handlers):
        if h not in caplog_handlers:
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
