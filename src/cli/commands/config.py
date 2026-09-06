# ruff: noqa
from __future__ import annotations

import argparse
import logging
from datetime import date

from src.core.calendar import get_calendar
from src.core.paths import DataPaths
from src.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        # Prefer cached settings; fallback to direct construction for validation
        settings = get_settings()
    except Exception:
        try:
            settings = Settings()  # type: ignore[call-arg]
        except Exception as exc:
            logger.error(f"[SYS] config_check status=fail error={exc!r}")
            return 1
    # Validate DataPaths
    try:
        paths = DataPaths(root=settings.data_root)
        # Probe paths without creating dirs (exercise bronze/silver/gold/state)
        paths.bronze("etp/etf_bydd_trd", date(2026, 8, 27))
        paths.silver("probe")
        paths.gold("probe")
        paths.state("probe")
        # Reference configure_logging to satisfy wiring without side effect
    except Exception as exc:
        logger.error(f"[SYS] config_check status=fail error={exc!r}")
        return 1
    krx_present = bool(settings.krx_openapi_key.get_secret_value())
    fred_present = bool(settings.fred_api.get_secret_value()) if settings.fred_api else False
    ecos_present = bool(settings.ecos_api.get_secret_value()) if settings.ecos_api else False
    dart_present = bool(settings.opendart_api_key.get_secret_value()) if settings.opendart_api_key else False
    logger.info(
        "[SYS] config_check status=ok "
        f"krx_openapi_key={krx_present} fred_api={fred_present} "
        f"ecos_api={ecos_present} opendart_api_key={dart_present} "
        f"data_root={settings.data_root} log_root={settings.log_root}"
    )
    return 0


def cmd_calendar(args: argparse.Namespace) -> int:
    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    except Exception as exc:
        logger.error(f"[SYS] calendar status=fail error={exc!r}")
        return 1
    try:
        cal = get_calendar()
        count = cal.session_count(start, end)
    except Exception as exc:
        logger.error(f"[SYS] calendar status=fail error={exc!r}")
        return 1
    logger.info(f"[SYS] calendar start={start} end={end} session_count={count}")
    return 0
