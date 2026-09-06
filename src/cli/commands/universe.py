# ruff: noqa
from __future__ import annotations

import argparse
import logging
from datetime import date

from src.core.config import config_path

logger = logging.getLogger(__name__)


def cmd_universe(args: argparse.Namespace) -> int:
    try:
        date_str = getattr(args, "date", None)
        mode_str = getattr(args, "mode", "deployment")
        max_adv_raw = getattr(args, "max_order_to_adv", None)
        if date_str is None:
            logger.error("[SYS] universe status=fail error=missing --date")
            return 1
        try:
            as_of = date.fromisoformat(str(date_str))
        except Exception as exc:
            logger.error(f"[SYS] universe status=fail error={exc!r}")
            return 1
        mode_val = str(mode_str).lower()
        if mode_val not in ("structural", "deployment"):
            logger.error(f"[SYS] universe status=fail error=invalid mode {mode_val!r}")
            return 1
        try:
            max_order_to_adv = float(max_adv_raw) if max_adv_raw is not None else 0.05
        except Exception:
            max_order_to_adv = 0.05
        from src.core.calendar import get_calendar
        from src.core.paths import DataPaths
        from src.core.settings import get_settings
        from src.universe.instruments import load_sponsor_brand_map
        from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
        from src.universe.taxonomy import Taxonomy

        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        cal = get_calendar()
        # Load panel if exists
        panel = None
        silver_path = paths.silver("etf_daily")
        if silver_path.exists():
            try:
                import polars as pl

                panel = pl.read_parquet(silver_path)
            except Exception:
                panel = None
        if panel is None or panel.height == 0:
            # No data: log empty universe but still succeed
            logger.info(f"[DATA] universe as_of={as_of} mode={mode_val} admitted=0 dropped={{}}")
            logger.info(f"[SYS] universe as_of={as_of} mode={mode_val} admitted=0")
            return 0
        # Ensure required columns exist
        # Build master
        try:
            brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
        except Exception:
            brand_map = {}
        try:
            taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
        except Exception:
            taxonomy = Taxonomy(rules=[])
        # Load universe config
        universe_config: dict[str, object] = {}
        try:
            import yaml

            with open(config_path("universe"), encoding="utf-8") as f:
                uc_raw = yaml.safe_load(f) or {}
            universe_config = uc_raw["universe"] if isinstance(uc_raw, dict) and "universe" in uc_raw else uc_raw
        except Exception:
            universe_config = {}
        # sponsor issuers tuple
        sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
        # Load manifest if present (handled inside for_mode)
        from src.universe.instruments import InstrumentMaster

        master = InstrumentMaster.build(panel, taxonomy, brand_map)
        umode = UniverseMode.STRUCTURAL if mode_val == "structural" else UniverseMode.DEPLOYMENT
        filt = UniverseFilters.for_mode(
            umode,
            universe_config,
            sponsor_issuers,
            max_order_to_adv=max_order_to_adv,
        )
        # Use adv_window from config if present
        adv_w = 20
        try:
            adv_w = int(universe_config.get("adv_window", 20))  # type: ignore[call-overload]
        except Exception:
            adv_w = 20
        universe = PointInTimeUniverse(panel, master, cal, adv_window=adv_w, brand_map=brand_map)
        snap = universe.get(as_of, filt)
        dropped_str = ", ".join(f"{k}={v}" for k, v in snap.dropped.items())
        logger.info(f"[DATA] universe as_of={as_of} mode={mode_val} admitted={len(snap.tickers)} dropped={dict(snap.dropped)} {dropped_str}")
        logger.info(f"[SYS] universe as_of={as_of} mode={mode_val} admitted={len(snap.tickers)}")
        return 0
    except Exception as exc:
        logger.error(f"[SYS] universe status=fail error={exc!r}")
        return 1
