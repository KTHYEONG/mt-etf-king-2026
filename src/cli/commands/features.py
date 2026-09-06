# ruff: noqa
from __future__ import annotations

import argparse
import logging
from datetime import date

from src.core.calendar import get_calendar
from src.core.config import config_path
from src.core.paths import DataPaths
from src.core.settings import get_settings

logger = logging.getLogger(__name__)


def cmd_features(args: argparse.Namespace) -> int:
    import time

    try:
        start_s = getattr(args, "start", None)
        end_s = getattr(args, "end", None)
        if start_s is None or end_s is None:
            logger.error("[SYS] features status=fail error=missing --start/--end")
            return 1
        try:
            start = date.fromisoformat(str(start_s))
            end = date.fromisoformat(str(end_s))
        except Exception as exc:
            logger.error(f"[SYS] features status=fail error={exc!r}")
            return 1
        if start > end:
            logger.error(f"[SYS] features status=fail error=start {start} > end {end}")
            return 1
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        cal = get_calendar()
        from src.features.builder import FeatureBuilder, FeatureConfig

        features_config_path = config_path("features")
        try:
            config = FeatureConfig.from_yaml(features_config_path)
        except Exception as exc:
            logger.error(f"[SYS] features status=fail error=load config {exc!r}")
            return 1
        builder = FeatureBuilder(cal, config)
        # Load silver panel
        silver_path = paths.silver("etf_daily")
        if not silver_path.exists():
            logger.error(f"[SYS] features status=fail error=silver not found {silver_path}")
            logger.error("[DATA] features status=fail error=silver not found")
            return 1
        import polars as pl

        try:
            panel = pl.read_parquet(silver_path)
        except Exception as exc:
            logger.error(f"[SYS] features status=fail error=read silver {exc!r}")
            return 1
        if "date" in panel.columns:
            panel = panel.filter(pl.col("date") <= end)
        t0 = time.time()
        # decision_date=end; input must not contain future sessions (PIT)
        try:
            feature_panel = builder.build_panel(panel, decision_date=end)
        except Exception as exc:
            logger.error(f"[SYS] features status=fail error=build_panel {exc!r}")
            return 1
        # Filter to requested range [start, end] for persistence
        if "date" in feature_panel.columns:
            feature_panel = feature_panel.filter((pl.col("date") >= start) & (pl.col("date") <= end))
            feature_panel = feature_panel.sort(["date", "ticker"])
        # Persist to gold
        gold_path = paths.gold("etf_features")
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            feature_panel.write_parquet(str(gold_path), compression="zstd", use_pyarrow=True)
        except TypeError:
            feature_panel.write_parquet(str(gold_path), compression="zstd")
        elapsed = time.time() - t0
        # Coverage: count rows and distinct tickers/dates
        rows = feature_panel.height
        # count distinct dates and tickers
        try:
            n_dates = int(feature_panel.select(pl.col("date").n_unique()).item()) if rows > 0 and "date" in feature_panel.columns else 0
        except Exception:
            n_dates = 0
        try:
            n_tickers = int(feature_panel.select(pl.col("ticker").n_unique()).item()) if rows > 0 and "ticker" in feature_panel.columns else 0
        except Exception:
            n_tickers = 0
        logger.info(f"[SYS] features start={start} end={end} elapsed={elapsed:.3f}s rows={rows}")
        logger.info(f"[DATA] features start={start} end={end} rows={rows} dates={n_dates} tickers={n_tickers} path={gold_path}")
        return 0
    except Exception as exc:
        logger.error(f"[SYS] features status=fail error={exc!r}")
        return 1
