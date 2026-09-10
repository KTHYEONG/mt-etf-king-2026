"""Daily pipeline orchestration (ingest + normalize + features, optional decide)."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Final

_INDEX_INGEST_DATASET: Final[str] = "kospi_index"
_INDEX_NORMALIZE_DATASET: Final[str] = "index_daily"


def cmd_daily_refresh(args: argparse.Namespace) -> int:
    """Run the daily batch: ingest, normalize, full-range features, optional decide."""
    from src.cli.commands.data import cmd_ingest, cmd_normalize
    from src.cli.commands.decide import cmd_decide
    from src.cli.commands.features import cmd_features
    from src.cli.constants import CHAMPION_STRATEGY
    from src.core.calendar import get_calendar
    from src.core.paths import DataPaths
    from src.core.settings import get_settings

    dataset = getattr(args, "dataset", None) or "etf_daily"
    as_of_raw = getattr(args, "as_of", None)
    lookback_days = getattr(args, "lookback_days", 10) or 10

    if as_of_raw is None:
        as_of = date.today()
    else:
        try:
            as_of = date.fromisoformat(as_of_raw)
        except ValueError:
            return 1

    sessions = get_calendar().sessions(as_of - timedelta(days=21), as_of)
    if not sessions:
        return 1
    end = sessions[-1]

    ingest_start = end - timedelta(days=lookback_days)
    rc = cmd_ingest(
        argparse.Namespace(
            dataset=dataset,
            start=ingest_start.isoformat(),
            end=end.isoformat(),
            dry_run=False,
        )
    )
    if rc != 0:
        return 1

    rc = cmd_normalize(argparse.Namespace(dataset=dataset, mode="incremental"))
    if rc != 0:
        return 1

    rc = cmd_ingest(argparse.Namespace(dataset=_INDEX_INGEST_DATASET, start=ingest_start.isoformat(), end=end.isoformat(), dry_run=False))
    if rc != 0:
        return 1
    rc = cmd_normalize(argparse.Namespace(dataset=_INDEX_NORMALIZE_DATASET, mode="incremental"))
    if rc != 0:
        return 1

    import polars as pl

    silver_path = DataPaths(root=get_settings().data_root).silver(dataset)
    panel = pl.read_parquet(silver_path) if silver_path.exists() else None
    if panel is None or panel.height == 0:
        return 1
    features_start_raw = panel["date"].min()
    if not isinstance(features_start_raw, date):
        return 1
    features_start = features_start_raw

    rc = cmd_features(
        argparse.Namespace(start=features_start.isoformat(), end=end.isoformat())
    )
    if rc != 0:
        return 1

    if not getattr(args, "decide", False):
        return 0

    output_dir = getattr(args, "output_dir", None) or "results/decide_daily"
    output_path = f"{output_dir}/{end.isoformat()}.json"
    rc = cmd_decide(
        argparse.Namespace(
            model=CHAMPION_STRATEGY,
            date=end.isoformat(),
            panel=None,
            capital=None,
            output=output_path,
            trace=False,
        )
    )
    if rc != 0:
        return 1
    return 0
