# ruff: noqa
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.calendar import get_calendar
from src.core.config import config_path
from src.core.paths import DataPaths
from src.core.settings import get_settings

logger = logging.getLogger(__name__)


from src.cli.commands.decide.scoring import _load_panel_for_backtest


def cmd_forensics(args: argparse.Namespace) -> int:
    try:
        run_id = getattr(args, "run_id", None)
        if run_id is None:
            run_id = getattr(args, "runId", None)
        try:
            top_q = float(getattr(args, "top_q", 0.95))
        except Exception:
            top_q = 0.95
        try:
            lo = float(getattr(args, "near_miss_lo", 0.20))
        except Exception:
            lo = 0.20
        try:
            hi = float(getattr(args, "near_miss_hi", 0.50))
        except Exception:
            hi = 0.50
        if not run_id:
            logger.error("[SYS] forensics status=fail error=missing --run-id")
            return 1
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        try:
            run_dir = paths.results(str(run_id))
        except Exception:
            run_dir = Path("docs/results") / str(run_id)
        windows_path = Path(run_dir) / "windows.parquet"
        trades_path = Path(run_dir) / "trades.parquet"
        if not windows_path.exists() or not trades_path.exists():
            logger.error(f"[SYS] forensics status=fail error=missing windows/trades for run {run_id}")
            return 1
        import polars as pl

        try:
            windows = pl.read_parquet(str(windows_path))
        except Exception as exc:
            logger.error(f"[SYS] forensics status=fail error=read windows {exc!r}")
            return 1
        try:
            trades = pl.read_parquet(str(trades_path))
        except Exception as exc:
            logger.error(f"[SYS] forensics status=fail error=read trades {exc!r}")
            return 1
        try:
            panel = _load_panel_for_backtest(paths, get_calendar())
        except Exception as exc:
            logger.error(f"[SYS] forensics status=fail error=load panel {exc!r}")
            return 1
        if panel is None or (hasattr(panel, "height") and panel.height == 0):
            logger.error("[SYS] forensics status=fail error=empty panel")
            return 1
        # Build master
        try:
            from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
            from src.universe.taxonomy import Taxonomy

            try:
                brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
            except Exception:
                brand_map = {}
            try:
                taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
            except Exception:
                taxonomy = Taxonomy(rules=[])
            master = InstrumentMaster.build(panel, taxonomy, brand_map)
        except Exception as exc:
            logger.error(f"[SYS] forensics status=fail error=build master {exc!r}")
            return 1
        try:
            if "date" in panel.columns:
                sessions = sorted(panel.select(pl.col("date")).unique().to_series().to_list())
            else:
                sessions = []
        except Exception:
            sessions = []
        from src.reporting.tail_forensics import summarise_tail_attribution, write_tail_attribution_report
        from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode

        try:
            import yaml as _yaml_forensics

            with open(config_path("universe"), encoding="utf-8") as _f_forensics:
                _uc_raw = _yaml_forensics.safe_load(_f_forensics) or {}
            _universe_config = _uc_raw.get("universe", _uc_raw) if isinstance(_uc_raw, dict) else {}
        except Exception:
            _universe_config = {}
        try:
            _sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
        except Exception:
            _sponsor_issuers = ()
        try:
            _forensics_filters = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, _universe_config, _sponsor_issuers)
        except Exception as exc:
            logger.error(f"[SYS] forensics status=fail error=build universe filters {exc!r}")
            return 1
        try:
            _forensics_universe = PointInTimeUniverse(panel, master, get_calendar(), brand_map=brand_map)
        except Exception as exc:
            logger.error(f"[SYS] forensics status=fail error=build pit universe {exc!r}")
            return 1
        summary = summarise_tail_attribution(
            windows=windows,
            trades=trades,
            panel=panel,
            master=master,
            sessions=sessions,
            top_q=top_q,
            near_miss_lo=lo,
            near_miss_hi=hi,
            universe=_forensics_universe,
            filters=_forensics_filters,
        )
        out = write_tail_attribution_report(Path(run_dir), summary)
        logger.info(
            f"[SYS] forensics run={run_id} primary_gap={summary.primary_gap} "
            f"mean_selection={summary.mean_selection_loss:.4f} mean_entry={summary.mean_entry_timing_loss:.4f} "
            f"mean_exit={summary.mean_exit_timing_loss:.4f} mean_giveback={summary.mean_giveback_loss:.4f} "
            f"n_analyzed={summary.n_analyzed} path={out}"
        )
        return 0
    except Exception as exc:
        logger.error(f"[SYS] forensics status=fail error={exc!r}")
        return 1
