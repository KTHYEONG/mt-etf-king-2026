"""Daily pipeline orchestration (ingest + normalize + features, optional decide)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Collection, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from src.core.calendar import TradingCalendar

logger = logging.getLogger(__name__)

_INDEX_INGEST_DATASET: Final[str] = "kospi_index"
_INDEX_NORMALIZE_DATASET: Final[str] = "index_daily"
_REGULAR_CLOSE: Final[time] = time(15, 30)


def _contest_mode_active() -> bool:
    """True when the contest weekly decision owns the decide step."""
    try:
        from src.core.config import load_config

        return bool(load_config("contest").get("contest", {}).get("enabled", False))
    except Exception:
        return False


def resolve_pipeline_target_session(
    as_of: date | None = None,
    *,
    now: datetime | None = None,
    calendar: TradingCalendar | None = None,
) -> date | None:
    """Resolve the pipeline target session with KST time-awareness."""
    kst = ZoneInfo("Asia/Seoul")
    now_kst = datetime.now(kst) if now is None else now.astimezone(kst)
    today = now_kst.date()
    if calendar is None:
        from src.core.calendar import get_calendar

        calendar = get_calendar()
    requested = as_of if as_of is not None else today
    if requested > today:
        return None
    if requested < today:
        sessions = calendar.sessions(requested - timedelta(days=21), requested)
        if not sessions:
            return None
        return sessions[-1]
    if now_kst.time() >= _REGULAR_CLOSE and calendar.is_session(today):
        return today
    sessions = calendar.sessions(today - timedelta(days=21), today)
    return sessions[-1 - int(sessions[-1] == today)]


def _sessions_missing_from(
    present: Collection[date], end: date, calendar: TradingCalendar
) -> list[date]:
    """Calendar sessions in `[min(present), end]` absent from `present`, ascending."""
    if not present:
        return []
    have = set(present)
    earliest = min(present)
    return [session for session in calendar.sessions(earliest, end) if session not in have]


def resolve_ingest_start(
    end: date,
    lookback_days: int,
    *,
    etf_dates: Collection[date],
    index_dates: Collection[date],
    calendar: TradingCalendar,
) -> date:
    """Ingest window start: the lookback start, extended back to the earliest calendar session missing from either
    silver table.

    A fixed lookback silently abandons any session that falls out of the window before a successful run (the
    2026-08-28 ETF gap). The bronze planner only fetches sessions absent from bronze, so widening the window costs
    one API call per genuinely missing session and nothing for present ones.

    Args:
        end: Target session (window end, inclusive).
        lookback_days: Minimum re-fetch window in calendar days.
        etf_dates: Dates in ETF silver (empty when the table does not exist).
        index_dates: Dates in index silver (empty when the table does not exist).
        calendar: Effective trading calendar.

    Returns:
        min(end - lookback_days, earliest missing session); each table is scanned from its own earliest date to
        `end`, and an empty table contributes nothing.
    """
    start = end - timedelta(days=lookback_days)
    missing = _sessions_missing_from(etf_dates, end, calendar)
    missing.extend(_sessions_missing_from(index_dates, end, calendar))
    if missing:
        start = min(start, min(missing))
    return start


def features_are_stale(gold_path: Path, input_paths: Sequence[Path]) -> bool:
    """True when the gold feature file is missing or older (st_mtime_ns) than any existing input.

    Inputs are the ETF silver parquet, `configs/features.yaml`, and every `*.py` under `src/features/`; a deploy that
    changes feature code or config therefore triggers one rebuild. Changes outside these inputs require
    `--force-rebuild`.
    """
    try:
        gold_mtime = gold_path.stat().st_mtime_ns
    except OSError:
        return True
    for candidate in input_paths:
        try:
            if candidate.is_file() and candidate.stat().st_mtime_ns > gold_mtime:
                return True
        except OSError:
            continue
    return False


def cmd_daily_refresh(args: argparse.Namespace) -> int:
    """Run the daily batch: gap-healing ingest, incremental normalize, gap gate, features when stale, optional decide.

    On a KRX holiday or when nothing new was published the run targets the previous session, ingests nothing,
    leaves silver untouched, skips the feature rebuild, and returns 0.

    Returns:
        0 on success (including no-op runs); 1 on an invalid `--as-of`, no resolvable session, a failed stage, an
        empty ETF silver table, a missing index silver table, a stale ETF-vs-index session gap, or a failed decide.
    """
    from src.cli.commands.data import cmd_ingest, cmd_normalize
    from src.cli.commands.decide import cmd_decide
    from src.cli.commands.features import cmd_features
    from src.cli.constants import CHAMPION_STRATEGY
    from src.core.calendar import get_calendar
    from src.core.config import project_root
    from src.core.paths import DataPaths
    from src.core.settings import get_settings
    from src.data.silver import silver_session_dates
    from src.data.validation import etf_index_session_gaps

    dataset = getattr(args, "dataset", None) or "etf_daily"
    as_of_raw = getattr(args, "as_of", None)
    lookback_days = getattr(args, "lookback_days", 10) or 10
    force_rebuild = bool(getattr(args, "force_rebuild", False))

    try:
        as_of = None if as_of_raw is None else date.fromisoformat(as_of_raw)
    except ValueError:
        return 1

    target = resolve_pipeline_target_session(as_of=as_of)
    if target is None:
        return 1
    calendar = get_calendar()
    sessions = calendar.sessions(target - timedelta(days=21), target)
    if not sessions:
        return 1
    end = sessions[-1]

    paths = DataPaths(root=get_settings().data_root)
    etf_dates = silver_session_dates(paths, dataset)
    index_dates = silver_session_dates(paths, _INDEX_NORMALIZE_DATASET)
    ingest_start = resolve_ingest_start(
        end, lookback_days, etf_dates=etf_dates, index_dates=index_dates, calendar=calendar
    )
    lookback_start = end - timedelta(days=lookback_days)
    if ingest_start < lookback_start:
        healed = set(_sessions_missing_from(etf_dates, end, calendar))
        healed.update(_sessions_missing_from(index_dates, end, calendar))
        logger.info(
            f"[DATA] daily_refresh heal_start={ingest_start.isoformat()} "
            f"lookback_start={lookback_start.isoformat()} missing_sessions={len(healed)}"
        )
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

    etf_dates = silver_session_dates(paths, dataset)
    if not etf_dates:
        return 1
    if not paths.silver(_INDEX_NORMALIZE_DATASET).is_file():
        logger.error("[DATA] gate=etf_index_session_gap status=fail reason=index_silver_missing")
        return 1
    index_dates = silver_session_dates(paths, _INDEX_NORMALIZE_DATASET)
    gaps = etf_index_session_gaps(etf_dates, index_dates)
    if gaps.stale:
        first_five = ",".join(day.isoformat() for day in gaps.stale[:5])
        logger.error(
            f"[DATA] gate=etf_index_session_gap status=fail count={len(gaps.stale)} stale={first_five}"
        )
        return 1
    if gaps.pending:
        logger.warning(
            f"[DATA] gate=etf_index_session_gap status=pending session={gaps.pending[0].isoformat()}"
        )

    gold_path = paths.gold("etf_features")
    project = project_root()
    feature_inputs = [paths.silver(dataset), project / "configs" / "features.yaml"]
    feature_inputs.extend(sorted((project / "src" / "features").rglob("*.py")))
    if not force_rebuild and not features_are_stale(gold_path, feature_inputs):
        logger.info("[DATA] features status=unchanged")
    else:
        rc = cmd_features(
            argparse.Namespace(start=min(etf_dates).isoformat(), end=end.isoformat())
        )
        if rc != 0:
            return 1

    if not getattr(args, "decide", False):
        return 0

    if _contest_mode_active():
        logger.info("[PORTFOLIO] daily champion decide skipped: contest_mode weekly decision active")
        return 0

    output_dir = getattr(args, "output_dir", None) or "results/decide_daily"

    from src.cli.commands.decide.models import _LIVE_STICKY_STRATEGIES, _sticky_live_state_name

    if CHAMPION_STRATEGY in _LIVE_STICKY_STRATEGIES:
        from src.data.panel import BACKTEST_PANEL_COLUMNS, load_backtest_panel
        from src.tournament.live_decision import StateDiscontinuityError, pending_catchup_sessions

        catchup_paths = DataPaths(root=get_settings().data_root)
        state_path = catchup_paths.state(_sticky_live_state_name(CHAMPION_STRATEGY))
        catchup_panel: Any = load_backtest_panel(catchup_paths, columns=BACKTEST_PANEL_COLUMNS)
        try:
            pending = pending_catchup_sessions(state_path, catchup_panel, target_session=end)
        except StateDiscontinuityError as exc:
            logger.error(f"[SYS] catchup status=fail reason=state_discontinuity {exc}")
            return 1
        for k, session in enumerate(pending, start=1):
            rc = cmd_decide(
                argparse.Namespace(
                    model=CHAMPION_STRATEGY,
                    date=session.isoformat(),
                    panel=None,
                    capital=None,
                    output=f"{output_dir}/{session.isoformat()}.json",
                    trace=False,
                )
            )
            logger.info(f"[SYS] catchup model={CHAMPION_STRATEGY} session={session} k={k}/{len(pending)} rc={rc}")
            if rc != 0:
                logger.error(f"[SYS] catchup status=fail session={session}")
                return 1

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
