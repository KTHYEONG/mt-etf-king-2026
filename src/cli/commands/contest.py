"""Contest CLI commands (leaderboard archive, reference seed)."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _contest_config() -> dict[str, Any]:
    from src.core.config import load_config

    data = load_config("contest")
    contest = data.get("contest")
    if not isinstance(contest, dict):
        raise ValueError("contest config missing 'contest' mapping")
    return contest


def cmd_contest_archive(args: argparse.Namespace) -> int:
    """Fetch and archive today's leaderboard snapshot; print our rank/return when visible.

    Returns 0 on success or idempotent no-op, 1 on fetch/schema/conflict failure (fail-closed, nothing written).
    """
    from src.contest.leaderboard import (
        LeaderboardConflictError,
        LeaderboardFetchError,
        LeaderboardSchemaError,
        archive_leaderboard,
        fetch_leaderboard_payloads,
    )
    from src.core.settings import get_settings

    try:
        contest = _contest_config()
        enabled = contest.get("enabled", False)
        if not enabled:
            logger.info("[SYS] contest disabled")
            return 0
        nickname = str(contest.get("nickname", ""))
        leaderboard = contest.get("leaderboard", {})
        if not isinstance(leaderboard, dict):
            raise ValueError("contest.leaderboard mapping missing")
        base_url = str(leaderboard["base_url"])
        endpoints = list(leaderboard["endpoints"])
        timeout_s = float(leaderboard["timeout_s"])
        archive_dir = str(leaderboard["archive_dir"])
        settings = get_settings()
        archive_root = Path(settings.data_root) / archive_dir
    except Exception as exc:
        logger.error(f"[SYS] contest_archive status=fail error={exc!r}")
        return 1

    try:
        payloads = fetch_leaderboard_payloads(base_url, endpoints, timeout_s)
    except LeaderboardFetchError as exc:
        logger.error(f"[SYS] contest_archive status=fail reason=fetch error={exc!r}")
        return 1
    except Exception as exc:
        logger.error(f"[SYS] contest_archive status=fail reason=fetch error={exc!r}")
        return 1

    try:
        base_date, written = archive_leaderboard(payloads, archive_root)
    except (LeaderboardSchemaError, LeaderboardConflictError) as exc:
        logger.error(f"[SYS] contest_archive status=fail reason={type(exc).__name__} error={exc!r}")
        return 1
    except Exception as exc:
        logger.error(f"[SYS] contest_archive status=fail error={exc!r}")
        return 1

    try:
        from src.contest.leaderboard import load_snapshot

        snapshot = load_snapshot(archive_root, base_date)
        entry = snapshot.entry_for(nickname) if nickname else None
        our_rank = str(entry.rank) if entry is not None else "none"
        our_total = f"{entry.total_return_pct}" if entry is not None else "none"
        logger.info(
            f"[DATA] contest_archive base_date={base_date.isoformat()} written={written} "
            f"our_rank={our_rank} our_total={our_total}"
        )
    except Exception as exc:
        logger.error(f"[SYS] contest_archive status=fail reason=load error={exc!r}")
        return 1
    return 0


def cmd_contest_seed_reference(args: argparse.Namespace) -> int:
    """Download the single-stock reference OHLC used for synthetic pre-listing legs.

    Returns 0 on success, 1 on ReferenceFetchError (fail-closed, nothing written).
    """
    from src.contest.reference import ReferenceFetchError, seed_single_stock_reference
    from src.core.settings import get_settings

    try:
        contest = _contest_config()
        reference = contest.get("reference", {})
        if not isinstance(reference, dict):
            raise ValueError("contest.reference mapping missing")
        symbols = {str(k): str(v) for k, v in dict(reference["symbols"]).items()}
        cutoff = date.fromisoformat(str(reference["cutoff"]))
        timeout_s = float(reference.get("timeout_s", 30))
        out_path = Path(get_settings().data_root) / str(reference["single_stock_path"])
    except Exception as exc:
        logger.error(f"[SYS] contest_seed_reference status=fail error={exc!r}")
        return 1
    try:
        n_rows = seed_single_stock_reference(symbols, cutoff, out_path, timeout_s)
    except ReferenceFetchError as exc:
        logger.error(f"[SYS] contest_seed_reference status=fail reason=fetch error={exc!r}")
        return 1
    except Exception as exc:
        logger.error(f"[SYS] contest_seed_reference status=fail error={exc!r}")
        return 1
    logger.info(f"[DATA] contest_seed_reference status=ok rows={n_rows} path={out_path}")
    return 0


def _resolve_target_session(calendar: Any, today: date) -> date | None:
    from datetime import timedelta

    sessions = calendar.sessions(today - timedelta(days=21), today)
    past = [s for s in sessions if s <= today]
    return past[-1] if past else None


def _read_state_alias(state_path: Path) -> str | None:
    import json

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    alias = data.get("recommended_alias") if isinstance(data, dict) else None
    return alias if isinstance(alias, str) else None


def _build_weekly_panel(contest: dict[str, Any], data_root: Path) -> Any:
    import polars as pl

    from src.contest.panel import build_vehicle_panel
    from src.core.paths import DataPaths

    paths = DataPaths(root=data_root)
    etf_daily = pl.read_parquet(paths.silver("etf_daily"))
    index_daily = pl.read_parquet(paths.silver("index_daily"))
    ref_path = data_root / str(contest["reference"]["single_stock_path"])
    reference = pl.read_parquet(ref_path)
    return build_vehicle_panel(
        etf_daily, index_daily, reference, contest["vehicles"], float(contest["simulation"]["synthetic_fee_annual"])
    )


def cmd_contest_weekly(args: argparse.Namespace) -> int:
    """Build the panel, load the archived leaderboard, run decide_week for the target session, persist the JSON
    card, the Korean markdown card and the position state.

    Target session: `--session` if given, else the last trading session on or before today (KST). A Saturday run
    therefore decides on Friday's close.

    Idempotent: when `<output_dir>/<session>.json` exists and `--force` is absent, log and return 0.
    Returns 0 on any written card (including NO_DATA/NEEDS_CONFIRMATION), 1 on unexpected exceptions.
    """
    from src.contest.decision import ContestAction, decide_week, decision_to_dict, render_decision_markdown
    from src.contest.leaderboard import latest_snapshot_on_or_before
    from src.core.calendar import get_calendar, kst_today
    from src.core.paths import DataPaths
    from src.core.settings import get_settings

    try:
        contest = _contest_config()
        decision_cfg = contest["decision"]
        output_dir = Path(str(decision_cfg["output_dir"]))
        calendar = get_calendar()
        raw_session = getattr(args, "session", None)
        session: date | None = (
            date.fromisoformat(str(raw_session)) if raw_session else _resolve_target_session(calendar, kst_today())
        )
        if session is None:
            raise ValueError("no trading session on or before today")
        card_path = output_dir / f"{session.isoformat()}.json"
        if card_path.is_file() and not getattr(args, "force", False):
            logger.info(f"[PORTFOLIO] contest_weekly session={session.isoformat()} status=exists_skip")
            return 0
        data_root = Path(get_settings().data_root)
        panel = _build_weekly_panel(contest, data_root)
        archive_root = data_root / str(contest["leaderboard"]["archive_dir"])
        snapshot = latest_snapshot_on_or_before(archive_root, session)
        state_path = DataPaths(root=data_root).state(str(decision_cfg["state_name"]))
        state_alias = _read_state_alias(state_path)
        n_worlds = int(getattr(args, "worlds", None) or decision_cfg["n_worlds"])
        contest_run = {**contest, "decision": {**decision_cfg, "n_worlds": n_worlds}}
        decision = decide_week(panel, snapshot, session, calendar, contest_run, state_alias)
    except Exception as exc:
        logger.error(f"[SYS] contest_weekly status=fail error={exc!r}")
        return 1

    import json

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        card = decision_to_dict(decision)
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        names = {a: str(v.get("name", a)) for a, v in contest["vehicles"].items()}
        (output_dir / f"{session.isoformat()}.md").write_text(
            render_decision_markdown(decision, names), encoding="utf-8"
        )
        if decision.action in (ContestAction.HOLD, ContestAction.SWITCH):
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "as_of": session.isoformat(),
                        "recommended_alias": decision.target_alias,
                        "recommended_ticker": decision.target_ticker,
                        "execution_session": decision.execution_session.isoformat()
                        if decision.execution_session
                        else None,
                        "source": "recommendation",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        logger.error(f"[SYS] contest_weekly status=fail reason=persist error={exc!r}")
        return 1
    scored = {s.alias: s for s in decision.scores}
    p1_best = scored[decision.target_alias].p1_mean if decision.target_alias in scored else "none"
    if "HOLD" in scored:
        p1_current: Any = scored["HOLD"].p1_mean
    elif decision.current_alias in scored:
        p1_current = scored[decision.current_alias].p1_mean
    else:
        p1_current = "none"
    logger.info(
        f"[PORTFOLIO] contest_weekly session={session.isoformat()} action={decision.action.value} "
        f"from={decision.current_alias or 'none'} to={decision.target_alias or 'none'} "
        f"p1_best={p1_best} p1_current={p1_current}"
    )
    return 0
