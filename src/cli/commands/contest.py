"""Contest CLI commands (leaderboard archive, reference seed, weekly decision, daily shadow)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
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


def _our_equity_override(
    args: argparse.Namespace,
    contest: dict[str, Any],
    panel: Any,
    archive_root: Path,
    snapshot: Any,
    session: date,
    state_alias: str | None,
) -> tuple[float, str] | None:
    """Our equity when the leaderboard cannot supply it.

    A user-entered `--our-return` (percent, as shown in the contest app) always wins. Otherwise, when the current
    snapshot does not list us (outside the public top-50), compound the recorded holding from the last archived
    session where we were visible. The estimate assumes that holding was held throughout, so it is tagged.
    """
    from src.contest.decision import realized_equity
    from src.contest.leaderboard import latest_entry_on_or_before

    manual = getattr(args, "our_return", None)
    if manual is not None:
        return 1.0 + float(manual) / 100.0, "OUR_RETURN_MANUAL"
    nickname = str(contest.get("nickname", ""))
    if snapshot is None or snapshot.base_date != session or snapshot.entry_for(nickname) is not None:
        return None
    if state_alias is None:
        return None
    seen = latest_entry_on_or_before(archive_root, nickname, session)
    if seen is None:
        return None
    seen_date, entry = seen
    try:
        equity = realized_equity(
            panel, state_alias, seen_date, session, 1.0 + entry.total_return_pct / 100.0,
            float(contest["decision"]["target_weight"]),
        )
    except (KeyError, ValueError):
        return None
    logger.info(
        f"[PORTFOLIO] contest_weekly our_equity=estimated from={seen_date.isoformat()} alias={state_alias} "
        f"equity={equity:.6f}"
    )
    return equity, "OUR_RETURN_ESTIMATED"


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
        override = _our_equity_override(args, contest, panel, archive_root, snapshot, session, state_alias)
        decision = decide_week(panel, snapshot, session, calendar, contest_run, state_alias, override)
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


def _read_daily_metrics(output_dir: Path) -> list[dict[str, Any]]:
    import json

    path = output_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _upsert_daily_metric(output_dir: Path, row: Mapping[str, Any]) -> None:
    import json

    path = output_dir / "metrics.jsonl"
    kept = [r for r in _read_daily_metrics(output_dir) if r.get("session") != row.get("session")]
    with path.open("w", encoding="utf-8") as handle:
        for item in [*kept, dict(row)]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def cmd_contest_daily(args: argparse.Namespace) -> int:
    """Fetch and archive today's contest ETF quotes, build the shadow card, and persist the JSON card, the Korean
    markdown card and the metrics row.

    Target session: `--session` if given, else the last trading session on or before today (KST). Quotes are
    fetched only for the target session and only when not yet archived. `--our-return` overrides our equity, as in
    contest-weekly.

    Idempotent: when `<output_dir>/<session>.json` exists and `--force` is absent, log and return 0. The metrics row
    for a session is upserted: `metrics.jsonl` keeps one row per session.

    Returns 0 on any written card (including NO_DATA), 1 on unexpected exceptions. `shadow.enabled: false` returns
    0 with a log line.
    """
    from src.contest.daily import (
        archive_quotes,
        build_daily_card,
        card_metric_row,
        daily_to_dict,
        fetch_quotes,
        load_quotes,
        render_daily_markdown,
    )
    from src.contest.leaderboard import latest_snapshot_on_or_before
    from src.core.calendar import get_calendar, kst_today
    from src.core.paths import DataPaths
    from src.core.settings import get_settings

    try:
        contest = _contest_config()
        shadow = contest.get("shadow", {})
        if not isinstance(shadow, dict) or not shadow.get("enabled", False):
            logger.info("[SYS] contest_daily status=disabled")
            return 0
        calendar = get_calendar()
        raw_session = getattr(args, "session", None)
        session: date | None = (
            date.fromisoformat(str(raw_session)) if raw_session else _resolve_target_session(calendar, kst_today())
        )
        if session is None:
            raise ValueError("no trading session on or before today")
        output_dir = Path(str(shadow["output_dir"]))
        card_path = output_dir / f"{session.isoformat()}.json"
        # NO_DATA 카드는 최종이 아니다: 같은 날 이후 타이머 실행에서 순위표/시세를 다시 시도한다
        if card_path.is_file() and not getattr(args, "force", False) and '"action": "NO_DATA"' not in card_path.read_text(
            encoding="utf-8"
        ):
            logger.info(f"[PORTFOLIO] contest_daily session={session.isoformat()} status=exists_skip")
            return 0
        data_root = Path(get_settings().data_root)
        archive_root = data_root / str(contest["leaderboard"]["archive_dir"])
        quotes_root = data_root / str(shadow["quotes_dir"])
        exposures = {str(k): [str(t) for t in v] for k, v in dict(shadow["exposures"]).items()}
        if load_quotes(quotes_root, session) is None:
            tickers = sorted({t for tickers in exposures.values() for t in tickers})
            try:
                bars = fetch_quotes(
                    tickers, session, str(shadow.get("yahoo_suffix", ".KS")),
                    float(shadow.get("timeout_s", 20)),
                )
            except Exception as exc:
                logger.info(
                    f"[DATA] contest_quotes session={session.isoformat()} status=fetch_fail error={exc!r}"
                )
                bars = {}
            # 기록은 한 번만 쓰므로, 미러 양쪽 노출이 모두 잡힌 경우에만 보관해 이후 재실행에서 재시도할 수 있게 한다
            sides = (str(shadow["long_alias"]), str(shadow["inverse_alias"]))
            if all(any(t in bars for t in exposures.get(side, [])) for side in sides):
                archive_quotes(quotes_root, session, bars)
            logger.info(f"[DATA] contest_quotes session={session.isoformat()} tickers={len(bars)}")
        snapshot = latest_snapshot_on_or_before(archive_root, session)
        try:
            panel = _build_weekly_panel(contest, data_root)
        except Exception:
            panel = None
        # 상태 파일은 읽기만 한다(그림자 모드는 쓰지 않음): REVIEW 판정과 50위 밖 평가액 추정에 필요
        state_name = str(contest.get("decision", {}).get("state_name", "contest_position"))
        state_alias = _read_state_alias(DataPaths(root=data_root).state(state_name))
        override = _our_equity_override(args, contest, panel, archive_root, snapshot, session, state_alias)
        prior_metrics = _read_daily_metrics(output_dir)
        card = build_daily_card(
            session, calendar, archive_root, quotes_root, contest, state_alias, override, prior_metrics
        )
    except Exception as exc:
        logger.error(f"[SYS] contest_daily status=fail error={exc!r}")
        return 1

    import json

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(daily_to_dict(card), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{session.isoformat()}.md").write_text(render_daily_markdown(card), encoding="utf-8")
        _upsert_daily_metric(output_dir, card_metric_row(card))
    except Exception as exc:
        logger.error(f"[SYS] contest_daily status=fail reason=persist error={exc!r}")
        return 1
    mirror = card.mirror_target or "none"
    share = f"{card.identified_top_share:.3f}" if card.identified_top_share is not None else "none"
    churn = f"{card.daily_churn:.3f}" if card.daily_churn is not None else "none"
    logger.info(
        f"[PORTFOLIO] contest_daily session={session.isoformat()} action={card.action.value} "
        f"mirror={mirror} gate={card.gate.value} share={share} churn={churn}"
    )
    return 0
