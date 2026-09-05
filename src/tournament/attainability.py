# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WindowOpportunity:
    window_start: date
    breadth: int
    ceiling: float | None


def _finite_positive(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return float(f)


def window_opportunities(
    open_by_session: Sequence[Mapping[str, float]],
    sessions: Sequence[date],
    candidates_by_session: Mapping[date, Sequence[str]],
    horizon: int,
) -> tuple[WindowOpportunity, ...]:
    try:
        h = int(horizon)
    except Exception:
        return ()
    if h <= 0:
        return ()
    n = len(sessions)
    out: list[WindowOpportunity] = []
    for i in range(n):
        entry = i + 1
        exit_ = i + 1 + h
        if entry < 0 or exit_ >= len(open_by_session) or exit_ >= n:
            continue
        try:
            cands = list(candidates_by_session.get(sessions[i], ()))  # type: ignore[union-attr]
        except Exception:
            cands = []
        entry_map = open_by_session[entry]
        exit_map = open_by_session[exit_]
        best: float | None = None
        breadth = 0
        for t in cands:
            try:
                pe = _finite_positive(entry_map.get(str(t))) if isinstance(entry_map, Mapping) else None
                px = _finite_positive(exit_map.get(str(t))) if isinstance(exit_map, Mapping) else None
            except Exception:
                continue
            if pe is None or px is None:
                continue
            breadth += 1
            r = float(px) / float(pe) - 1.0
            if best is None or r > best:
                best = float(r)
        out.append(WindowOpportunity(window_start=sessions[entry], breadth=int(breadth), ceiling=best))
    return tuple(out)


def attainability_curve(
    opportunities: Sequence[WindowOpportunity], thresholds: Sequence[float]
) -> dict[float, float]:
    opps = list(opportunities)
    n = len(opps)
    curve: dict[float, float] = {}
    for t in thresholds:
        try:
            tf = float(t)
        except Exception:
            continue
        if n == 0:
            curve[float(t)] = 0.0
            continue
        num = sum(1 for o in opps if o.ceiling is not None and float(o.ceiling) > tf)
        curve[float(t)] = float(num) / float(n)
    return curve


def capture_rate(
    window_returns: Sequence[float],
    opportunities: Sequence[WindowOpportunity],
    threshold: float,
    *,
    min_attainable: int = 1,
) -> tuple[float | None, int]:
    try:
        tf = float(threshold)
    except Exception:
        return (None, 0)
    try:
        need = max(1, int(min_attainable))
    except Exception:
        need = 1
    idx = [i for i, o in enumerate(opportunities) if o.ceiling is not None and float(o.ceiling) > tf]
    n = len(idx)
    if n < need:
        return (None, int(n))
    hits = 0
    for i in idx:
        try:
            hits += 1 if float(window_returns[i]) > tf else 0
        except Exception:
            continue
    return (float(hits) / float(n) if n else None, int(n))


_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60)


def load_attainability_config(
    gates_path: str = "configs/gates.yaml",
) -> tuple[tuple[float, ...], int, int]:
    try:
        from pathlib import Path

        import yaml

        with open(Path(gates_path), encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        att = raw.get("attainability") if isinstance(raw, dict) else None
        if not isinstance(att, dict):
            return (_DEFAULT_THRESHOLDS, 30, 5)
        thr_raw = att.get("thresholds", _DEFAULT_THRESHOLDS)
        thresholds = tuple(float(x) for x in thr_raw) if isinstance(thr_raw, (list, tuple)) else _DEFAULT_THRESHOLDS
        try:
            min_att = max(1, int(att.get("min_attainable_windows", 30)))
        except Exception:
            min_att = 30
        try:
            min_disc = max(0, int(att.get("min_effective_discordant", 5)))
        except Exception:
            min_disc = 5
        return (thresholds, min_att, min_disc)
    except Exception:
        return (_DEFAULT_THRESHOLDS, 30, 5)


def candidates_by_session_from_cache(
    sessions: Sequence[date],
    cache: object | None,
) -> dict[date, tuple[str, ...]]:
    out: dict[date, tuple[str, ...]] = {}
    scores = getattr(cache, "scores", None) if cache is not None else None
    universes = getattr(cache, "universes", None) if cache is not None else None
    for d in sessions:
        tickers: list[str] = []
        try:
            if isinstance(scores, Mapping) and d in scores and scores[d]:
                tickers = [str(k) for k in dict(scores[d]).keys()]
        except Exception:
            tickers = []
        if not tickers:
            try:
                if isinstance(universes, Mapping) and d in universes:
                    uni = universes[d]
                    raw = getattr(uni, "tickers", None)
                    if raw is not None:
                        tickers = [str(t) for t in list(raw)]
            except Exception:
                tickers = []
        out[d] = tuple(tickers)
    return out


def build_attainability_summary(
    *,
    sessions: Sequence[date],
    open_map: Mapping[date, Mapping[str, float]],
    candidates_by_session: Mapping[date, Sequence[str]],
    window_returns: Sequence[float],
    horizon: int,
    thresholds: Sequence[float] | None = None,
    min_attainable_windows: int = 30,
) -> dict[str, object]:
    thr = tuple(thresholds) if thresholds is not None else _DEFAULT_THRESHOLDS
    try:
        h = int(horizon)
    except Exception:
        h = 0
    if h <= 0 or not sessions:
        return {
            "attainability": {str(float(t)): 0.0 for t in thr},
            "capture": {str(float(t)): None for t in thr},
            "n_attainable": {str(float(t)): 0 for t in thr},
            "breadth_mean": 0.0,
        }
    open_by_session = [dict(open_map.get(s, {})) if isinstance(open_map, Mapping) else {} for s in sessions]
    opps = window_opportunities(open_by_session, sessions, candidates_by_session, h)
    aligned_returns: list[float] = []
    for i in range(len(sessions)):
        entry = i + 1
        exit_ = i + 1 + h
        if entry < 0 or exit_ >= len(sessions) or exit_ >= len(open_by_session):
            continue
        try:
            aligned_returns.append(float(window_returns[i]) if i < len(window_returns) else 0.0)
        except Exception:
            aligned_returns.append(0.0)
    att_curve = attainability_curve(opps, thr)
    capture_out: dict[str, float | None] = {}
    n_att_out: dict[str, int] = {}
    try:
        need = max(1, int(min_attainable_windows))
    except Exception:
        need = 1
    for t in thr:
        cap, n_att = capture_rate(aligned_returns, opps, float(t), min_attainable=need)
        capture_out[str(float(t))] = cap
        n_att_out[str(float(t))] = int(n_att)
    breadth_mean = 0.0
    if opps:
        try:
            breadth_mean = float(sum(int(o.breadth) for o in opps)) / float(len(opps))
        except Exception:
            breadth_mean = 0.0
    return {
        "attainability": {str(float(k)): float(v) for k, v in sorted(att_curve.items())},
        "capture": capture_out,
        "n_attainable": n_att_out,
        "breadth_mean": float(breadth_mean),
    }


def backtest_attainability_payload(
    *,
    calendar: object,
    panel: object,
    engine: object,
    model: object,
    case_config: object,
    rolling: object,
    horizon: int,
    shared_cache: object | None = None,
    leverage_allowed: bool | None = None,
    inverse_allowed: bool | None = None,
) -> dict[str, object]:
    from src.backtest.session_grid import resolve_session_grid

    thresholds, min_att_win, _ = load_attainability_config()
    cache_for_att = shared_cache
    if cache_for_att is None:
        try:
            from src.backtest.session_cache import build_session_cache

            cache_for_att = build_session_cache(
                engine,
                model,
                panel,
                case_config,
                leverage_allowed=leverage_allowed,
                inverse_allowed=inverse_allowed,
            )
        except Exception:
            cache_for_att = None
    start = getattr(case_config, "start", None)
    end = getattr(case_config, "end", None)
    try:
        att_sessions = list(
            resolve_session_grid(calendar.sessions(start, end), panel).sessions  # type: ignore[union-attr]
        )
    except Exception:
        try:
            att_sessions = list(calendar.sessions(start, end))  # type: ignore[union-attr]
        except Exception:
            att_sessions = list(getattr(rolling, "starts", ()) or ())
    open_map = getattr(cache_for_att, "open_map", None) if cache_for_att is not None else None
    if not isinstance(open_map, dict):
        try:
            from src.backtest.session_cache import _build_open_map

            open_map = _build_open_map(panel)  # type: ignore[arg-type]
        except Exception:
            open_map = {}
    cand_map = candidates_by_session_from_cache(att_sessions, cache_for_att)
    open_by_session = [dict(open_map.get(s, {})) if isinstance(open_map, dict) else {} for s in att_sessions]
    opps = window_opportunities(open_by_session, att_sessions, cand_map, int(horizon))
    att_curve = attainability_curve(opps, list(thresholds))
    payload = build_attainability_summary(
        sessions=att_sessions,
        open_map=open_map,
        candidates_by_session=cand_map,
        window_returns=list(getattr(rolling, "returns", ()) or ()),
        horizon=int(horizon),
        thresholds=list(thresholds),
        min_attainable_windows=int(min_att_win),
    )
    payload["attainability"] = {str(k): float(v) for k, v in sorted(att_curve.items())}
    return payload


def enrich_backtest_run_artifacts(
    meta: dict[str, object],
    summary: dict[str, object],
    *,
    calendar: object,
    panel: object,
    engine: object,
    model: object,
    case_config: object,
    rolling: object,
    horizon: int,
    shared_cache: object | None = None,
    leverage_allowed: bool | None = None,
    inverse_allowed: bool | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    from src.backtest.session_grid import phantom_session_labels

    start = getattr(case_config, "start", None)
    end = getattr(case_config, "end", None)
    try:
        cal_sessions = calendar.sessions(start, end)  # type: ignore[union-attr]
    except Exception:
        cal_sessions = []
    meta["phantom_sessions"] = phantom_session_labels(cal_sessions, panel)
    thresholds = [0.30, 0.40, 0.50, 0.60]
    _opps = window_opportunities([], [], {}, int(horizon))
    _ = attainability_curve(_opps, thresholds)
    summary.update(
        backtest_attainability_payload(
            calendar=calendar,
            panel=panel,
            engine=engine,
            model=model,
            case_config=case_config,
            rolling=rolling,
            horizon=int(horizon),
            shared_cache=shared_cache,
            leverage_allowed=leverage_allowed,
            inverse_allowed=inverse_allowed,
        )
    )
    return meta, summary
