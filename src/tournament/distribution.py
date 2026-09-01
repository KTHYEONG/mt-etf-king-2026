# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date


def ruin_probability(returns: Sequence[float], threshold: float) -> float:
    if not returns:
        return 0.0
    n = len(returns)
    cnt = sum(1 for r in returns if float(r) < float(threshold))
    return float(cnt) / float(n) if n else 0.0


def effective_sample_size(n_windows: int, horizon: int) -> int:
    if horizon <= 0:
        raise ValueError("horizon must be >0")
    return int(n_windows // horizon)


def exceedance_curve(returns: Sequence[float], thresholds: Sequence[float]) -> dict[float, float]:
    if not returns:
        return {float(t): 0.0 for t in thresholds}
    n = len(returns)
    out: dict[float, float] = {}
    for t in thresholds:
        ft = float(t)
        cnt = sum(1 for r in returns if float(r) > ft)
        out[ft] = cnt / n if n else 0.0
    return out


def right_tail_score(returns: Sequence[float], weights: Mapping[float, float]) -> float:
    if not returns:
        return 0.0
    if not weights:
        return 0.0
    sorted_r = sorted(float(x) for x in returns)
    n = len(sorted_r)
    score = 0.0
    for q, w in weights.items():
        qf = float(q)
        wf = float(w)
        # empirical quantile: use linear interpolation or nearest rank?
        # Use numpy-style: index = q * (n-1) -> linear interpolate
        # For deterministic matching to expected tests, use ceil? Let's use linear.
        pos = qf * (n - 1) if n > 1 else 0.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            val = float(sorted_r[lo])
        else:
            frac = pos - lo
            val = float(sorted_r[lo]) * (1 - frac) + float(sorted_r[hi]) * frac
        score += wf * val
    return float(score)


def stationary_bootstrap_ci(
    returns: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    expected_block: int,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    # Stationary bootstrap: block lengths ~ Geom(p=1/expected_block)
    # Circular bootstrap over returns
    n = len(returns)
    if n == 0:
        return (0.0, 0.0)
    if expected_block <= 0:
        raise ValueError("expected_block must be >0")
    p = 1.0 / float(expected_block)
    rng = random.Random(seed)
    base_stat = float(statistic(list(returns)))
    # Generate resamples
    stats: list[float] = []
    arr = list(float(x) for x in returns)
    for _ in range(n_resamples):
        # generate stationary bootstrap sample of length n
        sample: list[float] = []
        while len(sample) < n:
            # start index uniformly
            start = rng.randrange(n)
            # block length geometric
            # Use geometric with p: L = ceil(log(U)/log(1-p))? But simpler: loop generating with p success per step.
            # Equivalent: sample length until termination with prob p after single draw
            length = 1
            while length < n - len(sample):
                # terminate with prob p, continue with 1-p
                if rng.random() < p:
                    break
                length += 1
            # Ensure at least 1
            length = max(1, length)
            # Append circular block
            for k in range(length):
                if len(sample) >= n:
                    break
                idx = (start + k) % n
                sample.append(arr[idx])
        # truncate to n
        sample = sample[:n]
        try:
            s = float(statistic(sample))
        except Exception:
            s = 0.0
        stats.append(s)
    stats.sort()
    # percentile interval
    lower_idx = int(math.floor((alpha / 2.0) * n_resamples))
    upper_idx = int(math.ceil((1 - alpha / 2.0) * n_resamples)) - 1
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))
    lower = float(stats[lower_idx])
    upper = float(stats[upper_idx])
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)

def vehicle_activity_rate(session_multiples: Sequence[int], risk_on_mask: Sequence[bool]) -> float:
    if not session_multiples or not risk_on_mask:
        return 0.0
    n = min(len(session_multiples), len(risk_on_mask))
    risk_count = 0
    active = 0
    for i in range(n):
        if bool(risk_on_mask[i]):
            risk_count += 1
            try:
                if int(session_multiples[i]) == 2:
                    active += 1
            except Exception:
                continue
    if risk_count == 0:
        return 0.0
    return float(active) / float(risk_count)


def evaluate_tail_gates(
    p_gt_40: float,
    b1_p_gt_40: float,
    p_gt_50: float,
    b1_p_gt_50: float,
    cvar: float,
    b1_cvar: float,
    activity_rate: float,
) -> tuple[str, list[str]]:
    fails: list[str] = []
    if not (float(p_gt_40) >= float(b1_p_gt_40) + 0.02 - 1e-12):
        fails.append("p_gt_40")
    if not (float(p_gt_50) >= float(b1_p_gt_50) + 0.01 - 1e-12):
        fails.append("p_gt_50")
    if not (float(cvar) >= float(b1_cvar) - 0.05 - 1e-12):
        fails.append("cvar_05")
    if not (float(activity_rate) >= 0.25 - 1e-12):
        fails.append("vehicle_activity")
    if not fails:
        return ("PASS", [])
    return ("FAIL", fails)


def evaluate_adoption_gates(
    p_gt_30: float,
    b1_p_gt_30: float,
    p_gt_40: float,
    b1_p_gt_40: float,
    cvar: float,
    b1_cvar: float,
    vehicle_rate: float,
    *,
    min_vehicle_rate: float = 0.25,
) -> tuple[str, list[str]]:
    fails: list[str] = []
    if not (float(p_gt_30) >= float(b1_p_gt_30) - 1e-12):
        fails.append("p_gt_30")
    if not (float(p_gt_40) >= float(b1_p_gt_40) + 0.02 - 1e-12):
        fails.append("p_gt_40")
    if not (float(cvar) >= float(b1_cvar) - 0.05 - 1e-12):
        fails.append("cvar_05")
    if not (float(vehicle_rate) >= float(min_vehicle_rate) - 1e-12):
        fails.append("vehicle_activity")
    if not fails:
        return ("PASS", [])
    return ("FAIL", fails)


def preflight_features_span_ok(gold_min: object, gold_max: object, silver_min: object, silver_max: object) -> bool:
    try:
        ok = gold_min <= silver_min and gold_max >= silver_max  # type: ignore[operator]
        return bool(ok)
    except Exception:
        return False


_RISK_ON_LABELS = frozenset({"RISK_ON", "STRONG_RISK_ON"})


def _regime_label(regime_snap: object | None) -> str | None:
    if regime_snap is None:
        return None
    st = getattr(regime_snap, "state", regime_snap)
    val = getattr(st, "value", st)
    return str(val)


def score_seed_for_vehicle_probe(master: object | None) -> dict[str, float]:
    if master is None:
        return {"069500": 1.0, "122630": 0.1}
    attrs = getattr(master, "attributes", None)
    if not isinstance(attrs, Mapping):
        return {"069500": 1.0, "122630": 0.1}
    by_family: dict[str, list[tuple[str, int]]] = {}
    for ticker, attr in attrs.items():
        fk = str(getattr(attr, "leverage_family_key", ticker))
        try:
            mult = int(getattr(attr, "leverage_multiple", 1))
        except Exception:
            mult = 1
        by_family.setdefault(fk, []).append((str(ticker), mult))
    for members in by_family.values():
        plus1 = sorted(t for t, m in members if m == 1)
        lev2 = sorted(t for t, m in members if m == 2)
        if plus1 and lev2:
            # spread required so confidence_vehicle_gate sees high conf (INV-12-4 probe)
            return {plus1[0]: 1.0, lev2[0]: 0.1}
    tickers = sorted(str(t) for t in attrs)
    if len(tickers) >= 2:
        return {tickers[0]: 1.0, tickers[1]: 0.1}
    if tickers:
        return {tickers[0]: 1.0}
    return {"069500": 1.0, "122630": 0.1}


def b1_gate_anchors_from_distribution(dist: ReturnDistribution) -> tuple[float, float, float]:
    exc = dist.exceedance if isinstance(dist.exceedance, Mapping) else {}
    p30 = float(exc.get(0.30, exc.get(0.3, 0.0)))
    p40 = float(exc.get(0.40, exc.get(0.4, 0.0)))
    if p30 == 0.0 or p40 == 0.0:
        for k, v in exc.items():
            try:
                fk = float(k)
                if p30 == 0.0 and abs(fk - 0.30) < 1e-9:
                    p30 = float(v)
                if p40 == 0.0 and abs(fk - 0.40) < 1e-9:
                    p40 = float(v)
            except Exception:
                continue
    return (float(p30), float(p40), float(dist.cvar_05))


def measure_vehicle_activity_from_allocate(
    model: object,
    sessions: Sequence[date],
    regimes: Mapping[date, object] | None,
    leverage_allowed: bool | None,
    score_seed: Mapping[str, float] | None = None,
) -> float:
    allocate = getattr(model, "allocate", None)
    if not callable(allocate):
        return 0.0
    seed = dict(score_seed) if score_seed else score_seed_for_vehicle_probe(getattr(model, "master", None))
    master = getattr(model, "master", None)
    attrs = getattr(master, "attributes", None) if master is not None else None
    multiples: list[int] = []
    risks: list[bool] = []
    for sess in sessions:
        regime_snap = regimes.get(sess) if regimes is not None else None
        regime_label = _regime_label(regime_snap)
        risk_on = regime_label in _RISK_ON_LABELS if regime_label is not None else False
        risks.append(risk_on)
        mult = 1
        try:
            dec = allocate(seed, regime=regime_label, leverage_allowed=leverage_allowed)
            weights = getattr(dec, "weights", {}) or {}
            for dst in weights:
                if attrs is not None and hasattr(attrs, "get"):
                    attr = attrs.get(dst)
                    if attr is not None:
                        mult = int(getattr(attr, "leverage_multiple", 1))
                        break
                mult = 1
        except Exception:
            mult = 1
        multiples.append(mult)
    return vehicle_activity_rate(multiples, risks)


def measure_vehicle_activity_from_session_cache(
    model: object,
    cache: object,
    regimes: Mapping[date, object] | None,
    leverage_allowed: bool | None,
    *,
    rescore_each_session: bool = True,
) -> float:
    allocate = getattr(model, "allocate", None)
    if not callable(allocate):
        return 0.0
    dates = getattr(cache, "dates", ())
    scores_map = getattr(cache, "scores", {})
    snapshots = getattr(cache, "snapshots", {})
    rules = getattr(cache, "rules", None)
    master = getattr(model, "master", None)
    attrs = getattr(master, "attributes", None) if master is not None else None
    multiples: list[int] = []
    risks: list[bool] = []
    for sess in dates:
        regime_snap = regimes.get(sess) if regimes is not None else None
        regime_label = _regime_label(regime_snap)
        risk_on = regime_label in _RISK_ON_LABELS if regime_label is not None else False
        risks.append(risk_on)
        if rescore_each_session:
            snap = snapshots.get(sess)
            score_fn = getattr(model, "score", None)
            if snap is not None and callable(score_fn):
                try:
                    from src.alpha.base import DecisionContext

                    ctx = DecisionContext(
                        decision_date=sess,
                        regime=regime_snap,  # type: ignore[arg-type]
                        capital=1_000_000_000.0,
                        held={},
                        rules=rules,  # type: ignore[arg-type]
                    )
                    score_fn(snap, ctx)
                except Exception:
                    pass
        scores = scores_map.get(sess, {})
        theme_states = None
        try:
            fn = getattr(model, "theme_states_by_representative", None)
            if callable(fn):
                theme_states = fn()
        except Exception:
            theme_states = None
        mult = 1
        try:
            dec = allocate(
                scores,
                regime=regime_label,
                leverage_allowed=leverage_allowed,
                theme_states=theme_states,
            )
            weights = getattr(dec, "weights", {}) or {}
            for dst, w in weights.items():
                if float(w) <= 1e-9:
                    continue
                if attrs is not None and hasattr(attrs, "get"):
                    attr = attrs.get(dst)
                    if attr is not None:
                        mult = int(getattr(attr, "leverage_multiple", 1))
                        break
        except Exception:
            mult = 1
        multiples.append(mult)
    return vehicle_activity_rate(multiples, risks)


def measure_vehicle_activity_from_top1_scores(
    model: object,
    cache: object,
    regimes: Mapping[date, object] | None,
    leverage_allowed: bool | None,
) -> float:
    _ = measure_vehicle_activity_from_top1_scores
    try:
        import polars as pl

        dates = getattr(cache, "dates", ())
        snapshots = getattr(cache, "snapshots", {})
        score_fn = getattr(model, "score", None)
        if not callable(score_fn):
            return 0.0
        multiples: list[int] = []
        risks: list[bool] = []
        for sess in dates:
            regime_snap = regimes.get(sess) if regimes is not None else None
            regime_label = _regime_label(regime_snap)
            risk_on = regime_label in _RISK_ON_LABELS if regime_label is not None else False
            risks.append(risk_on)
            snap = snapshots.get(sess) if isinstance(snapshots, Mapping) else None
            mult = 1
            try:
                if snap is None:
                    mult = 1
                else:
                    from src.alpha.base import DecisionContext

                    # build minimal context for scoring
                    try:
                        rules = getattr(cache, "rules", None)
                    except Exception:
                        rules = None
                    try:
                        ctx = DecisionContext(
                            decision_date=sess,
                            regime=regime_snap,  # type: ignore[arg-type]
                            capital=1_000_000_000.0,
                            held={},
                            rules=rules,  # type: ignore[arg-type]
                        )
                    except Exception:
                        ctx = None  # type: ignore[assignment]
                    scores = {}
                    try:
                        if ctx is not None:
                            scores = score_fn(snap, ctx) or {}
                        else:
                            scores = score_fn(snap, {})  # type: ignore[arg-type]
                    except Exception:
                        scores = {}
                    if not scores:
                        mult = 1
                    else:
                        # top ticker by (-score, ticker)
                        sorted_items = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
                        top_ticker = str(sorted_items[0][0])
                        # find name for top ticker from snapshot
                        name = ""
                        try:
                            if hasattr(snap, "filter"):
                                # polars DataFrame
                                filt = snap.filter(pl.col("ticker") == top_ticker) if "ticker" in snap.columns else None  # type: ignore[attr-defined]
                                if filt is not None and filt.height > 0 and "name" in filt.columns:
                                    name = str(filt.select(pl.col("name")).to_series().to_list()[0] or "")
                        except Exception:
                            name = ""
                        if not name:
                            # fallback iterate
                            try:
                                for row in snap.iter_rows(named=True):  # type: ignore[attr-defined]
                                    if str(row.get("ticker")) == top_ticker:
                                        nv = row.get("name")
                                        if nv is not None:
                                            name = str(nv)
                                        break
                            except Exception:
                                name = ""
                        if name:
                            try:
                                from src.universe.instruments import resolve_leverage as _res

                                lev, _conf = _res(name)
                                mult = int(lev)
                            except Exception:
                                mult = 1
                        else:
                            mult = 1
            except Exception:
                mult = 1
            multiples.append(mult)
        return vehicle_activity_rate(multiples, risks)
    except Exception:
        return 0.0


def resolve_adoption_vehicle_rate(
    model: object,
    engine: object,
    panel: object,
    config: object,
    regimes: Mapping[date, object] | None,
    leverage_allowed: bool | None,
    inverse_allowed: bool | None = None,
) -> float:
    allocate = getattr(model, "allocate", None)
    if not callable(allocate):
        # score-only path: build session cache and measure via TOP1
        try:
            from src.backtest.session_cache import build_session_cache

            cache = build_session_cache(
                engine,
                model,
                panel,  # type: ignore[arg-type]
                config,
                leverage_allowed=leverage_allowed,
                inverse_allowed=inverse_allowed,
            )
        except Exception:
            return 0.0
        return measure_vehicle_activity_from_top1_scores(model, cache, regimes, leverage_allowed)
    reset = getattr(model, "reset_trackers", None)
    if callable(reset):
        import contextlib

        with contextlib.suppress(Exception):
            reset()
    try:
        from src.backtest.session_cache import build_session_cache

        cache = build_session_cache(
            engine,
            model,
            panel,  # type: ignore[arg-type]
            config,
            leverage_allowed=leverage_allowed,
            inverse_allowed=inverse_allowed,
        )
    except Exception:
        return 0.0
    return measure_vehicle_activity_from_session_cache(
        model,
        cache,
        regimes,
        leverage_allowed,
    )


@dataclass(frozen=True)
class ReturnDistribution:
    name: str
    horizon: int
    returns: tuple[float, ...]
    n_windows: int
    n_effective: int
    quantiles: Mapping[float, float]
    exceedance: Mapping[float, float]
    cvar_05: float
    right_tail_score: float
    giveback_median: float = 0.0
    giveback_q90: float = 0.0

    @classmethod
    def summarise(
        cls,
        name: str,
        returns: Sequence[float],
        horizon: int,
        thresholds: Sequence[float],
        tail_weights: Mapping[float, float],
        givebacks: Sequence[float] = (),
    ) -> ReturnDistribution:
        n = len(returns)
        n_eff = effective_sample_size(n, horizon)
        # quantiles at 0.05,0.25,0.50,0.75,0.90,0.95,0.99
        q_levels = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        sorted_r = sorted(float(x) for x in returns)
        quantiles: dict[float, float] = {}
        for q in q_levels:
            if n == 0:
                quantiles[float(q)] = 0.0
            elif n == 1:
                quantiles[float(q)] = float(sorted_r[0])
            else:
                pos = q * (n - 1)
                lo = int(math.floor(pos))
                hi = int(math.ceil(pos))
                if lo == hi:
                    val = float(sorted_r[lo])
                else:
                    frac = pos - lo
                    val = float(sorted_r[lo]) * (1 - frac) + float(sorted_r[hi]) * frac
                quantiles[float(q)] = float(val)
        exc = exceedance_curve(returns, thresholds)
        # CVaR at 5%: mean of returns <= quantile 0.05
        if n == 0:
            cvar = 0.0
        else:
            q05 = quantiles[0.05]
            tail = [float(x) for x in returns if float(x) <= q05]
            if not tail:
                # if no tail beyond due to interpolation, take smallest 5% count
                k = max(1, int(math.ceil(0.05 * n)))
                tail = sorted_r[:k]
            cvar = sum(tail) / len(tail) if tail else 0.0
        rts = right_tail_score(returns, tail_weights)
        # giveback quantiles using same linear interpolation
        def _q(vals: Sequence[float], level: float) -> float:
            if not vals:
                return 0.0
            s = sorted(float(x) for x in vals)
            nn = len(s)
            if nn == 1:
                return float(s[0])
            pos = level * (nn - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(s[lo])
            frac = pos - lo
            return float(s[lo]) * (1 - frac) + float(s[hi]) * frac

        gb_median = _q(givebacks, 0.50) if givebacks else 0.0
        gb_q90 = _q(givebacks, 0.90) if givebacks else 0.0
        return cls(
            name=name,
            horizon=horizon,
            returns=tuple(float(x) for x in returns),
            n_windows=n,
            n_effective=n_eff,
            quantiles=quantiles,
            exceedance=exc,
            cvar_05=float(cvar),
            right_tail_score=float(rts),
            giveback_median=float(gb_median),
            giveback_q90=float(gb_q90),
        )


def locked_window_returns(
    daily_rets: Sequence[float], horizon: int, lock_level: float
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        ll = float(lock_level)
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        locked = False
        for k in range(h):
            if locked:
                continue
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            # compound
            equity *= 1.0 + r
            if equity >= 1.0 + ll - 1e-12:
                locked = True
        out.append(float(equity - 1.0))
    return out


def championship_lock_returns(
    daily_rets: Sequence[float], horizon: int, arm: float, trail: float = 0.0
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    try:
        ll = float(arm)
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        tr = float(trail)
    except Exception:
        tr = float("nan")
    if not math.isfinite(tr) or tr <= 0:
        return locked_window_returns(daily_rets, horizon, arm)
    # trail >0 hysteresis: floor-protected
    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        peak = 1.0
        locked = False
        locked_equity = 1.0
        for k in range(h):
            if locked:
                continue
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
            if equity > peak:
                peak = float(equity)
            stop = max(1.0 + ll, float(peak) - float(tr))
            if float(peak) > float(stop) and float(equity) < float(stop):
                equity = float(stop)
                locked = True
                peak = float(stop)
        out.append(float(equity - 1.0))
    return out


def house_money_ratchet_returns(
    daily_rets: Sequence[float], horizon: int, arm: float, lock_remaining: int = 5
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    try:
        ll = float(arm)  # type: ignore[arg-type]
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        lf = float(lock_remaining)  # type: ignore[arg-type]
        if not math.isfinite(lf) or lf < 0:
            lr = 5
        else:
            lr = int(lf)
    except Exception:
        lr = 5
    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        armed = False
        locked = False
        for k in range(h):
            if locked:
                break
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
            if not armed and equity >= 1.0 + ll - 1e-12:
                armed = True
            if armed:
                if equity < 1.0 + ll - 1e-12:
                    equity = 1.0 + ll
                    locked = True
                else:
                    remaining_after = h - 1 - k
                    if remaining_after <= lr:
                        locked = True
        out.append(float(equity - 1.0))
    return out


def execution_faithful_late_lock_returns(
    daily_rets: Sequence[float], horizon: int, arm: float, lock_remaining: int
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    try:
        ll = float(arm)  # type: ignore[arg-type]
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        lf = float(lock_remaining)  # type: ignore[arg-type]
        if not math.isfinite(lf) or lf < 0:
            lr = 5
        else:
            lr = int(lf)
    except Exception:
        lr = 5
    # use live predicate without floor invention
    from src.tournament.policy import house_money_should_cash

    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        cashed = False
        cashed_equity = 1.0
        for k in range(h):
            if cashed:
                break
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
            ret_from_start = float(equity - 1.0)
            remaining_after = h - 1 - k
            try:
                if house_money_should_cash(ret_from_start, remaining_after, ll, lr):
                    cashed = True
                    cashed_equity = float(equity)
                    break
            except Exception:
                continue
        if cashed:
            out.append(float(cashed_equity - 1.0))
        else:
            out.append(float(equity - 1.0))
    return out


def continuation_capture(
    unlocked: Sequence[float],
    freeze: Sequence[float],
    ratchet: Sequence[float],
    arm: float,
    *,
    eps: float = 0.10,
) -> float:
    try:
        af = float(arm)  # type: ignore[arg-type]
        if not math.isfinite(af):
            return 0.0
    except Exception:
        return 0.0
    try:
        ef = float(eps)  # type: ignore[arg-type]
        if not math.isfinite(ef):
            ef = 0.10
    except Exception:
        ef = 0.10
    threshold = af + ef
    n = min(len(unlocked), len(freeze), len(ratchet))
    if n == 0:
        return 0.0
    diffs: list[float] = []
    for idx in range(n):
        try:
            uv = float(unlocked[idx])  # type: ignore[index]
        except Exception:
            continue
        if not math.isfinite(uv):
            continue
        if uv > threshold:
            try:
                fv = float(freeze[idx])  # type: ignore[index]
                rv = float(ratchet[idx])  # type: ignore[index]
            except Exception:
                continue
            if not math.isfinite(fv) or not math.isfinite(rv):
                continue
            diffs.append(float(rv - fv))
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def overlay_right_tail_stats(terminals: Sequence[float]) -> dict[str, float]:
    if not terminals:
        return {"q99": 0.0, "mean": 0.0, "p_gt_50": 0.0, "p_gt_60": 0.0, "p_gt_80": 0.0}
    try:
        vals = [float(x) for x in terminals]  # type: ignore[arg-type]
        vals = [v for v in vals if math.isfinite(v)]
    except Exception:
        return {"q99": 0.0, "mean": 0.0, "p_gt_50": 0.0, "p_gt_60": 0.0, "p_gt_80": 0.0}
    if not vals:
        return {"q99": 0.0, "mean": 0.0, "p_gt_50": 0.0, "p_gt_60": 0.0, "p_gt_80": 0.0}
    n = len(vals)
    mean_v = float(sum(vals) / n) if n else 0.0
    # q99 via linear interpolation pos = 0.99*(n-1)
    sorted_v = sorted(vals)
    if n == 1:
        q99 = float(sorted_v[0])
    else:
        pos = 0.99 * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            q99 = float(sorted_v[lo])
        else:
            frac = pos - lo
            q99 = float(sorted_v[lo]) * (1 - frac) + float(sorted_v[hi]) * frac
    def _p(th: float) -> float:
        cnt = sum(1 for v in vals if v > th)
        return float(cnt) / float(n) if n else 0.0
    return {
        "q99": float(q99),
        "mean": float(mean_v),
        "p_gt_50": float(_p(0.50)),
        "p_gt_60": float(_p(0.60)),
        "p_gt_80": float(_p(0.80)),
    }


def evaluate_p25_adoption_gates(
    q99_ratchet: float,
    q99_freeze: float,
    p_gt_60_ratchet: float,
    p_gt_60_freeze: float,
    p_gt_60_unlocked: float,
    continuation_capture: float,
    ruin: float,
    vehicle_rate: float,
    *,
    ruin_max: float = 0.05,
    min_vehicle_rate: float = 0.25,
) -> tuple[str, list[str]]:
    def _f(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv):
                return 0.0
            return float(fv)
        except Exception:
            return 0.0
    q99_r = _f(q99_ratchet)
    q99_f = _f(q99_freeze)
    p60_r = _f(p_gt_60_ratchet)
    p60_f = _f(p_gt_60_freeze)
    p60_u = _f(p_gt_60_unlocked)
    cc = _f(continuation_capture)
    ru = _f(ruin)
    vr = _f(vehicle_rate)
    try:
        rm = float(ruin_max)  # type: ignore[arg-type]
        if not math.isfinite(rm):
            rm = 0.05
    except Exception:
        rm = 0.05
    try:
        mv = float(min_vehicle_rate)  # type: ignore[arg-type]
        if not math.isfinite(mv):
            mv = 0.25
    except Exception:
        mv = 0.25
    fails: list[str] = []
    if q99_r + 1e-12 < q99_f:
        fails.append("q99")
    if p60_u > 1e-12 and p60_r <= 1e-12:
        fails.append("right_tail_censored")
    if p60_r + 1e-12 < p60_f:
        fails.append("p_gt_60")
    if p60_u > 1e-12 and cc <= 1e-12:
        fails.append("continuation")
    if ru > rm + 1e-12:
        fails.append("ruin")
    if vr + 1e-12 < mv:
        fails.append("vehicle_activity")
    if not fails:
        return ("PASS", [])
    return ("FAIL", fails)


def evaluate_p24_adoption_gates(
    p_gt_30: float,
    b1_p_gt_30: float,
    p_gt_40: float,
    b1_p_gt_40: float,
    p_gt_50: float,
    b1_p_gt_50: float,
    cvar: float,
    b1_cvar: float,
    vehicle_rate: float,
    *,
    min_vehicle_rate: float = 0.25,
) -> tuple[str, list[str]]:
    def _cand(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            return 0.0
        if not math.isfinite(fv):
            return 0.0
        return float(fv)

    def _b1(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            return float("inf")
        if not math.isfinite(fv):
            return float("inf")
        return float(fv)

    p30 = _cand(p_gt_30)
    b30 = _b1(b1_p_gt_30)
    p40 = _cand(p_gt_40)
    b40 = _b1(b1_p_gt_40)
    p50 = _cand(p_gt_50)
    b50 = _b1(b1_p_gt_50)
    cv = _cand(cvar)
    bcv = _b1(b1_cvar)
    vr = _cand(vehicle_rate)
    try:
        mv = float(min_vehicle_rate)  # type: ignore[arg-type]
        if not math.isfinite(mv):
            mv = float("inf")
    except Exception:
        mv = float("inf")
    fails: list[str] = []
    if not (float(p30) >= float(b30) - 1e-12):
        fails.append("p_gt_30")
    if not (float(p40) >= float(b40) + 0.02 - 1e-12):
        fails.append("p_gt_40")
    if not (float(p50) >= float(b50) + 0.02 - 1e-12):
        fails.append("p_gt_50")
    if not (float(cv) >= float(bcv) - 0.05 - 1e-12):
        fails.append("cvar_05")
    if not (float(vr) >= float(mv) - 1e-12):
        fails.append("vehicle_activity")
    if not fails:
        return ("PASS", [])
    return ("FAIL", fails)
