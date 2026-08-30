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
        return 0.0
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
