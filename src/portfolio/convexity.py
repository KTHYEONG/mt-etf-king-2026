# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ConvexityHoldConfig:
    enabled: bool = False
    min_gap: float = 0.0
    score_drop_pct: float = 0.30
    crisis_regimes: frozenset[str] = frozenset({"STRONG_RISK_OFF"})
    w_top: float = 1.0
    max_gross: float = 2.0
    skip_capacity_route: bool = True

    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> ConvexityHoldConfig:
        if not isinstance(raw, Mapping):
            return cls()
        enabled = False
        min_gap = 0.0
        score_drop_pct = 0.30
        crisis_regimes: frozenset[str] = frozenset({"STRONG_RISK_OFF"})
        w_top = 1.0
        max_gross = 2.0
        skip_capacity_route = True
        try:
            if "enabled" in raw:
                enabled = bool(raw["enabled"])
        except Exception:
            enabled = False
        try:
            if "min_gap" in raw:
                min_gap = float(raw["min_gap"])  # type: ignore[arg-type]
        except Exception:
            min_gap = 0.0
        try:
            if "score_drop_pct" in raw:
                score_drop_pct = float(raw["score_drop_pct"])  # type: ignore[arg-type]
        except Exception:
            score_drop_pct = 0.30
        try:
            if "crisis_regimes" in raw:
                val = raw["crisis_regimes"]
                if isinstance(val, (list, tuple, set, frozenset)):
                    crisis_regimes = frozenset(str(x) for x in val)
                elif val is not None:
                    crisis_regimes = frozenset({str(val)})
        except Exception:
            crisis_regimes = frozenset({"STRONG_RISK_OFF"})
        try:
            if "w_top" in raw:
                w_top = float(raw["w_top"])  # type: ignore[arg-type]
        except Exception:
            w_top = 1.0
        try:
            if "max_gross" in raw:
                max_gross = float(raw["max_gross"])  # type: ignore[arg-type]
        except Exception:
            max_gross = 2.0
        try:
            if "skip_capacity_route" in raw:
                skip_capacity_route = bool(raw["skip_capacity_route"])
        except Exception:
            skip_capacity_route = True
        # if enabled not set or false due to missing, keep false
        if not isinstance(raw, Mapping) or "enabled" not in raw:
            # ensure defaults with enabled false when missing - already false unless explicit
            pass
        # malformed case still returns defaults with enabled False per spec: if any coercion failed but enabled true then?
        # spec says missing/malformed Mapping returns defaults with enabled=False never raise
        # if raw empty, enabled is False per above
        return cls(
            enabled=bool(enabled),
            min_gap=float(min_gap),
            score_drop_pct=float(score_drop_pct),
            crisis_regimes=crisis_regimes,
            w_top=float(w_top),
            max_gross=float(max_gross),
            skip_capacity_route=bool(skip_capacity_route),
        )


def convexity_active(
    leverage_allowed: bool | None,
    regime: str | None,
    scores: Mapping[str, float],
    config: ConvexityHoldConfig | None,
) -> bool:
    if config is None:
        return False
    try:
        if not bool(getattr(config, "enabled", False)):
            return False
    except Exception:
        return False
    if leverage_allowed is not True:
        return False
    if regime is not None:
        try:
            crisis = getattr(config, "crisis_regimes", frozenset({"STRONG_RISK_OFF"}))
            if str(regime) in crisis:
                return False
        except Exception:
            pass
    if not scores:
        return False
    try:
        mg = float(getattr(config, "min_gap", 0.0))
    except Exception:
        mg = 0.0
    if mg > 0:
        try:
            from src.portfolio.sizing import compute_confidence

            conf = float(compute_confidence(scores))
            if conf + 1e-12 < mg:
                return False
        except Exception:
            return False
    return True


def convexity_should_exit(
    current_top_score: float | None,
    entry_score: float | None,
    regime: str | None,
    config: ConvexityHoldConfig,
) -> bool:
    try:
        if not bool(getattr(config, "enabled", False)):
            return False
    except Exception:
        return False
    if regime is not None:
        try:
            crisis = getattr(config, "crisis_regimes", frozenset({"STRONG_RISK_OFF"}))
            if str(regime) in crisis:
                return True
        except Exception:
            pass
    if entry_score is None or current_top_score is None:
        return False
    try:
        entry = float(entry_score)
        cur = float(current_top_score)
        pct = float(getattr(config, "score_drop_pct", 0.30))
        if entry > 0 and cur <= entry * (1.0 - pct) + 1e-12:
            return True
    except Exception:
        return False
    return False


def resolve_convexity_vehicle(
    source_ticker: str,
    master: object,
    *,
    leverage_allowed: bool | None,
    confidence_low: bool = False,
) -> str:
    if master is None:
        return str(source_ticker)
    try:
        attrs = getattr(master, "attributes", None)
        if not isinstance(attrs, Mapping):
            return str(source_ticker)
        src_attr = attrs.get(str(source_ticker))
        if src_attr is None:
            return str(source_ticker)
        fam = str(getattr(src_attr, "leverage_family_key", str(source_ticker)))
    except Exception:
        return str(source_ticker)
    # collect members O(F)
    members: list[tuple[str, object]] = []
    try:
        for t, a in attrs.items():  # type: ignore[union-attr]
            try:
                fk = str(getattr(a, "leverage_family_key", t))
            except Exception:
                continue
            if fk == fam:
                members.append((str(t), a))
    except Exception:
        return str(source_ticker)
    if not members:
        return str(source_ticker)
    # prefer non-synthetic
    non_sync = [(t, a) for t, a in members if not bool(getattr(a, "is_synthetic", False))]
    candidates = non_sync if non_sync else members
    # if leverage allowed and confidence not low, prefer +2x
    if leverage_allowed is True and confidence_low is False:
        lev2 = sorted(t for t, a in candidates if int(getattr(a, "leverage_multiple", 1)) == 2)
        if lev2:
            return str(lev2[0])
        # fallback to +1x
        plus1 = sorted(t for t, a in candidates if int(getattr(a, "leverage_multiple", 1)) == 1)
        if plus1:
            return str(plus1[0])
        # no +1, return source
        return str(source_ticker)
    else:
        # return +1x
        plus1 = sorted(t for t, a in candidates if int(getattr(a, "leverage_multiple", 1)) == 1)
        if plus1:
            return str(plus1[0])
        # if no +1, return source if source is in family? else smallest ticker
        # tie-break ticker asc among +1? already sorted
        # fallback: if source in candidates return source else first candidate ticker asc
        tickers_sorted = sorted(t for t, _ in candidates)
        if str(source_ticker) in tickers_sorted:
            # if source not +1 but we forced +1, try to find +1 else keep source
            return str(source_ticker)
        return str(tickers_sorted[0]) if tickers_sorted else str(source_ticker)
