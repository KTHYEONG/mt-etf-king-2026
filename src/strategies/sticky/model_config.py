# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl

from src.alpha.base import DecisionContext
from src.strategies.sticky.capacity import apply_capacity_filter, cached_filtered_scores
from src.universe.instruments import resolve_leverage


logger = logging.getLogger(__name__)


DEFAULT_EXCLUDE_NAME_TOKENS: Final[tuple[str, ...]] = ("국채", "채권", "달러", "엔선물", "골드", "금선물", "gold", "커버드콜", "스티프너", "플래트너")

# Fail-closed default: the crash-rebound bypass is never parsed from YAML.
crash_rebound_abs_mom_bypass: bool = False


def name_excluded(name: str, tokens: Sequence[str]) -> bool:
    try:
        text = name.casefold() if isinstance(name, str) else ""
    except Exception:
        return True
    if not text.strip():
        return True
    for tok in tokens:
        t = tok.casefold() if isinstance(tok, str) else ""
        if t and t in text:
            return True
    return False


def cross_section_percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    # Min-max scaling to [0,1] so near-tied momenta stay near-tied (P29V volume tie-break).
    items: list[tuple[str, float]] = []
    try:
        raw = dict(values)
    except Exception:
        return {}
    for k, v in raw.items():
        try:
            fv = float(v)
            if math.isfinite(fv):
                items.append((str(k), float(fv)))
        except Exception:
            continue
    if not items:
        return {}
    lo = min(v for _, v in items)
    hi = max(v for _, v in items)
    if not math.isfinite(hi - lo) or hi <= lo:
        return {k: 1.0 for k, _ in items}
    return {k: (v - lo) / (hi - lo) for k, v in items}


def blend_rank_scores(primary: Mapping[str, float], aux: Mapping[str, float], *, w_primary: float, w_aux: float) -> dict[str, float]:
    try:
        wp, wa = float(w_primary), float(w_aux)
        p = {str(k): float(v) for k, v in dict(primary).items()}
        a = {str(k): float(v) for k, v in dict(aux).items()}
    except Exception:
        return {}
    if not math.isfinite(wp) or not math.isfinite(wa):
        return {}
    keys = [k for k in p if k in a and math.isfinite(p[k]) and math.isfinite(a[k])]
    if not keys:
        return {}
    rp = cross_section_percentile_ranks({k: p[k] for k in keys})
    ra = cross_section_percentile_ranks({k: a[k] for k in keys})
    return {k: wp * rp[k] + wa * ra[k] for k in keys}


def momentum_horizon(mom_col: str, *, default: int = 5) -> int:
    try: d = int(default)
    except Exception: d = 5
    if d <= 0: d = 5
    try:
        s = str(mom_col).strip().rsplit("_", 1)[-1] if isinstance(mom_col, str) and "_" in str(mom_col) else ""
        v = int(s) if s.isdigit() else int(d)
        return int(v) if int(v) > 0 else int(d)
    except Exception: return int(d)


@dataclass
class StickyLeaderConfig:
    mom_col: str = "mom_20"
    only_plus_2: bool = True
    no_inverse: bool = True
    min_gap: float = 0.08
    min_hold: int = 3
    impulse_col: str = "mom_5"
    impulse_gap: float = 0.0
    impulse_require_volx: bool = True
    cash_drawdown: float = 0.0
    collapse_family: bool = False
    lock_level: float = 0.0
    same_leader_hold: bool = False
    abs_mom_cash: bool = False
    abs_mom_exit: float = 0.0
    exclude_name_tokens: tuple[str, ...] = ()
    score_aux_col: str | None = None
    score_aux_weight: float = 0.0
    exclude_synthetic: bool = False
    min_fill_ratio: float = 0.0
    runner_reversal_exit: bool = False
    runner_mom_col: str = "mom_5"
    inactive_participation: bool = False
    inactive_score_col: str = "rv_20"
    inactive_min_weight: float = 0.30
    inactive_stop_drawdown: float = 0.15
    crash_rebound_abs_mom_bypass: bool = crash_rebound_abs_mom_bypass
    post_crash_anchor: bool = False
    anchor_tickers: tuple[str, ...] = ()
    anchor_score_col: str = "mom_20"
    anchor_stop_drawdown: float = 0.15
    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> StickyLeaderConfig:
        defaults = cls()
        if not isinstance(raw, Mapping):
            return defaults
        # mom_col
        mom_col = defaults.mom_col
        try:
            if "mom_col" in raw:
                v = raw["mom_col"]
                if isinstance(v, str) and v:
                    mom_col = str(v)
        except Exception:
            mom_col = defaults.mom_col
        # only_plus_2
        only_plus_2 = defaults.only_plus_2
        try:
            if "only_plus_2" in raw:
                only_plus_2 = bool(raw["only_plus_2"])
        except Exception:
            only_plus_2 = defaults.only_plus_2
        # no_inverse
        no_inverse = defaults.no_inverse
        try:
            if "no_inverse" in raw:
                no_inverse = bool(raw["no_inverse"])
        except Exception:
            no_inverse = defaults.no_inverse
        # min_gap
        min_gap = defaults.min_gap
        try:
            if "min_gap" in raw:
                mg = float(raw["min_gap"])  # type: ignore[arg-type]
                if not math.isfinite(mg) or mg < 0:
                    min_gap = defaults.min_gap
                else:
                    min_gap = float(mg)
        except Exception:
            min_gap = defaults.min_gap
        # handle non-finite after parse (already handled) and negative
        if not math.isfinite(min_gap) or min_gap < 0:
            min_gap = defaults.min_gap
        # min_hold
        min_hold = defaults.min_hold
        try:
            if "min_hold" in raw:
                mh_raw = raw["min_hold"]
                mh = int(mh_raw)  # type: ignore[arg-type]
                # also check finiteness via float
                try:
                    f = float(mh_raw)  # type: ignore[arg-type]
                    if not math.isfinite(f):
                        raise ValueError
                except Exception:
                    raise
                if mh < 0:
                    min_hold = defaults.min_hold
                else:
                    min_hold = int(mh)
        except Exception:
            # if min_hold present but invalid, fail to defaults per spec (e.g., -2)
            if "min_hold" in raw:
                min_hold = defaults.min_hold
            else:
                min_hold = defaults.min_hold
        # additional guard for non-finite / negative after
        try:
            if not math.isfinite(float(min_hold)):
                min_hold = defaults.min_hold
        except Exception:
            min_hold = defaults.min_hold
        if min_hold < 0:
            min_hold = defaults.min_hold
        # impulse_col
        impulse_col = defaults.impulse_col
        try:
            if "impulse_col" in raw:
                v = raw["impulse_col"]
                if isinstance(v, str) and v.strip():
                    impulse_col = str(v).strip()
                else:
                    impulse_col = defaults.impulse_col
        except Exception:
            impulse_col = defaults.impulse_col
        if not isinstance(impulse_col, str) or not impulse_col.strip():
            impulse_col = defaults.impulse_col
        # impulse_gap fail-closed: NaN/negative -> 0.0 disabled
        impulse_gap = defaults.impulse_gap
        try:
            if "impulse_gap" in raw:
                ig = float(raw["impulse_gap"])  # type: ignore[arg-type]
                if not math.isfinite(ig) or ig < 0:
                    impulse_gap = 0.0
                else:
                    impulse_gap = float(ig)
        except Exception:
            impulse_gap = 0.0
        if not math.isfinite(impulse_gap) or impulse_gap < 0:
            impulse_gap = 0.0
        # impulse_require_volx
        impulse_require_volx = defaults.impulse_require_volx
        try:
            if "impulse_require_volx" in raw:
                impulse_require_volx = bool(raw["impulse_require_volx"])
        except Exception:
            impulse_require_volx = defaults.impulse_require_volx
        # cash_drawdown fail-closed: >0 -> 0.0, non-finite ->0.0, default 0.0 disabled
        cash_drawdown = defaults.cash_drawdown
        try:
            if "cash_drawdown" in raw:
                cd = float(raw["cash_drawdown"])  # type: ignore[arg-type]
                if not math.isfinite(cd):
                    cash_drawdown = 0.0
                elif cd > 0:
                    cash_drawdown = 0.0
                else:
                    cash_drawdown = float(cd)
        except Exception:
            cash_drawdown = 0.0
        if not math.isfinite(cash_drawdown) or cash_drawdown > 0:
            cash_drawdown = 0.0
        collapse_family = defaults.collapse_family
        try:
            if "collapse_family" in raw:
                collapse_family = bool(raw["collapse_family"])
        except Exception:
            collapse_family = defaults.collapse_family
        exclude_name_tokens = tuple(defaults.exclude_name_tokens)
        try:
            v = raw.get("exclude_name_tokens")
            if isinstance(v, (list, tuple)):
                exclude_name_tokens = tuple(str(t) for t in v if isinstance(t, str) and str(t))
        except Exception:
            pass
        score_aux_col = defaults.score_aux_col
        try:
            v = raw.get("score_aux_col")
            if isinstance(v, str) and v.strip():
                score_aux_col = str(v).strip()
            elif v is None:
                score_aux_col = None
        except Exception:
            pass
        score_aux_weight = defaults.score_aux_weight
        try:
            w = float(raw.get("score_aux_weight", score_aux_weight))  # type: ignore[arg-type]
            score_aux_weight = float(w) if math.isfinite(w) and w >= 0 else 0.0
        except Exception:
            score_aux_weight = 0.0
        exclude_synthetic = defaults.exclude_synthetic
        try:
            if "exclude_synthetic" in raw:
                exclude_synthetic = bool(raw["exclude_synthetic"])
        except Exception:
            exclude_synthetic = defaults.exclude_synthetic
        min_fill_ratio = defaults.min_fill_ratio
        try:
            if "min_fill_ratio" in raw:
                mfr = float(raw["min_fill_ratio"])  # type: ignore[arg-type]
                min_fill_ratio = float(mfr) if math.isfinite(mfr) and mfr >= 0 else 0.0
        except Exception:
            min_fill_ratio = 0.0
        if not math.isfinite(float(min_fill_ratio)) or float(min_fill_ratio) < 0:
            min_fill_ratio = 0.0
        abs_mom_cash = defaults.abs_mom_cash
        try:
            if "abs_mom_cash" in raw:
                abs_mom_cash = bool(raw["abs_mom_cash"])
        except Exception:
            abs_mom_cash = defaults.abs_mom_cash
        try: abs_mom_exit = float(raw.get("abs_mom_exit", defaults.abs_mom_exit))  # type: ignore[arg-type]
        except Exception: abs_mom_exit = defaults.abs_mom_exit
        same_leader_hold = defaults.same_leader_hold
        try:
            if "same_leader_hold" in raw:
                same_leader_hold = bool(raw["same_leader_hold"])
        except Exception:
            same_leader_hold = defaults.same_leader_hold
        runner_reversal_exit = defaults.runner_reversal_exit
        try:
            if "runner_reversal_exit" in raw:
                runner_reversal_exit = bool(raw["runner_reversal_exit"])
        except Exception:
            runner_reversal_exit = defaults.runner_reversal_exit
        runner_mom_col = defaults.runner_mom_col
        try:
            value = raw.get("runner_mom_col")
            if isinstance(value, str) and value.strip():
                runner_mom_col = value.strip()
        except Exception:
            runner_mom_col = defaults.runner_mom_col
        inactive_participation = defaults.inactive_participation
        try:
            if "inactive_participation" in raw:
                v = raw["inactive_participation"]
                if isinstance(v, bool):
                    inactive_participation = bool(v)
        except (TypeError, ValueError):
            inactive_participation = defaults.inactive_participation
        inactive_score_col = defaults.inactive_score_col
        try:
            v = raw.get("inactive_score_col")
            if isinstance(v, str) and v:
                inactive_score_col = str(v)
        except (TypeError, ValueError):
            inactive_score_col = defaults.inactive_score_col
        inactive_min_weight = defaults.inactive_min_weight
        try:
            if "inactive_min_weight" in raw:
                vv = float(raw["inactive_min_weight"])  # type: ignore[arg-type]
                if math.isfinite(vv) and 0 < vv <= 1.0:
                    inactive_min_weight = float(vv)
        except (TypeError, ValueError):
            inactive_min_weight = defaults.inactive_min_weight
        inactive_stop_drawdown = defaults.inactive_stop_drawdown
        try:
            if "inactive_stop_drawdown" in raw:
                vv = float(raw["inactive_stop_drawdown"])  # type: ignore[arg-type]
                if math.isfinite(vv) and 0 < vv < 1.0:
                    inactive_stop_drawdown = float(vv)
        except (TypeError, ValueError):
            inactive_stop_drawdown = defaults.inactive_stop_drawdown
        if "post_crash_anchor" in raw:
            if not isinstance(raw["post_crash_anchor"], bool):
                raise ValueError("post_crash_anchor must be bool")
            post_crash_anchor = bool(raw["post_crash_anchor"])
        else:
            post_crash_anchor = False
        if "anchor_tickers" in raw:
            tickers_raw = raw["anchor_tickers"]
            if not isinstance(tickers_raw, (list, tuple)):
                raise ValueError("anchor_tickers must be a list or tuple of non-empty strings")
            cleaned: list[str] = []
            for el in tickers_raw:
                if not isinstance(el, str) or not el.strip():
                    raise ValueError("anchor_tickers must be a list or tuple of non-empty strings")
                cleaned.append(el.strip())
            anchor_tickers = tuple(cleaned)
        else:
            anchor_tickers = ()
        if "anchor_score_col" in raw:
            score_raw = raw["anchor_score_col"]
            if not isinstance(score_raw, str) or not score_raw.strip():
                raise ValueError("anchor_score_col must be a non-empty string")
            anchor_score_col = str(score_raw)
        else:
            anchor_score_col = "mom_20"
        if "anchor_stop_drawdown" in raw:
            stop_raw = raw["anchor_stop_drawdown"]
            if isinstance(stop_raw, bool) or not isinstance(stop_raw, (int, float, str)):
                raise ValueError("anchor_stop_drawdown must be in (0, 1)")
            stop_val = float(stop_raw)
            if not math.isfinite(stop_val) or not 0 < stop_val < 1:
                raise ValueError(f"anchor_stop_drawdown must be in (0, 1), got {stop_raw!r}")
            anchor_stop_drawdown = float(stop_val)
        else:
            anchor_stop_drawdown = 0.15
        if post_crash_anchor and not anchor_tickers:
            raise ValueError("post_crash_anchor requires non-empty anchor_tickers")
        return cls(
            mom_col=str(mom_col),
            only_plus_2=bool(only_plus_2),
            no_inverse=bool(no_inverse),
            min_gap=float(min_gap),
            min_hold=int(min_hold),
            impulse_col=str(impulse_col),
            impulse_gap=float(impulse_gap),
            impulse_require_volx=bool(impulse_require_volx),
            cash_drawdown=float(cash_drawdown),
            collapse_family=bool(collapse_family),
            lock_level=float(defaults.lock_level),
            same_leader_hold=bool(same_leader_hold),
            exclude_name_tokens=tuple(exclude_name_tokens),
            score_aux_col=score_aux_col,
            score_aux_weight=float(score_aux_weight),
            exclude_synthetic=bool(exclude_synthetic),
            min_fill_ratio=float(min_fill_ratio),
            abs_mom_cash=bool(abs_mom_cash),
            abs_mom_exit=float(abs_mom_exit),
            runner_reversal_exit=bool(runner_reversal_exit),
            runner_mom_col=str(runner_mom_col),
            inactive_participation=bool(inactive_participation),
            inactive_score_col=str(inactive_score_col),
            inactive_min_weight=float(inactive_min_weight),
            inactive_stop_drawdown=float(inactive_stop_drawdown),
            crash_rebound_abs_mom_bypass=bool(defaults.crash_rebound_abs_mom_bypass),
            post_crash_anchor=bool(post_crash_anchor),
            anchor_tickers=tuple(anchor_tickers),
            anchor_score_col=str(anchor_score_col),
            anchor_stop_drawdown=float(anchor_stop_drawdown),
        )
