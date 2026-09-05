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
        s = str(mom_col).strip().rsplit("_", 1)[-1] if isinstance(mom_col, str) and "_" in str(mom_col) else ""; v = int(s) if s.isdigit() else int(d); return int(v) if int(v) > 0 else int(d)
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
        )
def collapse_plus2_by_family(scores: Mapping[str, float], snapshot: pl.DataFrame, adv_col: str = "trading_value") -> dict[str, float]:
    if not scores:
        return {}
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return {}
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return {}
    except Exception:
        return {}
    if "ticker" not in snapshot.columns:
        # without ticker column, treat each ticker as own family
        return dict(scores)
    # Build row lookup and family groups
    row_by_ticker: dict[str, dict] = {}
    try:
        for row in snapshot.iter_rows(named=True):
            try:
                t = str(row.get("ticker"))
            except Exception:
                continue
            if t in scores:
                row_by_ticker[t] = row
    except Exception:
        return dict(scores)
    has_underlying = "underlying_index_name" in snapshot.columns
    has_adv = adv_col in snapshot.columns
    # Group tickers by family
    family_groups: dict[str, list[str]] = {}
    for ticker in scores.keys():
        t_str = str(ticker)
        family: str
        row = row_by_ticker.get(t_str)
        if has_underlying and row is not None:
            try:
                val = row.get("underlying_index_name")
                if val is not None:
                    # handle polars null or nan
                    s = str(val).strip()
                    if s and s.lower() != "none" and s.lower() != "nan":
                        # need to check if original was None, but str(None) == "None" filtered above
                        # also check if val is float nan
                        try:
                            if isinstance(val, float) and not math.isfinite(val):
                                family = t_str
                            else:
                                family = s
                        except Exception:
                            family = s
                    else:
                        family = t_str
                else:
                    family = t_str
            except Exception:
                family = t_str
        else:
            family = t_str
        family_groups.setdefault(family, []).append(t_str)
    out: dict[str, float] = {}
    for family, tickers in family_groups.items():
        if len(tickers) == 1:
            t = tickers[0]
            try:
                out[t] = float(scores[t])
            except Exception:
                continue
            continue
        # Multiple tickers in same family -> pick vehicle
        # If adv column missing, pick max score then ticker id
        if not has_adv:
            # pick max score, tie ticker id
            best = sorted(tickers, key=lambda tk: (-float(scores.get(tk, float("-inf"))), str(tk)))[0]
            try:
                out[best] = float(scores[best])
            except Exception:
                continue
            continue
        # adv column exists: consider finite adv values
        finite_cands: list[tuple[str, float, float]] = []
        for tk in tickers:
            row = row_by_ticker.get(tk)
            adv_val: float | None = None
            if row is not None:
                raw = row.get(adv_col)
                if raw is not None:
                    try:
                        fv = float(raw)
                        if math.isfinite(fv):
                            adv_val = float(fv)
                    except Exception:
                        adv_val = None
            if adv_val is not None:
                try:
                    sc = float(scores[tk])
                except Exception:
                    sc = float("-inf")
                if math.isfinite(sc):
                    finite_cands.append((tk, float(adv_val), float(sc)))
                else:
                    finite_cands.append((tk, float(adv_val), float("-inf")))
        if finite_cands:
            # vehicle = max finite adv (tie: max score, then ticker id)
            # sort by (-adv, -score, ticker)
            finite_cands_sorted = sorted(finite_cands, key=lambda x: (-x[1], -x[2], x[0]))
            winner = finite_cands_sorted[0][0]
            try:
                out[winner] = float(scores[winner])
            except Exception:
                continue
        else:
            # all adv NaN/non-finite -> pick max score then ticker id
            best = sorted(tickers, key=lambda tk: (-float(scores.get(tk, float("-inf"))), str(tk)))[0]
            try:
                out[best] = float(scores[best])
            except Exception:
                continue
    return out
def filter_plus2_scores(snapshot: pl.DataFrame, config: StickyLeaderConfig) -> dict[str, float]:
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return {}
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return {}
    except Exception:
        return {}
    if config.mom_col not in snapshot.columns:
        # mom_col missing -> empty per fail-closed iterate would skip all
        # Instead return {} directly
        return {}
    if "ticker" not in snapshot.columns:
        return {}
    out: dict[str, float] = {}
    # Check if name column exists; if not, treat name as empty -> skip all
    has_name = "name" in snapshot.columns
    for row in snapshot.iter_rows(named=True):
        try:
            ticker = str(row.get("ticker"))
        except Exception:
            continue
        v = row.get(config.mom_col)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        name = ""
        if has_name:
            try:
                nv = row.get("name")
                if nv is not None:
                    name = str(nv)
            except Exception:
                name = ""
        if not name:
            continue
        try:
            _tokens = tuple(getattr(config, "exclude_name_tokens", ()) or ())
        except Exception:
            _tokens = ()
        if _tokens and name_excluded(name, _tokens):
            continue
        try:
            _excl_synth = bool(getattr(config, "exclude_synthetic", False))
        except Exception:
            _excl_synth = False
        if _excl_synth and "(합성" in str(name):
            continue
        try:
            lev, _conf = resolve_leverage(name)
        except Exception:
            continue
        if config.only_plus_2 and lev != 2:
            continue
        if config.no_inverse and lev < 0:
            continue
        out[ticker] = float(fv)
    return out
def apply_sticky_leader(
    scores: Mapping[str, float], held: str | None, config: StickyLeaderConfig, hold_len: int
) -> dict[str, float]:
    if not scores:
        return {}
    # normalize config
    try:
        mg = float(config.min_gap)
        if not math.isfinite(mg) or mg < 0:
            mg = 0.0
    except Exception:
        mg = 0.0
    try:
        mh = int(config.min_hold)
        if mh < 0:
            mh = 0
        # check finiteness via float
        if not math.isfinite(float(mh)):
            mh = 0
    except Exception:
        mh = 0
    # copy
    out = dict(scores)
    # determine top
    try:
        sorted_items = sorted(out.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        top_ticker = str(sorted_items[0][0])
        top_score = float(sorted_items[0][1])
    except Exception:
        return out
    if held is None or held not in out:
        return out
    try:
        hl = int(hold_len)
    except Exception:
        hl = 0
    # stay condition
    try:
        held_score = float(out[held])
    except Exception:
        return out
    stay = False
    if hl < mh:
        stay = True
    elif held_score + mg >= top_score - 1e-12:
        stay = True
    if stay:
        max_sc = max(held_score, top_score)
        out[held] = float(max_sc) + 1e-6
    return out
class StickyLeaderModel:
    name: str
    config: StickyLeaderConfig
    path_dependent: bool = True
    scores_path_independent: bool = False
    def __init__(self, name: str = "P20", config: StickyLeaderConfig | None = None) -> None:
        self.name = str(name); self.config = config if config is not None else StickyLeaderConfig(); self._held: str | None = None; self._hold_len: int = 0; self._runner_ticker: str | None = None; self._runner_entry_capital: float | None = None; self._runner_peak_capital: float | None = None; self._runner_held_sessions: int = 0; self._runner_armed: bool = False; self._filtered_scores_by_snapshot: dict[int, dict[str, float]] = {}
    def reset_trackers(self) -> None:
        self._held = None; self._hold_len = 0; self._runner_ticker = None; self._runner_entry_capital = None; self._runner_peak_capital = None; self._runner_held_sessions = 0; self._runner_armed = False
    def restore_state(self, held: str | None, hold_len: int) -> None:
        import math as _math
        if held is not None and not isinstance(held, str):
            raise ValueError("held must be str or None")
        try:
            hl = int(hold_len)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"hold_len invalid: {exc}") from exc
        # check finite via float
        try:
            fv = float(hold_len)  # type: ignore[arg-type]
            if not _math.isfinite(fv):
                raise ValueError("hold_len must be finite")
        except Exception as exc:
            raise ValueError(f"hold_len invalid: {exc}") from exc
        if hl < 0:
            raise ValueError("hold_len must be >=0")
        self._held = held
        self._hold_len = int(hl)
    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | object:
        filtered = cached_filtered_scores(self._filtered_scores_by_snapshot, snapshot, lambda frame: filter_plus2_scores(frame, self.config))
        try:
            _mfr = float(getattr(self.config, "min_fill_ratio", 0.0) or 0.0)
        except Exception:
            _mfr = 0.0
        if math.isfinite(_mfr) and _mfr > 0 and filtered:
            try:
                _cap = float(getattr(context, "capital", float("nan")))
            except Exception:
                _cap = float("nan")
            try:
                _rules = getattr(context, "rules", None)
                _phi = float(getattr(_rules, "max_order_to_adv", 0.01))
            except Exception:
                _phi = 0.01
            if not math.isfinite(_phi) or _phi <= 0:
                _phi = 0.01
            filtered = apply_capacity_filter(
                filtered, snapshot, capital=_cap, max_order_to_adv=_phi, min_fill_ratio=_mfr
            )
            if not filtered:
                from src.portfolio.intent import CASH_INTENT as _CASH_CAP
                return _CASH_CAP
        try:
            _aux_col = getattr(self.config, "score_aux_col", None)
            _aux_w = float(getattr(self.config, "score_aux_weight", 0.0) or 0.0)
        except Exception:
            _aux_col, _aux_w = None, 0.0
        if isinstance(_aux_col, str) and _aux_col and math.isfinite(_aux_w) and _aux_w > 0:
            if _aux_col not in snapshot.columns:
                filtered = cross_section_percentile_ranks(filtered)
            else:
                _aux_raw = {}
                try:
                    rows = snapshot.iter_rows(named=True)
                except Exception:
                    return {}
                for _row in rows:
                    try:
                        _t = str(_row.get("ticker"))
                        _af = float(_row.get(_aux_col))  # type: ignore[arg-type]
                        if _t in filtered and math.isfinite(_af):
                            _aux_raw[_t] = float(_af)
                    except Exception:
                        continue
                filtered = blend_rank_scores(filtered, _aux_raw, w_primary=1.0 - _aux_w, w_aux=_aux_w)
            if not filtered:
                from src.portfolio.intent import CASH_INTENT as _CASH_AUX
                return _CASH_AUX
        if getattr(self.config, "collapse_family", False):
            try:
                filtered = collapse_plus2_by_family(filtered, snapshot)
            except Exception:
                pass
        # derive held from context.held by (-weight, ticker)
        held: str | None = None
        try:
            held_map = getattr(context, "held", {})
            if isinstance(held_map, Mapping) and len(held_map) > 0:
                # filter to numeric weights?
                items = []
                for k, v in held_map.items():
                    try:
                        items.append((str(k), float(v)))
                    except Exception:
                        continue
                if items:
                    items_sorted = sorted(items, key=lambda kv: (-kv[1], kv[0]))
                    held = str(items_sorted[0][0])
        except Exception:
            held = None
        # update internal hold_len
        if held != self._held:
            self._held = held
            self._hold_len = 1 if held else 0
        elif held is not None:
            self._hold_len += 1
        # else held is None and _held is None -> keep 0
        # P33 confirmed-runner-reversal tracker (model-local, reversible cash exit only)
        _runner_exit = bool(getattr(self.config, "runner_reversal_exit", False)); _runner_cap: float | None = None; _runner_mom: float | None = None; _runner_mc = "mom_5"
        if _runner_exit:
            try: _c = float(getattr(context, "capital", float("nan"))); _runner_cap = float(_c) if math.isfinite(_c) and _c > 0 else None
            except Exception: _runner_cap = None
            try: _runner_mc = str(getattr(self.config, "runner_mom_col", "mom_5"))
            except Exception: _runner_mc = "mom_5"
            try: _hz = int(momentum_horizon(_runner_mc))
            except Exception: _hz = 5
            if held is None or _runner_cap is None or _hz <= 0:
                if held is None: self._runner_ticker = None; self._runner_entry_capital = None; self._runner_peak_capital = None; self._runner_held_sessions = 0
                self._runner_armed = False
            elif getattr(self, "_runner_ticker", None) != held: self._runner_ticker = str(held); self._runner_entry_capital = float(_runner_cap); self._runner_peak_capital = float(_runner_cap); self._runner_held_sessions = 1; self._runner_armed = False
            else:
                self._runner_held_sessions = int(getattr(self, "_runner_held_sessions", 0) or 0) + 1
                try: _pk = float(getattr(self, "_runner_peak_capital", _runner_cap)); _pk = float(_runner_cap) if not math.isfinite(_pk) else max(float(_pk), float(_runner_cap)); self._runner_peak_capital = float(_pk); _en = float(getattr(self, "_runner_entry_capital", float("nan"))); _pn = float(getattr(self, "_runner_peak_capital", float("nan"))); self._runner_armed = bool(math.isfinite(_en) and math.isfinite(_pn) and int(getattr(self, "_runner_held_sessions", 0)) >= _hz and _pn > _en)
                except Exception: self._runner_armed = False
            if held is not None:
                try:
                    if isinstance(snapshot, pl.DataFrame) and _runner_mc in snapshot.columns and "ticker" in snapshot.columns: _df = snapshot.filter(pl.col("ticker") == str(held)).head(1); _runner_mom = (lambda _v: (lambda _f: float(_f) if math.isfinite(float(_f)) else None)(float(_v) if _v is not None else float("nan")))(_df.row(0, named=True).get(_runner_mc)) if _df.height > 0 else None
                except Exception: _runner_mom = None
        if getattr(self.config, "collapse_family", False):
            try: _ns = len(filtered); _tt = sorted(filtered.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0] if filtered else ""; logger.debug(f"[ALGO] ticker={_tt} held={held} n_scores={_ns}")
            except Exception: pass
        sticky = apply_sticky_leader(filtered, held, self.config, self._hold_len)
        from src.strategies.sticky.overlays import (
            apply_abs_mom_cash,
            apply_crash_cash,
            apply_impulse_switch,
            apply_same_leader_hold,
        )
        impulsed = apply_impulse_switch(sticky, held, snapshot, self.config); crashed = apply_crash_cash(impulsed, held, snapshot, self.config); abs_gated = apply_abs_mom_cash(crashed, self.config, held=held); out = apply_same_leader_hold(abs_gated, held, bool(getattr(self.config, "same_leader_hold", False)))
        if _runner_exit and held is not None and _runner_cap is not None and _runner_mom is not None:
            try:
                _pf = float(getattr(self, "_runner_peak_capital", float("nan"))); _ef = float(getattr(self, "_runner_entry_capital", float("nan")))
                if bool(getattr(self, "_runner_armed", False)) and math.isfinite(_pf) and math.isfinite(_ef) and _pf > _ef and float(_runner_cap) < _pf and float(_runner_mom) <= 0:
                    from src.portfolio.intent import CASH_INTENT as _CASH_RUNNER; return _CASH_RUNNER
            except Exception: pass
        try:
            from collections.abc import Mapping as _Mapping
            from src.portfolio.intent import CASH_INTENT as _CASH_EMPTY
            if isinstance(out, _Mapping) and len(out) == 0:
                return _CASH_EMPTY
        except Exception:
            pass
        return out
