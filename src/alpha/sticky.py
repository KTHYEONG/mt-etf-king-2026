# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from src.alpha.base import DecisionContext
from src.universe.instruments import resolve_leverage

logger = logging.getLogger(__name__)


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
        )


def resolve_lock_level(value: object, *, default: float = 0.50) -> float:
    try:
        ll = float(value)  # type: ignore[arg-type]
        if not math.isfinite(ll) or ll < 0:
            return float(default)
        return float(ll)
    except Exception:
        return float(default)


def load_p22_lock_level(*, default: float = 0.50) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p22 = port.get("p22")
        if not isinstance(p22, Mapping) or "lock_level" not in p22:
            return float(default)
        return resolve_lock_level(p22["lock_level"], default=default)
    except Exception:
        return float(default)


def load_p24_lock_level(*, default: float = 0.50) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p24 = port.get("p24")
        if not isinstance(p24, Mapping) or "lock_level" not in p24:
            return float(default)
        return resolve_lock_level(p24["lock_level"], default=default)
    except Exception:
        return float(default)


def load_p24_trail(*, default: float = 0.0) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p24 = port.get("p24")
        if not isinstance(p24, Mapping) or "trail" not in p24:
            # also support trail_level alias
            if isinstance(p24, Mapping) and "trail_level" in p24:
                v = p24["trail_level"]
                try:
                    fv = float(v)  # type: ignore[arg-type]
                    if not math.isfinite(fv) or fv < 0:
                        return float(default)
                    return float(fv)
                except Exception:
                    return float(default)
            return float(default)
        v = p24["trail"]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv) or fv < 0:
                return float(default)
            return float(fv)
        except Exception:
            return float(default)
    except Exception:
        return float(default)


def load_p24_mom_col(*, default: str = "mom_60") -> str:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return str(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return str(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return str(default)
        p24 = port.get("p24")
        if not isinstance(p24, Mapping) or "mom_col" not in p24:
            return str(default)
        v = p24["mom_col"]
        if not isinstance(v, str) or not v.strip():
            return str(default)
        s = str(v).strip()
        # allow only mom_ prefixed? fallback for invalid names still allow but require non-empty
        if not s:
            return str(default)
        try:
            # if value is numeric-like string that is nan/inf, treat as invalid
            fv = float(s)  # type: ignore[arg-type]
            if math.isfinite(fv):
                # if s is numeric it is invalid for mom_col
                return str(default)
        except Exception:
            pass
        # validate mom_col pattern: startswith mom_
        if not s.startswith("mom_"):
            return str(default)
        return str(s)
    except Exception:
        return str(default)


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


def apply_impulse_switch(
    scores: Mapping[str, float], held: str | None, snapshot: pl.DataFrame, config: StickyLeaderConfig
) -> dict[str, float]:
    if not scores:
        return {}
    try:
        gap = float(config.impulse_gap)
        if not math.isfinite(gap) or gap <= 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if held is None or held not in scores:
        return dict(scores)
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return dict(scores)
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if config.impulse_col not in snapshot.columns:
        return dict(scores)
    if "ticker" not in snapshot.columns:
        return dict(scores)
    # build row map for relevant tickers
    row_by_ticker: dict[str, dict] = {}
    try:
        for row in snapshot.iter_rows(named=True):
            try:
                t = str(row.get("ticker"))
            except Exception:
                continue
            if t in scores or t == held:
                row_by_ticker[t] = row
    except Exception:
        return dict(scores)
    if held not in row_by_ticker:
        return dict(scores)
    held_row = row_by_ticker[held]
    try:
        held_imp_raw = held_row.get(config.impulse_col)
        held_imp = float(held_imp_raw) if held_imp_raw is not None else float("nan")
    except Exception:
        return dict(scores)
    if not math.isfinite(held_imp):
        return dict(scores)
    candidates = [t for t in scores.keys() if t != held]
    if not candidates:
        return dict(scores)

    def _mom_val(ticker: str) -> float:
        row = row_by_ticker.get(ticker)
        if row is not None and config.mom_col in row and row.get(config.mom_col) is not None:
            try:
                fv = float(row.get(config.mom_col))
                if math.isfinite(fv):
                    return float(fv)
            except Exception:
                pass
        try:
            return float(scores[ticker])
        except Exception:
            return float("-inf")

    candidates_sorted = sorted(candidates, key=lambda t: (-_mom_val(t), str(t)))
    for chal in candidates_sorted:
        row = row_by_ticker.get(chal)
        if row is None:
            continue
        chal_raw = row.get(config.impulse_col)
        try:
            chal_imp = float(chal_raw) if chal_raw is not None else float("nan")
        except Exception:
            continue
        if not math.isfinite(chal_imp):
            continue
        # mom_5 >0 requirement
        mom5_ok = False
        if "mom_5" in snapshot.columns:
            mv = row.get("mom_5")
            try:
                mfv = float(mv) if mv is not None else float("nan")
                if math.isfinite(mfv) and mfv > 0:
                    mom5_ok = True
            except Exception:
                mom5_ok = False
        else:
            # fallback: impulse value >0
            if chal_imp > 0:
                mom5_ok = True
        if not mom5_ok:
            continue
        if config.impulse_require_volx:
            vol_raw = row.get("volume_expansion")
            try:
                vfv = float(vol_raw) if vol_raw is not None else 0.0
                if not math.isfinite(vfv) or vfv <= 0:
                    continue
            except Exception:
                continue
        if chal_imp - held_imp >= float(gap) - 1e-12:
            out = dict(scores)
            try:
                held_score = float(out[held])
            except Exception:
                try:
                    held_score = float(max(out.values()))
                except Exception:
                    held_score = 0.0
            out[chal] = float(held_score) + 1e-6
            return out
    return dict(scores)


def apply_crash_cash(
    scores: Mapping[str, float], held: str | None, snapshot: pl.DataFrame, config: StickyLeaderConfig
) -> dict[str, float]:
    if not scores:
        return {}
    try:
        cd = float(config.cash_drawdown)
        if not math.isfinite(cd) or cd >= 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if held is None or held not in scores:
        return dict(scores)
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return dict(scores)
    try:
        if snapshot.height == 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if "drawdown_20" not in snapshot.columns or "ticker" not in snapshot.columns:
        return dict(scores)
    held_dd = None
    try:
        for row in snapshot.iter_rows(named=True):
            if str(row.get("ticker")) == held:
                held_dd = row.get("drawdown_20")
                break
    except Exception:
        return dict(scores)
    if held_dd is None:
        return dict(scores)
    try:
        hf = float(held_dd)
    except Exception:
        return dict(scores)
    if not math.isfinite(hf):
        return dict(scores)
    if hf < float(cd) - 1e-12:
        return {}
    return dict(scores)


class StickyLeaderModel:
    name: str
    config: StickyLeaderConfig

    def __init__(self, name: str = "P20", config: StickyLeaderConfig | None = None) -> None:
        self.name = str(name)
        self.config = config if config is not None else StickyLeaderConfig()
        self._held: str | None = None
        self._hold_len: int = 0

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        filtered = filter_plus2_scores(snapshot, self.config)
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
        if getattr(self.config, "collapse_family", False):
            try:
                n_scores = len(filtered)
                if filtered:
                    top_ticker = sorted(filtered.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0]
                else:
                    top_ticker = ""
                logger.debug(f"[ALGO] ticker={top_ticker} held={held} n_scores={n_scores}")
            except Exception:
                pass
        sticky = apply_sticky_leader(filtered, held, self.config, self._hold_len)
        impulsed = apply_impulse_switch(sticky, held, snapshot, self.config)
        crashed = apply_crash_cash(impulsed, held, snapshot, self.config)
        return crashed
