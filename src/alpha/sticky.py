# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from src.alpha.base import DecisionContext
from src.universe.instruments import resolve_leverage


@dataclass
class StickyLeaderConfig:
    mom_col: str = "mom_20"
    only_plus_2: bool = True
    no_inverse: bool = True
    min_gap: float = 0.08
    min_hold: int = 3

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
        return cls(
            mom_col=str(mom_col),
            only_plus_2=bool(only_plus_2),
            no_inverse=bool(no_inverse),
            min_gap=float(min_gap),
            min_hold=int(min_hold),
        )


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
        return apply_sticky_leader(filtered, held, self.config, self._hold_len)
