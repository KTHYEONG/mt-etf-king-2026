# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import polars as pl

from src.alpha.base import DecisionContext
from src.strategies.sticky.capacity import apply_capacity_filter
from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS, name_excluded
from src.universe.instruments import resolve_leverage

DEFAULT_BETA_FAMILY_KEYS: Final[tuple[str, ...]] = (
    "코스피 200",
    "코스피 200 선물지수",
    "코스닥 150",
    "F-코스닥150 지수",
    "KRX 300",
)


@dataclass
class ConvexImpulseConfig:
    impulse_min: float = 0.08
    volx_min: float = 1.2
    continuation_min: float = 0.20
    crash_drawdown: float = -0.12
    min_gap: float = 0.04
    min_hold: int = 2
    exclude_name_tokens: tuple[str, ...] = DEFAULT_EXCLUDE_NAME_TOKENS
    exclude_synthetic: bool = True
    min_fill_ratio: float = 0.25
    beta_family_keys: tuple[str, ...] = DEFAULT_BETA_FAMILY_KEYS

    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> ConvexImpulseConfig:
        defaults = cls()
        if not isinstance(raw, Mapping):
            return defaults
        try:
            impulse_min = float(raw.get("impulse_min", defaults.impulse_min))  # type: ignore[arg-type]
            if not math.isfinite(impulse_min):
                impulse_min = defaults.impulse_min
        except Exception:
            impulse_min = defaults.impulse_min
        try:
            volx_min = float(raw.get("volx_min", defaults.volx_min))  # type: ignore[arg-type]
            if not math.isfinite(volx_min):
                volx_min = defaults.volx_min
        except Exception:
            volx_min = defaults.volx_min
        try:
            continuation_min = float(raw.get("continuation_min", defaults.continuation_min))  # type: ignore[arg-type]
            if not math.isfinite(continuation_min):
                continuation_min = defaults.continuation_min
        except Exception:
            continuation_min = defaults.continuation_min
        try:
            crash_drawdown = float(raw.get("crash_drawdown", defaults.crash_drawdown))  # type: ignore[arg-type]
            if not math.isfinite(crash_drawdown):
                crash_drawdown = defaults.crash_drawdown
        except Exception:
            crash_drawdown = defaults.crash_drawdown
        try:
            min_gap = float(raw.get("min_gap", defaults.min_gap))  # type: ignore[arg-type]
            if not math.isfinite(min_gap) or min_gap < 0:
                min_gap = defaults.min_gap
        except Exception:
            min_gap = defaults.min_gap
        try:
            min_hold = int(raw.get("min_hold", defaults.min_hold))  # type: ignore[arg-type]
            if min_hold < 0:
                min_hold = defaults.min_hold
        except Exception:
            min_hold = defaults.min_hold
        try:
            _tokens = raw.get("exclude_name_tokens")
            if isinstance(_tokens, (list, tuple)):
                exclude_name_tokens = tuple(str(t) for t in _tokens if isinstance(t, str) and str(t))
            else:
                exclude_name_tokens = tuple(defaults.exclude_name_tokens)
        except Exception:
            exclude_name_tokens = tuple(defaults.exclude_name_tokens)
        try:
            exclude_synthetic = bool(raw.get("exclude_synthetic", defaults.exclude_synthetic))
        except Exception:
            exclude_synthetic = defaults.exclude_synthetic
        try:
            min_fill_ratio = float(raw.get("min_fill_ratio", defaults.min_fill_ratio))  # type: ignore[arg-type]
            if not math.isfinite(min_fill_ratio) or min_fill_ratio < 0:
                min_fill_ratio = defaults.min_fill_ratio
        except Exception:
            min_fill_ratio = defaults.min_fill_ratio
        try:
            _beta = raw.get("beta_family_keys")
            if isinstance(_beta, (list, tuple)) and any(isinstance(t, str) and str(t).strip() for t in _beta):
                extra = [str(t).strip() for t in _beta if isinstance(t, str) and str(t).strip()]
                merged = list(DEFAULT_BETA_FAMILY_KEYS)
                for key in extra:
                    if key not in merged:
                        merged.append(key)
                beta_family_keys = tuple(merged)
            else:
                beta_family_keys = tuple(defaults.beta_family_keys)
        except Exception:
            beta_family_keys = tuple(defaults.beta_family_keys)
        return cls(
            impulse_min=float(impulse_min),
            volx_min=float(volx_min),
            continuation_min=float(continuation_min),
            crash_drawdown=float(crash_drawdown),
            min_gap=float(min_gap),
            min_hold=int(min_hold),
            exclude_name_tokens=tuple(exclude_name_tokens),
            exclude_synthetic=bool(exclude_synthetic),
            min_fill_ratio=float(min_fill_ratio),
            beta_family_keys=tuple(beta_family_keys),
        )


def is_beta_family(family_key: str, beta_keys: tuple[str, ...] | None = None) -> bool:
    keys = DEFAULT_BETA_FAMILY_KEYS if beta_keys is None else tuple(beta_keys)
    if not isinstance(family_key, str):
        return False
    norm = family_key.strip()
    if not norm:
        return False
    for key in keys:
        if isinstance(key, str) and norm == key.strip():
            return True
    return False


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(fv):
        return None
    return float(fv)


def classify_setup(row: Mapping[str, object], config: ConvexImpulseConfig) -> str:
    mom_5 = _finite_float(row.get("mom_5")) if "mom_5" in row else None
    mom_10 = _finite_float(row.get("mom_10")) if "mom_10" in row else None
    mom_20 = _finite_float(row.get("mom_20")) if "mom_20" in row else None
    mom_60 = _finite_float(row.get("mom_60")) if "mom_60" in row else None
    volx = _finite_float(row.get("volume_expansion")) if "volume_expansion" in row else None
    if mom_5 is not None and mom_10 is not None and mom_20 is not None and volx is not None:
        if mom_5 > mom_20 and mom_10 >= float(config.impulse_min) and volx >= float(config.volx_min):
            return "impulse"
    if mom_60 is not None and mom_20 is not None:
        if mom_60 >= float(config.continuation_min) and mom_20 > 0:
            return "continuation"
    return "none"


def filter_convex_plus2_rows(snapshot: pl.DataFrame, config: ConvexImpulseConfig) -> list[dict[str, object]]:
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return []
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return []
    except Exception:
        return []
    if "ticker" not in snapshot.columns:
        return []
    has_name = "name" in snapshot.columns
    has_family = "underlying_index_name" in snapshot.columns
    out: list[dict[str, object]] = []
    for row in snapshot.iter_rows(named=True):
        try:
            ticker = str(row.get("ticker"))
        except Exception:
            continue
        if not ticker:
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
        if lev != 2:
            continue
        if lev < 0:
            continue
        if has_family:
            try:
                fam = row.get("underlying_index_name")
                fam_str = str(fam).strip() if fam is not None else ""
                try:
                    _beta_keys = tuple(getattr(config, "beta_family_keys", DEFAULT_BETA_FAMILY_KEYS))
                except Exception:
                    _beta_keys = DEFAULT_BETA_FAMILY_KEYS
                if fam_str and is_beta_family(fam_str, _beta_keys):
                    continue
            except Exception:
                pass
        out.append(dict(row))
    return out


def pick_convex_ticker(
    rows: Sequence[Mapping[str, object]],
    config: ConvexImpulseConfig,
    held: str | None,
    hold_len: int,
) -> str | None:
    try:
        seq = list(rows)
    except Exception:
        return None
    if not seq:
        return None
    impulse: list[tuple[str, float]] = []
    continuation: list[tuple[str, float]] = []
    setup_by_ticker: dict[str, str] = {}
    rank_by_ticker: dict[str, float] = {}
    for row in seq:
        try:
            ticker = str(row.get("ticker"))  # type: ignore[union-attr]
        except Exception:
            continue
        if not ticker:
            continue
        setup = classify_setup(row, config)
        if setup == "impulse":
            fv = _finite_float(row.get("mom_10"))
            if fv is None:
                continue
            impulse.append((ticker, float(fv)))
            setup_by_ticker[ticker] = setup
            rank_by_ticker[ticker] = float(fv)
        elif setup == "continuation":
            fv = _finite_float(row.get("mom_60"))
            if fv is None:
                continue
            continuation.append((ticker, float(fv)))
            setup_by_ticker[ticker] = setup
            rank_by_ticker[ticker] = float(fv)
    fresh: str | None = None
    fresh_mode = ""
    if impulse:
        fresh = sorted(impulse, key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0]
        fresh_mode = "impulse"
    elif continuation:
        fresh = sorted(continuation, key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0]
        fresh_mode = "continuation"
    if fresh is None:
        return None
    if held is None or held not in setup_by_ticker:
        return fresh
    if held == fresh:
        return held
    try:
        mg = float(config.min_gap)
        if not math.isfinite(mg) or mg < 0:
            mg = 0.0
    except Exception:
        mg = 0.0
    try:
        mh = int(hold_len)
        if mh < 0:
            mh = 0
    except Exception:
        mh = 0
    try:
        mm = int(config.min_hold)
        if mm < 0:
            mm = 0
    except Exception:
        mm = 0
    if mh < mm:
        return held
    if fresh_mode == "impulse":
        col = "mom_10"
    else:
        col = "mom_60"
    held_val = rank_by_ticker.get(str(held))
    fresh_val = rank_by_ticker.get(str(fresh))
    if held_val is None or fresh_val is None:
        return fresh
    _ = col
    if float(fresh_val) - float(held_val) < float(mg) - 1e-12:
        return held
    return fresh


def overlay_should_cash(*args: object, **kwargs: object) -> bool:
    return False


class ConvexImpulseModel:
    name: str
    config: ConvexImpulseConfig
    path_dependent: bool = True
    scores_path_independent: bool = False

    def __init__(self, name: str = "P31", config: ConvexImpulseConfig | None = None) -> None:
        self.name = str(name)
        self.config = config if config is not None else ConvexImpulseConfig()
        self._held: str | None = None
        self._hold_len: int = 0

    def reset_trackers(self) -> None:
        self._held = None
        self._hold_len = 0

    def restore_state(self, held: str | None, hold_len: int) -> None:
        if held is not None and not isinstance(held, str):
            raise ValueError("held must be str or None")
        try:
            hl = int(hold_len)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"hold_len invalid: {exc}") from exc
        try:
            fv = float(hold_len)  # type: ignore[arg-type]
            if not math.isfinite(fv):
                raise ValueError("hold_len must be finite")
        except Exception as exc:
            raise ValueError(f"hold_len invalid: {exc}") from exc
        if hl < 0:
            raise ValueError("hold_len must be >=0")
        self._held = held
        self._hold_len = int(hl)

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | object:
        from src.portfolio.intent import CASH_INTENT as _CASH

        rows = filter_convex_plus2_rows(snapshot, self.config)
        if not rows:
            return _CASH
        try:
            _mfr = float(getattr(self.config, "min_fill_ratio", 0.0) or 0.0)
        except Exception:
            _mfr = 0.0
        eligible = list(rows)
        if math.isfinite(_mfr) and _mfr > 0 and eligible:
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
            seed = {str(r.get("ticker")): 1.0 for r in eligible if str(r.get("ticker"))}
            try:
                kept = apply_capacity_filter(
                    seed, snapshot, capital=_cap, max_order_to_adv=_phi, min_fill_ratio=_mfr, sleeve_weight=0.95
                )
            except Exception:
                kept = seed
            eligible = [r for r in eligible if str(r.get("ticker")) in kept]
            if not eligible:
                return _CASH
        held: str | None = None
        try:
            held_map = getattr(context, "held", {})
            if isinstance(held_map, Mapping) and len(held_map) > 0:
                items = []
                for k, v in held_map.items():
                    try:
                        items.append((str(k), float(v)))
                    except Exception:
                        continue
                if items:
                    held = str(sorted(items, key=lambda kv: (-kv[1], kv[0]))[0][0])
        except Exception:
            held = None
        if held != self._held:
            self._held = held
            self._hold_len = 1 if held else 0
        elif held is not None:
            self._hold_len += 1
        if held is not None:
            try:
                _cd = float(getattr(self.config, "crash_drawdown", -0.12))
            except Exception:
                _cd = -0.12
            if math.isfinite(_cd) and _cd < 0:
                try:
                    if "ticker" in snapshot.columns and "drawdown_20" in snapshot.columns:
                        for row in snapshot.iter_rows(named=True):
                            if str(row.get("ticker")) == held:
                                raw_dd = row.get("drawdown_20")
                                if raw_dd is not None:
                                    dd = float(raw_dd)  # type: ignore[arg-type]
                                    if math.isfinite(dd) and dd < float(_cd) - 1e-12:
                                        return _CASH
                                break
                except Exception:
                    pass
        if overlay_should_cash() is True:
            return _CASH
        gated: list[Mapping[str, object]] = []
        for row in eligible:
            if classify_setup(row, self.config) in ("impulse", "continuation"):
                gated.append(row)
        if not gated:
            return _CASH
        picked = pick_convex_ticker(gated, self.config, held, self._hold_len)
        if picked is None:
            return _CASH
        for row in gated:
            if str(row.get("ticker")) == picked:
                setup = classify_setup(row, self.config)
                col = "mom_10" if setup == "impulse" else "mom_60"
                val = _finite_float(row.get(col))
                return {str(picked): float(val) if val is not None else 1.0}
        return {str(picked): 1.0}
