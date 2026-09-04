# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from src.strategies.ids import (
    CONVEX_LOTTERY_IMPULSE,
    ALPHA_FAMILY_INTENSITY,
    ALPHA_SECTOR_LEADERSHIP,
    BASELINE_BUY_HOLD,
    BASELINE_MOM20_EQUAL3,
    BASELINE_MOM20_MA_GATE,
    BASELINE_MOM20_TOP1,
    BASELINE_REGIME_GATED_THEME,
    BASELINE_THEME_MOMENTUM,
    PORTFOLIO_CONVEXITY_HOLD,
    PORTFOLIO_CONVEXITY_REBALANCE,
    PORTFOLIO_CONVEXITY_VARIANT,
    PORTFOLIO_LEADERSHIP_CONFIDENCE,
    PORTFOLIO_LEADERSHIP_POLICY,
    PORTFOLIO_LOTTERY_EXPOSURE,
    PORTFOLIO_LOTTERY_REBALANCE,
    PORTFOLIO_MOMENTUM_CONFIDENCE,
    PORTFOLIO_MOMENTUM_POLICY,
    PORTFOLIO_MOMENTUM_VEHICLE,
    PORTFOLIO_TAIL_CONCENTRATION,
    STICKY_EQUITY_MOM60,
    STICKY_EQUITY_MOM60_VOL,
    STICKY_FILLABLE_MOM60,
    STICKY_FAMILY_PEAK_LOCK,
    STICKY_HOUSE_MONEY,
    STICKY_IMPULSE_CRASH,
    STICKY_LEADER_BASE,
    STICKY_MOM60_ABS_CASH,
    STICKY_MOM60_CONCENTRATED,
    STICKY_MOM60_HOLD,
    STICKY_MOM60_PEAK_LOCK,
    STICKY_MOM60_RAW,
    STICKY_SPLIT_FILL_LOCK,
)

LEGACY_ALIASES: Final[Mapping[str, str]] = {
    "B0": BASELINE_BUY_HOLD,
    "B1": BASELINE_MOM20_TOP1,
    "B2": BASELINE_MOM20_EQUAL3,
    "B3": BASELINE_MOM20_MA_GATE,
    "B4": BASELINE_THEME_MOMENTUM,
    "B5": BASELINE_REGIME_GATED_THEME,
    "M07": ALPHA_SECTOR_LEADERSHIP,
    "M13": ALPHA_FAMILY_INTENSITY,
    "P08": PORTFOLIO_MOMENTUM_POLICY,
    "P10": PORTFOLIO_MOMENTUM_VEHICLE,
    "P11": PORTFOLIO_MOMENTUM_CONFIDENCE,
    "P12": PORTFOLIO_LEADERSHIP_POLICY,
    "P13": PORTFOLIO_LEADERSHIP_CONFIDENCE,
    "P14": PORTFOLIO_LOTTERY_EXPOSURE,
    "P15": PORTFOLIO_TAIL_CONCENTRATION,
    "P16": PORTFOLIO_CONVEXITY_HOLD,
    "P17": PORTFOLIO_CONVEXITY_REBALANCE,
    "P18": PORTFOLIO_CONVEXITY_VARIANT,
    "P19": PORTFOLIO_LOTTERY_REBALANCE,
    "P20": STICKY_LEADER_BASE,
    "P21": STICKY_IMPULSE_CRASH,
    "P22": STICKY_FAMILY_PEAK_LOCK,
    "P23": STICKY_SPLIT_FILL_LOCK,
    "P24": STICKY_MOM60_PEAK_LOCK,
    "P25": STICKY_HOUSE_MONEY,
    "P26": STICKY_MOM60_CONCENTRATED,
    "P27": STICKY_MOM60_RAW,
    "P28A": STICKY_MOM60_HOLD,
    "P28B": STICKY_MOM60_ABS_CASH,
    "P29": STICKY_EQUITY_MOM60,
    "P29V": STICKY_EQUITY_MOM60_VOL,
    "P30": STICKY_FILLABLE_MOM60,
    "P31": CONVEX_LOTTERY_IMPULSE,
}

SEMANTIC_ALIASES: Final[Mapping[str, str]] = {v: k for k, v in LEGACY_ALIASES.items()}

_ALL_SEMANTIC: Final[frozenset[str]] = frozenset(LEGACY_ALIASES.values())
_LEGACY_UPPER: Final[Mapping[str, str]] = {k.upper(): v for k, v in LEGACY_ALIASES.items()}


def resolve_strategy_id(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"unknown strategy key: {key!r}")
    k = key.strip()
    if k in _ALL_SEMANTIC:
        return k
    kl = k.lower()
    for sem in _ALL_SEMANTIC:
        if sem.lower() == kl:
            return sem
    uk = k.upper()
    if uk in _LEGACY_UPPER:
        return _LEGACY_UPPER[uk]
    raise ValueError(f"unknown strategy key: {key!r}")


def branch_model_key(key: str) -> str:
    semantic = resolve_strategy_id(key)
    return SEMANTIC_ALIASES.get(semantic, key)


def build_strategy_registry() -> Mapping[str, Callable[[], object]]:
    from src.alpha.baselines import (
        _make_b0,
        _make_b1,
        _make_b2,
        _make_b3,
        _make_b4,
        _make_b5,
        _make_m07,
        _make_m13,
        _make_p08,
        _make_p10,
        _make_p11,
        _make_p12,
        _make_p13,
        _make_p14,
        _make_p15,
        _make_p16,
        _make_p17,
        _make_p18,
        _make_p19,
        _make_p20,
        _make_p21,
        _make_p22,
        _make_p23,
        _make_p24,
        _make_p25,
        _make_p26,
        _make_p27,
        _make_p28a,
        _make_p28b,
        _make_p29,
        _make_p29v,
        _make_p30,
        _make_p31,
    )
    from src.strategies.sticky.factories import FACTORY_REGISTRY

    _ = FACTORY_REGISTRY
    _ = STICKY_EQUITY_MOM60
    _ = STICKY_EQUITY_MOM60_VOL
    _ = STICKY_FILLABLE_MOM60
    _ = CONVEX_LOTTERY_IMPULSE
    _ = LEGACY_ALIASES["P31"]

    raw: Mapping[str, Callable[[], object]] = {
        BASELINE_BUY_HOLD: _make_b0,
        BASELINE_MOM20_TOP1: _make_b1,
        BASELINE_MOM20_EQUAL3: _make_b2,
        BASELINE_MOM20_MA_GATE: _make_b3,
        BASELINE_THEME_MOMENTUM: _make_b4,
        BASELINE_REGIME_GATED_THEME: _make_b5,
        ALPHA_SECTOR_LEADERSHIP: _make_m07,
        ALPHA_FAMILY_INTENSITY: _make_m13,
        PORTFOLIO_MOMENTUM_POLICY: _make_p08,
        PORTFOLIO_MOMENTUM_VEHICLE: _make_p10,
        PORTFOLIO_MOMENTUM_CONFIDENCE: _make_p11,
        PORTFOLIO_LEADERSHIP_POLICY: _make_p12,
        PORTFOLIO_LEADERSHIP_CONFIDENCE: _make_p13,
        PORTFOLIO_LOTTERY_EXPOSURE: _make_p14,
        PORTFOLIO_TAIL_CONCENTRATION: _make_p15,
        PORTFOLIO_CONVEXITY_HOLD: _make_p16,
        PORTFOLIO_CONVEXITY_REBALANCE: _make_p17,
        PORTFOLIO_CONVEXITY_VARIANT: _make_p18,
        PORTFOLIO_LOTTERY_REBALANCE: _make_p19,
        STICKY_LEADER_BASE: _make_p20,
        STICKY_IMPULSE_CRASH: _make_p21,
        STICKY_FAMILY_PEAK_LOCK: _make_p22,
        STICKY_SPLIT_FILL_LOCK: _make_p23,
        STICKY_MOM60_PEAK_LOCK: _make_p24,
        STICKY_HOUSE_MONEY: _make_p25,
        STICKY_MOM60_CONCENTRATED: _make_p26,
        STICKY_MOM60_RAW: _make_p27,
        STICKY_MOM60_HOLD: _make_p28a,
        STICKY_MOM60_ABS_CASH: _make_p28b,
        CONVEX_LOTTERY_IMPULSE: _make_p31,
    }

    def _wrap(semantic: str, factory: Callable[[], object]) -> Callable[[], object]:
        def _factory() -> object:
            obj = factory()
            try:
                setattr(obj, "name", semantic)
            except Exception:
                pass
            return obj

        return _factory

    registry = {k: _wrap(k, v) for k, v in raw.items()}
    # P29/P29V keep legacy names from their factories (contract: BASELINES["P29"]().name == "P29").
    registry[STICKY_EQUITY_MOM60] = _make_p29
    registry[STICKY_EQUITY_MOM60_VOL] = _make_p29v
    registry[STICKY_FILLABLE_MOM60] = _make_p30
    # Champion path (champion_path_logic contract): P26/P27/P28A/P28B keep legacy
    # names from their factories (BASELINES["P27"]().name == "P27").
    registry[STICKY_MOM60_CONCENTRATED] = _make_p26
    registry[STICKY_MOM60_RAW] = _make_p27
    registry[STICKY_MOM60_HOLD] = _make_p28a
    registry[STICKY_MOM60_ABS_CASH] = _make_p28b
    registry[CONVEX_LOTTERY_IMPULSE] = _make_p31
    _ = "P30"
    _ = "P31"
    return registry


STRATEGIES: Final[Mapping[str, Callable[[], object]]] = build_strategy_registry()
