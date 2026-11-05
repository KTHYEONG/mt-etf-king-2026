"""Pre-registered 자율형 candidates and reference baselines as causal target-weight decisions over the arena panel."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any, ClassVar, Final, Literal

import numpy as np
import polars as pl

from src.arena.panel import Panel
from src.arena.research import Strategy
from src.arena.signals import efficiency_ratio, flow_imbalance, rolling_z, trend_score, wide
from src.core.config import ConfigError, load_config

Holding = str | None
_CASH: Final = "CASH"
_KOSPI: Final = "kospi200"
_TICKER: Final = re.compile(r"[0-9A-Z]{6}")
_TARGET_SCHEMA: Final[dict[str, pl.DataType]] = {
    "decision_date": pl.Date(),
    "ticker": pl.String(),
    "weight": pl.Float64(),
}
_SEGMENT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "decision_date": pl.Date(),
    "leg": pl.String(),
    "ticker": pl.String(),
    "weight": pl.Float64(),
}
SegmentSignal = Literal["own", "vote3"]
SegmentOvernight = Literal["cash", "long"]
_SEGMENT_SIGNALS: Final[tuple[str, ...]] = ("own", "vote3")
_SEGMENT_OVERNIGHT: Final[tuple[str, ...]] = ("cash", "long")
_STRATEGY_KEYS: Final = frozenset(
    {
        "trend",
        "sleeves",
        "inverse",
        "min_listed_sessions",
        "trend_switch",
        "rotation",
        "flow_contrarian",
        "committee",
        "brake",
        "segment_fade",
        "baselines",
    }
)
_TREND_KEYS: Final = frozenset({"horizons", "vol_window", "band", "er_window", "er_min"})
_SWITCH_KEYS: Final = frozenset({"sleeves", "chop_filter"})
_ROTATION_KEYS: Final = frozenset({"hold"})
_FLOW_KEYS: Final = frozenset({"flow_window", "z_window", "z_threshold"})
_COMMITTEE_KEYS: Final = frozenset({"enabled"})
_BRAKE_KEYS: Final = frozenset({"down_streak"})
_SEGMENT_KEYS: Final = frozenset({"sleeves", "signal", "overnight", "vote_basket"})
_BASELINE_KEYS: Final = frozenset({"hold", "coin_pair", "coin_seed"})


@dataclass(frozen=True)
class StrategiesConfig:
    horizons: tuple[int, ...]
    vol_window: int
    band: float
    er_window: int
    er_min: float
    sleeves: Mapping[str, tuple[str, str]]
    inverse: Mapping[str, str]
    min_listed_sessions: int
    switch_sleeves: tuple[str, ...]
    chop_filter: tuple[bool, ...]
    rotation_hold: tuple[int, ...]
    flow_window: tuple[int, ...]
    z_window: int
    z_threshold: float
    committee: bool
    down_streak: int
    baseline_hold: tuple[str, ...]
    coin_pair: tuple[str, str]
    coin_seed: int
    segment_sleeves: tuple[str, ...]
    segment_signal: tuple[SegmentSignal, ...]
    segment_overnight: tuple[SegmentOvernight, ...]
    vote_basket: tuple[str, ...]


@dataclass(frozen=True)
class TrendSwitch:
    sleeve: str
    chop_filter: bool
    cfg: StrategiesConfig

    @property
    def family(self) -> str:
        return "trend_switch"

    @property
    def strategy_id(self) -> str:
        return f"trend_switch:{self.sleeve}:{'chop' if self.chop_filter else 'plain'}"

    @property
    def params(self) -> Mapping[str, Any]:
        signal, long = self.cfg.sleeves[self.sleeve]
        return {
            **_trend_params(self.cfg),
            "sleeve": self.sleeve,
            "signal": signal,
            "long": long,
            "inverse": self.cfg.inverse[self.sleeve],
            "chop_filter": self.chop_filter,
        }

    def holdings(self, panel: Panel) -> list[Holding]:
        signal, long = self.cfg.sleeves[self.sleeve]
        return _trend_holdings(panel, self.cfg, signal, long, self.cfg.inverse[self.sleeve], self.chop_filter)

    def targets(self, panel: Panel) -> pl.DataFrame:
        return _emit(panel.sessions, self.holdings(panel), self.cfg.sleeves[self.sleeve][1])


@dataclass(frozen=True)
class Rotation:
    hold: int
    cfg: StrategiesConfig

    @property
    def family(self) -> str:
        return "rotation"

    @property
    def strategy_id(self) -> str:
        return f"rotation:hold{self.hold}"

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            **_trend_params(self.cfg),
            "min_listed_sessions": self.cfg.min_listed_sessions,
            "hold": self.hold,
            "sleeves": {name: list(pair) for name, pair in sorted(self.cfg.sleeves.items())},
            "inverse": self.cfg.inverse[_KOSPI],
        }

    def targets(self, panel: Panel) -> pl.DataFrame:
        return _emit(panel.sessions, _rotation_holdings(panel, self.cfg, self.hold), self.cfg.sleeves[_KOSPI][1])


@dataclass(frozen=True)
class FlowContrarian:
    flow_window: int
    cfg: StrategiesConfig

    @property
    def family(self) -> str:
        return "flow_contrarian"

    @property
    def strategy_id(self) -> str:
        return f"flow_contrarian:w{self.flow_window}"

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            "flow_window": self.flow_window,
            "z_window": self.cfg.z_window,
            "z_threshold": self.cfg.z_threshold,
            "long": self.cfg.sleeves[_KOSPI][1],
            "inverse": self.cfg.inverse[_KOSPI],
        }

    def holdings(self, panel: Panel) -> list[Holding]:
        lev, inv = self.cfg.sleeves[_KOSPI][1], self.cfg.inverse[_KOSPI]
        shares = wide(panel, "shares_outstanding", [lev, inv])
        prices = wide(panel, "close_ref", [lev, inv])
        flow = flow_imbalance(
            shares.get_column(lev),
            prices.get_column(lev),
            shares.get_column(inv),
            prices.get_column(inv),
            self.flow_window,
        )
        threshold = self.cfg.z_threshold
        out: list[Holding] = []
        current: Holding = _CASH
        for value in _values(rolling_z(flow, self.cfg.z_window)):
            if math.isnan(value):
                out.append(None)
                continue
            if value >= threshold:
                current = inv
            elif value <= -threshold:
                current = lev
            out.append(current)
        return out

    def targets(self, panel: Panel) -> pl.DataFrame:
        return _emit(panel.sessions, self.holdings(panel), self.cfg.sleeves[_KOSPI][1])


@dataclass(frozen=True)
class Committee:
    cfg: StrategiesConfig

    @property
    def family(self) -> str:
        return "committee"

    @property
    def strategy_id(self) -> str:
        return "committee:trend+flow"

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            **_trend_params(self.cfg),
            "long": self.cfg.sleeves[_KOSPI][1],
            "inverse": self.cfg.inverse[_KOSPI],
            "flow_window": max(self.cfg.flow_window),
            "z_window": self.cfg.z_window,
            "z_threshold": self.cfg.z_threshold,
        }

    def holdings(self, panel: Panel) -> list[Holding]:
        long, inv = self.cfg.sleeves[_KOSPI][1], self.cfg.inverse[_KOSPI]
        trend = TrendSwitch(_KOSPI, False, self.cfg).holdings(panel)
        flow = FlowContrarian(max(self.cfg.flow_window), self.cfg).holdings(panel)
        out: list[Holding] = []
        for trend_hold, flow_hold in zip(trend, flow, strict=True):
            if trend_hold is None or flow_hold is None:
                out.append(None)
                continue
            votes = _vote(trend_hold, long, inv) + _vote(flow_hold, long, inv)
            out.append(long if votes > 0 else inv if votes < 0 else _CASH)
        return out

    def targets(self, panel: Panel) -> pl.DataFrame:
        return _emit(panel.sessions, self.holdings(panel), self.cfg.sleeves[_KOSPI][1])


@dataclass(frozen=True)
class Brake:
    cfg: StrategiesConfig

    @property
    def family(self) -> str:
        return "brake"

    @property
    def strategy_id(self) -> str:
        return f"brake:{_KOSPI}:down{self.cfg.down_streak}"

    @property
    def params(self) -> Mapping[str, Any]:
        signal, long = self.cfg.sleeves[_KOSPI]
        return {
            **_trend_params(self.cfg),
            "signal": signal,
            "long": long,
            "inverse": self.cfg.inverse[_KOSPI],
            "down_streak": self.cfg.down_streak,
        }

    def holdings(self, panel: Panel) -> list[Holding]:
        signal, long = self.cfg.sleeves[_KOSPI]
        trend = _trend_holdings(panel, self.cfg, signal, long, self.cfg.inverse[_KOSPI], False)
        streak = _down_streak(_values(_series(panel, "r_ref", signal)), self.cfg.down_streak)
        return [_CASH if hold == long and down else hold for hold, down in zip(trend, streak, strict=True)]

    def targets(self, panel: Panel) -> pl.DataFrame:
        return _emit(panel.sessions, self.holdings(panel), self.cfg.sleeves[_KOSPI][1])


@dataclass(frozen=True)
class SegmentFade:
    sleeve: str
    signal: SegmentSignal
    overnight: SegmentOvernight
    cfg: StrategiesConfig
    execution_model: ClassVar[Literal["segment"]] = "segment"

    @property
    def family(self) -> str:
        return "segment_fade"

    @property
    def strategy_id(self) -> str:
        return f"segment_fade:{self.sleeve}:{self.signal}:{self.overnight}"

    @property
    def params(self) -> Mapping[str, Any]:
        signal, long = self.cfg.sleeves[self.sleeve]
        out: dict[str, Any] = {
            "sleeve": self.sleeve,
            "signal_ticker": signal,
            "long": long,
            "inverse": self.cfg.inverse[self.sleeve],
            "signal": self.signal,
            "overnight": self.overnight,
            "min_listed_sessions": self.cfg.min_listed_sessions,
        }
        if self.signal == "vote3":
            out["vote_basket"] = list(self.cfg.vote_basket)
        return out

    def targets(self, panel: Panel) -> pl.DataFrame:
        signal, long = self.cfg.sleeves[self.sleeve]
        inverse = self.cfg.inverse[self.sleeve]
        members = (signal, *self.cfg.vote_basket) if self.signal == "vote3" else (signal,)
        returns = {ticker: _values(_series(panel, "r_ref", ticker)) for ticker in members}
        listed = {ticker: _listed_sessions(_series(panel, "listed", ticker)) for ticker in (long, inverse)}
        minimum = self.cfg.min_listed_sessions
        days: list[date] = []
        legs: list[str] = []
        tickers: list[str] = []
        weights: list[float] = []
        for day_index, day in enumerate(panel.sessions):
            signs = [_sign(returns[ticker][day_index]) for ticker in members]
            defined = (
                all(value is not None for value in signs)
                and listed[long][day_index] >= minimum
                and listed[inverse][day_index] >= minimum
            )
            side = _sign(sum(value for value in signs if value is not None)) if defined else None
            if side is not None and side > 0:
                intraday: Holding = inverse
            elif side is not None and side < 0:
                intraday = long
            else:
                intraday = None
            days.extend([day, day])
            legs.extend(["intraday", "overnight"])
            tickers.extend([inverse if intraday == inverse else long, long])
            weights.extend([0.0 if intraday is None else 1.0, 1.0 if defined and self.overnight == "long" else 0.0])
        return pl.DataFrame(
            {"decision_date": days, "leg": legs, "ticker": tickers, "weight": weights}, schema=_SEGMENT_SCHEMA
        )


@dataclass(frozen=True)
class HoldLong:
    ticker: str
    min_listed_sessions: int

    @property
    def family(self) -> str:
        return "baseline_hold"

    @property
    def strategy_id(self) -> str:
        return f"hold:{self.ticker}"

    @property
    def params(self) -> Mapping[str, Any]:
        return {"ticker": self.ticker, "min_listed_sessions": self.min_listed_sessions}

    def targets(self, panel: Panel) -> pl.DataFrame:
        listed = _listed_sessions(_series(panel, "listed", self.ticker))
        holdings: list[Holding] = [self.ticker if count >= self.min_listed_sessions else None for count in listed]
        return _emit(panel.sessions, holdings, self.ticker)


@dataclass(frozen=True)
class CoinFlip:
    long: str
    short: str
    seed: int
    min_listed_sessions: int

    @property
    def family(self) -> str:
        return "baseline_coin"

    @property
    def strategy_id(self) -> str:
        return f"coin:{self.long}-{self.short}:{self.seed}"

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            "long": self.long,
            "short": self.short,
            "seed": self.seed,
            "min_listed_sessions": self.min_listed_sessions,
        }

    def targets(self, panel: Panel) -> pl.DataFrame:
        long_ready = _listed_sessions(_series(panel, "listed", self.long))
        short_ready = _listed_sessions(_series(panel, "listed", self.short))
        rng = np.random.default_rng(self.seed)
        holdings: list[Holding] = []
        for long_count, short_count in zip(long_ready, short_ready, strict=True):
            if long_count < self.min_listed_sessions or short_count < self.min_listed_sessions:
                holdings.append(None)
                continue
            holdings.append(self.long if rng.random() < 0.5 else self.short)
        return _emit(panel.sessions, holdings, self.long)


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{where} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise ConfigError(f"{where} must not repeat entries")
    return tuple(value)


def _ticker(value: Any, where: str) -> str:
    if not isinstance(value, str) or _TICKER.fullmatch(value) is None:
        raise ConfigError(f"{where} must be a 6-character ticker string")
    return value


def _tickers(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a non-empty list of tickers")
    return tuple(_ticker(item, where) for item in value)


def _pair(value: Any, where: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{where} must be a [ticker, ticker] pair")
    return _ticker(value[0], where), _ticker(value[1], where)


def _integer(value: Any, where: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _int_list(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a non-empty list of integers")
    items = tuple(_integer(item, where, 1) for item in value)
    if len(set(items)) != len(items):
        raise ConfigError(f"{where} must not repeat entries")
    return items


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    return float(value)


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a boolean")
    return value


def _bool_list(value: Any, where: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a non-empty list of booleans")
    items = tuple(_boolean(item, where) for item in value)
    if len(set(items)) != len(items):
        raise ConfigError(f"{where} must not repeat entries")
    return items


def _choices(value: Any, allowed: tuple[str, ...], where: str) -> tuple[Any, ...]:
    items = _strings(value, where)
    if not set(items) <= set(allowed):
        raise ConfigError(f"{where} values must be in {list(allowed)}")
    return items


def _exact(value: Any, keys: frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigError(f"{where} keys must be exactly {sorted(keys)}")
    return value


def load_strategies_config(name: str = "arena") -> StrategiesConfig:
    """Load and validate the `strategies` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("strategies")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a strategies mapping")
    root = _exact(section, _STRATEGY_KEYS, "strategies")

    trend = _exact(root["trend"], _TREND_KEYS, "strategies.trend")
    horizons = _int_list(trend["horizons"], "trend.horizons")
    if any(later <= earlier for earlier, later in pairwise(horizons)):
        raise ConfigError("trend.horizons must be strictly increasing")
    band = _number(trend["band"], "trend.band")
    if band < 0.0:
        raise ConfigError("trend.band must be >= 0")
    er_min = _number(trend["er_min"], "trend.er_min")
    if not 0.0 < er_min < 1.0:
        raise ConfigError("trend.er_min must be in (0, 1)")

    sleeves_raw = root["sleeves"]
    if not isinstance(sleeves_raw, Mapping) or not sleeves_raw:
        raise ConfigError("strategies.sleeves must be a non-empty mapping")
    sleeves = {_name(key, "sleeves"): _pair(pair, f"sleeves.{key}") for key, pair in sleeves_raw.items()}
    inverse_raw = root["inverse"]
    if not isinstance(inverse_raw, Mapping):
        raise ConfigError("strategies.inverse must be a mapping")
    inverse = {_name(key, "inverse"): _ticker(ticker, f"inverse.{key}") for key, ticker in inverse_raw.items()}
    if not set(inverse) <= set(sleeves):
        raise ConfigError("strategies.inverse keys must be sleeves")
    if _KOSPI not in sleeves or _KOSPI not in inverse:
        raise ConfigError(f"strategies.sleeves and strategies.inverse must both define {_KOSPI}")

    switch = _exact(root["trend_switch"], _SWITCH_KEYS, "strategies.trend_switch")
    switch_sleeves = _strings(switch["sleeves"], "trend_switch.sleeves")
    for sleeve in switch_sleeves:
        if sleeve not in sleeves or sleeve not in inverse:
            raise ConfigError(f"trend_switch sleeve {sleeve} must be in sleeves and inverse")

    rotation = _exact(root["rotation"], _ROTATION_KEYS, "strategies.rotation")
    flow = _exact(root["flow_contrarian"], _FLOW_KEYS, "strategies.flow_contrarian")
    committee = _exact(root["committee"], _COMMITTEE_KEYS, "strategies.committee")
    brake = _exact(root["brake"], _BRAKE_KEYS, "strategies.brake")
    baselines = _exact(root["baselines"], _BASELINE_KEYS, "strategies.baselines")
    segment = _exact(root["segment_fade"], _SEGMENT_KEYS, "strategies.segment_fade")
    segment_sleeves = _strings(segment["sleeves"], "segment_fade.sleeves")
    for sleeve in segment_sleeves:
        if sleeve not in sleeves or sleeve not in inverse:
            raise ConfigError(f"segment_fade sleeve {sleeve} must be in sleeves and inverse")
    vote_basket = _tickers(segment["vote_basket"], "segment_fade.vote_basket")
    if len(set(vote_basket)) != len(vote_basket):
        raise ConfigError("segment_fade.vote_basket must not repeat entries")
    if (1 + len(vote_basket)) % 2 == 0:
        raise ConfigError("segment_fade vote count (1 + vote_basket) must be odd")

    return StrategiesConfig(
        horizons=horizons,
        vol_window=_integer(trend["vol_window"], "trend.vol_window", 2),
        band=band,
        er_window=_integer(trend["er_window"], "trend.er_window", 1),
        er_min=er_min,
        sleeves=sleeves,
        inverse=inverse,
        min_listed_sessions=_integer(root["min_listed_sessions"], "min_listed_sessions", 1),
        switch_sleeves=switch_sleeves,
        chop_filter=_bool_list(switch["chop_filter"], "trend_switch.chop_filter"),
        rotation_hold=_int_list(rotation["hold"], "rotation.hold"),
        flow_window=_int_list(flow["flow_window"], "flow_contrarian.flow_window"),
        z_window=_integer(flow["z_window"], "flow_contrarian.z_window", 2),
        z_threshold=_positive(flow["z_threshold"], "flow_contrarian.z_threshold"),
        committee=_boolean(committee["enabled"], "committee.enabled"),
        down_streak=_integer(brake["down_streak"], "brake.down_streak", 1),
        baseline_hold=_tickers(baselines["hold"], "baselines.hold"),
        coin_pair=_pair(baselines["coin_pair"], "baselines.coin_pair"),
        coin_seed=_integer(baselines["coin_seed"], "baselines.coin_seed", 0),
        segment_sleeves=segment_sleeves,
        segment_signal=_choices(segment["signal"], _SEGMENT_SIGNALS, "segment_fade.signal"),
        segment_overnight=_choices(segment["overnight"], _SEGMENT_OVERNIGHT, "segment_fade.overnight"),
        vote_basket=vote_basket,
    )


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"strategies.{where} keys must be non-empty strings")
    return value


def _positive(value: Any, where: str) -> float:
    number = _number(value, where)
    if number <= 0.0:
        raise ConfigError(f"{where} must be > 0")
    return number


def candidate_grid(cfg: StrategiesConfig) -> tuple[Strategy, ...]:
    """Return the pre-registered candidates in fixed family order; the count follows the config grid."""
    grid: list[Strategy] = [TrendSwitch(sleeve, chop, cfg) for sleeve in cfg.switch_sleeves for chop in cfg.chop_filter]
    grid.extend(Rotation(hold, cfg) for hold in cfg.rotation_hold)
    grid.extend(FlowContrarian(window, cfg) for window in cfg.flow_window)
    if cfg.committee:
        grid.append(Committee(cfg))
    grid.append(Brake(cfg))
    grid.extend(
        SegmentFade(sleeve, signal, overnight, cfg)
        for sleeve in cfg.segment_sleeves
        for signal in cfg.segment_signal
        for overnight in cfg.segment_overnight
    )
    return tuple(grid)


def baselines(cfg: StrategiesConfig) -> tuple[Strategy, ...]:
    """Return the reference baselines; they are reported on their own track and never selected."""
    holds: list[Strategy] = [HoldLong(ticker, cfg.min_listed_sessions) for ticker in cfg.baseline_hold]
    long, short = cfg.coin_pair
    return (*holds, CoinFlip(long, short, cfg.coin_seed, cfg.min_listed_sessions))


def strategy_tickers(strategies: Sequence[Strategy]) -> frozenset[str]:
    """Return every ticker a strategy reads for signals or trades; raises TypeError for an unknown strategy type."""
    found: set[str] = set()
    for strategy in strategies:
        found |= _tickers_of(strategy)
    return frozenset(found)


def _tickers_of(strategy: Strategy) -> frozenset[str]:
    if isinstance(strategy, TrendSwitch):
        signal, long = strategy.cfg.sleeves[strategy.sleeve]
        return frozenset({signal, long, strategy.cfg.inverse[strategy.sleeve]})
    if isinstance(strategy, Rotation):
        pairs = {ticker for pair in strategy.cfg.sleeves.values() for ticker in pair}
        return frozenset(pairs | {strategy.cfg.inverse[_KOSPI]})
    if isinstance(strategy, FlowContrarian):
        return frozenset({strategy.cfg.sleeves[_KOSPI][1], strategy.cfg.inverse[_KOSPI]})
    if isinstance(strategy, Committee | Brake):
        signal, long = strategy.cfg.sleeves[_KOSPI]
        return frozenset({signal, long, strategy.cfg.inverse[_KOSPI]})
    if isinstance(strategy, SegmentFade):
        signal, long = strategy.cfg.sleeves[strategy.sleeve]
        inverse = strategy.cfg.inverse[strategy.sleeve]
        basket = strategy.cfg.vote_basket if strategy.signal == "vote3" else ()
        return frozenset({signal, long, inverse, *basket})
    if isinstance(strategy, HoldLong):
        return frozenset({strategy.ticker})
    if isinstance(strategy, CoinFlip):
        return frozenset({strategy.long, strategy.short})
    raise TypeError(f"unknown strategy type {type(strategy).__name__}")


def _trend_params(cfg: StrategiesConfig) -> dict[str, Any]:
    return {
        "horizons": list(cfg.horizons),
        "vol_window": cfg.vol_window,
        "band": cfg.band,
        "er_window": cfg.er_window,
        "er_min": cfg.er_min,
    }


def _sign(value: float) -> int | None:
    if math.isnan(value):
        return None
    return (value > 0.0) - (value < 0.0)


def _values(series: pl.Series) -> list[float]:
    return series.cast(pl.Float64).fill_null(math.nan).to_list()


def _series(panel: Panel, column: str, ticker: str) -> pl.Series:
    return wide(panel, column, [ticker]).get_column(ticker)


def _listed_sessions(listed: pl.Series) -> list[int]:
    total = 0
    counts: list[int] = []
    for flag in listed.fill_null(False).cast(pl.Int64).to_list():
        total += flag
        counts.append(total)
    return counts


def _trend_holdings(
    panel: Panel, cfg: StrategiesConfig, signal: str, long: str, inverse: str, chop_filter: bool
) -> list[Holding]:
    prices = _series(panel, "close_ref", signal)
    score = _values(trend_score(prices, _series(panel, "r_ref", signal), cfg.horizons, cfg.vol_window))
    ratio = _values(efficiency_ratio(prices, cfg.er_window))
    out: list[Holding] = []
    for value, efficiency in zip(score, ratio, strict=True):
        if math.isnan(value):
            out.append(None)
        elif chop_filter and efficiency < cfg.er_min:
            out.append(_CASH)
        elif value > cfg.band:
            out.append(long)
        elif value < -cfg.band:
            out.append(inverse)
        else:
            out.append(_CASH)
    return out


def _rotation_holdings(panel: Panel, cfg: StrategiesConfig, hold: int) -> list[Holding]:
    names = sorted(cfg.sleeves)
    signals = [cfg.sleeves[name][0] for name in names]
    longs = [cfg.sleeves[name][1] for name in names]
    prices = wide(panel, "close_ref", signals)
    returns = wide(panel, "r_ref", signals)
    listed = wide(panel, "listed", longs)
    scores = {
        name: _values(trend_score(prices.get_column(sig), returns.get_column(sig), cfg.horizons, cfg.vol_window))
        for name, sig in zip(names, signals, strict=True)
    }
    listed_counts = {name: _listed_sessions(listed.get_column(long)) for name, long in zip(names, longs, strict=True)}
    n = len(panel.sessions)
    kospi_scores = scores[_KOSPI]
    first = next((day for day, value in enumerate(kospi_scores) if not math.isnan(value)), n)

    def pick(day: int) -> Holding:
        best_name: str | None = None
        best = -math.inf
        for name in names:
            score = scores[name][day]
            if math.isnan(score) or listed_counts[name][day] < cfg.min_listed_sessions:
                continue
            if score > best:
                best, best_name = score, name
        if best_name is not None and best > cfg.band:
            return cfg.sleeves[best_name][1]
        if not math.isnan(kospi_scores[day]) and kospi_scores[day] < -cfg.band:
            return cfg.inverse[_KOSPI]
        return _CASH

    out: list[Holding] = []
    current: Holding = None
    for day in range(n):
        if day >= first and (day - first) % hold == 0:
            current = pick(day)
        out.append(current)
    return out


def _down_streak(returns: list[float], length: int) -> list[bool]:
    run = 0
    out: list[bool] = []
    for value in returns:
        run = run + 1 if not math.isnan(value) and value < 0.0 else 0
        out.append(run >= length)
    return out


def _vote(hold: str, long: str, inverse: str) -> int:
    if hold == long:
        return 1
    if hold == inverse:
        return -1
    return 0


def _emit(sessions: Sequence[date], holdings: Sequence[Holding], anchor: str) -> pl.DataFrame:
    """Turn per-session holdings into decision rows, emitting only on the first defined state and on changes."""
    days: list[date] = []
    tickers: list[str] = []
    weights: list[float] = []
    previous: Holding = None
    for day, hold in zip(sessions, holdings, strict=True):
        if hold is None or hold == previous:
            continue
        previous = hold
        days.append(day)
        tickers.append(anchor if hold == _CASH else hold)
        weights.append(0.0 if hold == _CASH else 1.0)
    return pl.DataFrame({"decision_date": days, "ticker": tickers, "weight": weights}, schema=_TARGET_SCHEMA)
