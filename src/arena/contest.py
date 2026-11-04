"""Typed contest rules of an arena edition, loaded from `configs/arena_editions.yaml`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Literal, cast, get_args

from src.core.config import ConfigError, load_config

Scope = Literal["any", "domestic_equity", "pension", "overseas_or_mixed"]

_EDITION_KEYS = frozenset(
    {
        "title",
        "start",
        "end",
        "initial_capital_krw",
        "ranking_metric",
        "sponsors",
        "tracks",
        "prize_places_per_track",
        "prize_line",
        "observed",
        "unknowns",
        "sources",
    }
)
_TRACK_KEYS = frozenset({"label", "scope", "leverage", "inverse", "futures"})
_OBSERVED_KEYS = frozenset({"track", "rank", "multiple", "as_of", "final", "source"})


@dataclass(frozen=True)
class TrackRule:
    key: str
    label: str
    scope: Scope
    leverage: bool | None
    inverse: bool | None
    futures: bool | None


@dataclass(frozen=True)
class ObservedResult:
    track: str
    rank: int
    multiple: float
    as_of: date
    final: bool
    source: str


@dataclass(frozen=True)
class EditionSpec:
    edition: int
    title: str
    start: date
    end: date
    initial_capital_krw: int
    ranking_metric: Literal["cumulative_return"]
    sponsors: frozenset[str]
    tracks: Mapping[str, TrackRule]
    prize_places_per_track: int
    prize_line: tuple[float, float] | None
    observed: tuple[ObservedResult, ...]
    unknowns: tuple[str, ...]
    sources: tuple[str, ...]


class PrizeLineUndecidedError(RuntimeError):
    """Raised when an edition's prize-line band is still undecided and a caller needs it."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _as_int(value: Any, where: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"{where} must be an integer")
    return value


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    return float(value)


def _as_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a boolean")
    return value


def _as_opt_bool(value: Any, where: str) -> bool | None:
    return None if value is None else _as_bool(value, where)


def _as_date(value: Any, where: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ConfigError(f"{where} must be a date")
    return value


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _as_str_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list")
    return tuple(_as_str(item, where) for item in value)


def _exact_keys(section: Any, expected: frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(section, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    if set(section) != expected:
        raise ConfigError(f"{where} keys must be exactly {sorted(expected)}")
    return section


def _load_tracks(value: Any) -> Mapping[str, TrackRule]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError("tracks must be a non-empty mapping")
    rules: dict[str, TrackRule] = {}
    for raw_key, raw in value.items():
        key = _as_str(raw_key, "track key")
        section = _exact_keys(raw, _TRACK_KEYS, f"tracks.{key}")
        scope = section["scope"]
        _require(scope in get_args(Scope), f"tracks.{key}.scope is not a known scope: {scope!r}")
        rules[key] = TrackRule(
            key=key,
            label=_as_str(section["label"], f"tracks.{key}.label"),
            scope=cast(Scope, scope),
            leverage=_as_opt_bool(section["leverage"], f"tracks.{key}.leverage"),
            inverse=_as_opt_bool(section["inverse"], f"tracks.{key}.inverse"),
            futures=_as_opt_bool(section["futures"], f"tracks.{key}.futures"),
        )
    return MappingProxyType(rules)


def _load_prize_line(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError("prize_line must be null or a [lo, hi] pair")
    lo = _as_number(value[0], "prize_line")
    hi = _as_number(value[1], "prize_line")
    _require(0.0 < lo <= hi, "prize_line must satisfy 0 < lo <= hi")
    return lo, hi


def _load_observed(value: Any, tracks: Mapping[str, TrackRule]) -> tuple[ObservedResult, ...]:
    if not isinstance(value, list):
        raise ConfigError("observed must be a list")
    results: list[ObservedResult] = []
    for raw in value:
        section = _exact_keys(raw, _OBSERVED_KEYS, "observed item")
        track = _as_str(section["track"], "observed.track")
        _require(track in tracks, f"observed track {track!r} is not a track of this edition")
        rank = _as_int(section["rank"], "observed.rank")
        _require(rank >= 1, "observed.rank must be >= 1")
        multiple = _as_number(section["multiple"], "observed.multiple")
        _require(multiple > 0.0, "observed.multiple must be > 0")
        results.append(
            ObservedResult(
                track=track,
                rank=rank,
                multiple=multiple,
                as_of=_as_date(section["as_of"], "observed.as_of"),
                final=_as_bool(section["final"], "observed.final"),
                source=_as_str(section["source"], "observed.source"),
            )
        )
    return tuple(results)


def load_edition(edition: int, name: str = "arena_editions") -> EditionSpec:
    """Return the validated rules of `editions[edition]`; raises ConfigError if missing or invalid."""
    editions = load_config(name).get("editions")
    if not isinstance(editions, Mapping) or edition not in editions:
        raise ConfigError(f"edition {edition} not found in config {name}")
    section = _exact_keys(editions[edition], _EDITION_KEYS, f"editions.{edition}")

    start = _as_date(section["start"], "start")
    end = _as_date(section["end"], "end")
    _require(start < end, "start must precede end")
    capital = _as_int(section["initial_capital_krw"], "initial_capital_krw")
    _require(capital > 0, "initial_capital_krw must be > 0")
    _require(section["ranking_metric"] == "cumulative_return", "ranking_metric must be cumulative_return")
    sponsors = _as_str_list(section["sponsors"], "sponsors")
    _require(bool(sponsors) and len(set(sponsors)) == len(sponsors), "sponsors must be non-empty without duplicates")
    places = _as_int(section["prize_places_per_track"], "prize_places_per_track")
    _require(places >= 1, "prize_places_per_track must be >= 1")
    tracks = _load_tracks(section["tracks"])

    return EditionSpec(
        edition=edition,
        title=_as_str(section["title"], "title"),
        start=start,
        end=end,
        initial_capital_krw=capital,
        ranking_metric="cumulative_return",
        sponsors=frozenset(sponsors),
        tracks=tracks,
        prize_places_per_track=places,
        prize_line=_load_prize_line(section["prize_line"]),
        observed=_load_observed(section["observed"], tracks),
        unknowns=_as_str_list(section["unknowns"], "unknowns"),
        sources=_as_str_list(section["sources"], "sources"),
    )


def require_prize_line(spec: EditionSpec) -> tuple[float, float]:
    """Return the edition's prize-line band; raises PrizeLineUndecidedError while it is still null."""
    if spec.prize_line is None:
        raise PrizeLineUndecidedError(f"prize_line of edition {spec.edition} is undecided")
    return spec.prize_line
