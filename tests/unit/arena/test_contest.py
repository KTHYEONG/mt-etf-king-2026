import copy
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, cast

import pytest

from src.arena import contest as ct
from src.arena.contest import PrizeLineUndecidedError, load_edition, require_prize_line
from src.core.config import ConfigError, load_config

_REAL: dict[str, Any] = load_config("arena_editions")["editions"][2026]
Mutator = Callable[[dict[str, Any]], None]


def _with_section(monkeypatch: pytest.MonkeyPatch, mutate: Mutator) -> None:
    section: dict[str, Any] = copy.deepcopy(_REAL)
    mutate(section)
    monkeypatch.setattr(ct, "load_config", lambda name: {"editions": {2026: section}})


def _set(**changes: Any) -> Mutator:
    return lambda section: section.update(changes)


def _set_track(key: str, **changes: Any) -> Mutator:
    return lambda section: section["tracks"][key].update(changes)


def _set_observed(**changes: Any) -> Mutator:
    return lambda section: section["observed"][0].update(changes)


def test_edition_2026_loads() -> None:
    spec = load_edition(2026)
    assert len(spec.sponsors) == 10
    assert len(spec.tracks) == 4
    assert spec.tracks["autonomous"].scope == "any"
    assert spec.prize_line is None
    assert len(spec.observed) == 1
    assert spec.observed[0].final is False


def test_missing_edition_rejected() -> None:
    with pytest.raises(ConfigError):
        load_edition(1999)


def test_edition_container_shape_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    for data in ({}, {"editions": []}, {"editions": {2026: 5}}):
        monkeypatch.setattr(ct, "load_config", lambda name, data=data: data)
        with pytest.raises(ConfigError):
            load_edition(2026)


def test_extra_or_missing_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _drop(key: str) -> Mutator:
        return lambda section: section.pop(key)

    def _drop_track_label(section: dict[str, Any]) -> None:
        section["tracks"]["pension"].pop("label")

    mutations: list[Mutator] = [
        _set(extra=1),
        _drop("end"),
        _set_track("pension", extra=1),
        _drop_track_label,
        _set_observed(extra=1),
    ]
    for mutate in mutations:
        _with_section(monkeypatch, mutate)
        with pytest.raises(ConfigError):
            load_edition(2026)


def test_bad_values_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_mutations: list[Mutator] = [
        _set(start=date(2026, 11, 13), end=date(2026, 9, 21)),
        _set(start=date(2026, 11, 13)),
        _set(start=datetime(2026, 9, 21)),
        _set(initial_capital_krw=True),
        _set(initial_capital_krw=0),
        _set(ranking_metric="sharpe"),
        _set(sponsors=[]),
        _set(sponsors=["A", "A"]),
        _set(sponsors="A"),
        _set(sponsors=[1]),
        _set(prize_places_per_track=0),
        _set(prize_places_per_track=True),
        _set(tracks={}),
        _set(tracks=[]),
        _set(tracks={"x": 5}),
        _set_track("autonomous", scope="spot"),
        _set_track("autonomous", scope=5),
        _set_track("autonomous", leverage="yes"),
        _set_track("autonomous", inverse=1),
        _set_track("autonomous", futures=0),
        _set_track("autonomous", label=""),
        _set(prize_line=[2.5, 2.0]),
        _set(prize_line=[0.0, 2.0]),
        _set(prize_line=[2.0]),
        _set(prize_line="2.0,2.5"),
        _set(prize_line=[2.0, "x"]),
        _set(observed="x"),
        _set_observed(track="nope"),
        _set_observed(rank=0),
        _set_observed(rank=True),
        _set_observed(multiple=0.0),
        _set_observed(multiple=False),
        _set_observed(as_of=datetime(2026, 11, 3)),
        _set_observed(final="no"),
        _set_observed(source=""),
        _set(unknowns="x"),
        _set(unknowns=[1]),
        _set(sources=[1]),
        _set(title=1),
    ]
    for mutate in bad_mutations:
        _with_section(monkeypatch, mutate)
        with pytest.raises(ConfigError):
            load_edition(2026)


def test_tracks_read_only() -> None:
    spec = load_edition(2026)
    with pytest.raises(TypeError):
        cast(dict[str, Any], spec.tracks)["new"] = spec.tracks["autonomous"]
    assert all(rule.key == key for key, rule in spec.tracks.items())


def test_prize_line_required_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_edition(2026)
    assert spec.observed
    with pytest.raises(PrizeLineUndecidedError, match="2026"):
        require_prize_line(spec)

    _with_section(monkeypatch, _set(prize_line=[2.0, 2.5]))
    assert require_prize_line(load_edition(2026)) == (2.0, 2.5)
