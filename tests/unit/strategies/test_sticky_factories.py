"""P3: src/strategies/sticky/factories.py now repoints to src.strategies.factories.sticky."""

from __future__ import annotations

from src.strategies.ids import STICKY_MOM60_RAW


def test_factory_registry_covers_sticky_semantic_ids() -> None:
    from src.strategies.sticky.factories import FACTORY_REGISTRY

    assert STICKY_MOM60_RAW in FACTORY_REGISTRY
    model = FACTORY_REGISTRY[STICKY_MOM60_RAW]()
    assert model.name == STICKY_MOM60_RAW


def test_make_sticky_mom60_raw_matches_registry_entry() -> None:
    from src.strategies.sticky.factories import make_sticky_mom60_raw

    model = make_sticky_mom60_raw()
    assert getattr(model, "name", None) == STICKY_MOM60_RAW
