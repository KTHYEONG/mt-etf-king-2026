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


def test_sticky_factory_facade_reexports_inactive_participate() -> None:
    import src.strategies.factories.sticky as facade
    from src.strategies.factories.sticky_mom60 import make_sticky_mom60_inactive_participate as origin

    # 파사드가 원본과 동일 객체를 재수출한다
    assert facade.make_sticky_mom60_inactive_participate is origin
    # __all__ 등재 (다른 sticky 팩토리와 동일 관례)
    assert "make_sticky_mom60_inactive_participate" in facade.__all__
    assert "make_sticky_mom60_runner_reversal" in facade.__all__

    # registry가 파사드 경유로 동일 팩토리를 등록했는지 행동으로 확인
    from src.strategies.ids import STICKY_MOM60_INACTIVE_PARTICIPATE
    from src.strategies.registry import STRATEGIES

    model = STRATEGIES[STICKY_MOM60_INACTIVE_PARTICIPATE]()
    assert model.name == STICKY_MOM60_INACTIVE_PARTICIPATE
    assert model.config.inactive_participation is True
