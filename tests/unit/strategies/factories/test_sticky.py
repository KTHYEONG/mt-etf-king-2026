# ruff: noqa
"""Facade coverage for the post-crash anchor factory (checker conventional path)."""


def test_sticky_factory_facade_reexports_post_crash_anchor() -> None:
    import src.strategies.factories.sticky as facade
    from src.strategies.factories.sticky_mom60 import make_sticky_mom60_post_crash_anchor as origin

    assert facade.make_sticky_mom60_post_crash_anchor is origin
    assert "make_sticky_mom60_post_crash_anchor" in facade.__all__
