def test_overlays_imports() -> None:
    from src.strategies.sticky.overlays import apply_impulse_switch
    assert callable(apply_impulse_switch)
