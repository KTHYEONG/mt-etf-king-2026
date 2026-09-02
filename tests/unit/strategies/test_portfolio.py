def test_portfolio_imports() -> None:
    from src.strategies.baselines.portfolio import make_portfolio_momentum_policy
    assert callable(make_portfolio_momentum_policy)
