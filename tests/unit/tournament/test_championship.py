def test_championship_imports() -> None:
    from src.tournament.championship import evaluate_championship_adoption
    assert callable(evaluate_championship_adoption)
