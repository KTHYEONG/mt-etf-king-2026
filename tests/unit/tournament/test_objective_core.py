def test_objective_core_imports() -> None:
    from src.tournament.objective_core import evaluate_objective_gates
    assert callable(evaluate_objective_gates)
