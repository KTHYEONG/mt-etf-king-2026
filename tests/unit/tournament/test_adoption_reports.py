def test_adoption_reports_imports() -> None:
    from src.tournament.objective_core import evaluate_p15_adoption_report
    assert callable(evaluate_p15_adoption_report)
