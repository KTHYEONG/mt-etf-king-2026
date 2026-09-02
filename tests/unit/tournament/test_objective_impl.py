def test_objective_impl_exports_gate_config() -> None:
    from src.tournament.objective import ObjectiveGateConfig
    from src.tournament.objective_impl import ObjectiveGateConfig as ImplConfig

    assert ObjectiveGateConfig is ImplConfig


def test_objective_impl_evaluate_championship_adoption() -> None:
    from src.tournament.championship import evaluate_championship_adoption as champ
    from src.tournament.objective_impl import evaluate_championship_adoption as impl

    assert callable(champ)
    assert callable(impl)
    assert champ.__name__ == impl.__name__
