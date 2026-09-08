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


def test_cutoff_auc_exported_from_objective_package() -> None:
    from src.tournament.objective import CUTOFF_AUC_IS_PRODUCTION_GATE
    from src.tournament.objective import cutoff_auc_score as pkg_score
    from src.tournament.objective import evaluate_attack_policy as pkg_eval
    from src.tournament.objective import evaluate_championship_adoption
    from src.tournament.objective_impl import cutoff_auc_score as impl_score

    assert CUTOFF_AUC_IS_PRODUCTION_GATE is False
    assert pkg_score is impl_score
    assert callable(pkg_eval)
    assert callable(evaluate_championship_adoption)
