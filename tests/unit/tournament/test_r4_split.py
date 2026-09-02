def test_r4_distribution_shim_reexports() -> None:
    from src.tournament.distribution import ReturnDistribution
    from src.tournament.distribution_core import ReturnDistribution as CoreDist

    assert ReturnDistribution is CoreDist

def test_r4_championship_module_exports_eval() -> None:
    from src.tournament.championship import evaluate_championship_adoption as direct
    from src.tournament.objective import evaluate_championship_adoption as shim

    assert direct is shim

def test_r4_tournament_shim_line_budget() -> None:
    from pathlib import Path

    for rel in ("src/tournament/distribution.py", "src/tournament/objective.py"):
        count = sum(1 for _ in Path(rel).open(encoding="utf-8"))  # noqa: SIM115
        assert count <= 120, f"{rel} has {count} lines"
