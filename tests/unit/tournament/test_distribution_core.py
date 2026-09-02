def test_distribution_core_exports_return_distribution() -> None:
    from src.tournament.distribution import ReturnDistribution
    from src.tournament.distribution_core import ReturnDistribution as CoreDist

    assert ReturnDistribution is CoreDist


def test_distribution_core_ruin_probability_callable() -> None:
    from src.tournament.distribution_core import ruin_probability

    assert callable(ruin_probability)
