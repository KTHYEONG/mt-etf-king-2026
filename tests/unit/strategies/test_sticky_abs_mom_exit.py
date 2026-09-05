def test_abs_mom_cash_exit_bar_is_hysteretic() -> None:
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.model import StickyLeaderConfig
    from src.strategies.sticky.overlays import apply_abs_mom_cash

    cfg = StickyLeaderConfig(abs_mom_cash=True, abs_mom_exit=-0.10)

    # Given a held position: -5% is above the -10% exit bar -> stay invested
    assert apply_abs_mom_cash({"A": -0.05}, cfg, held="A") == {"A": -0.05}
    # Below the exit bar -> cash
    assert apply_abs_mom_cash({"A": -0.15}, cfg, held="A") is CASH_INTENT
    # Flat book: entry bar stays at 0.0 regardless of abs_mom_exit
    assert apply_abs_mom_cash({"A": -0.05}, cfg, held=None) is CASH_INTENT
    assert apply_abs_mom_cash({"A": 0.05}, cfg, held=None) == {"A": 0.05}
    # Fail-closed: a positive exit bar is rejected back to 0.0
    bad = StickyLeaderConfig(abs_mom_cash=True, abs_mom_exit=0.20)
    assert apply_abs_mom_cash({"A": -0.01}, bad, held="A") is CASH_INTENT


def test_abs_mom_exit_bar_handles_getattr_failure() -> None:
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.overlays import apply_abs_mom_cash

    class _BadCfg:
        abs_mom_cash = True

        @property
        def abs_mom_exit(self) -> float:
            raise RuntimeError("boom")

    assert apply_abs_mom_cash({"A": -0.01}, _BadCfg(), held="A") is CASH_INTENT


def test_abs_mom_exit_bar_handles_non_finite_config() -> None:
    from types import SimpleNamespace

    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.overlays import apply_abs_mom_cash

    cfg = SimpleNamespace(abs_mom_cash=True, abs_mom_exit=float("nan"))
    assert apply_abs_mom_cash({"A": -0.01}, cfg, held="A") is CASH_INTENT
