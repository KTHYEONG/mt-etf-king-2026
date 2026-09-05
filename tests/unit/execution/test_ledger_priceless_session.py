from datetime import date


def test_transition_priceless_session_conserves_equity() -> None:
    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import PortfolioIntent

    # Given: 1.0B KRW portfolio, 95% in X at 100,000 KRW, on a session with no prices at all
    prior = PortfolioLedgerState(cash=50_000_000.0, shares={"X": 9_500.0})
    intent = PortfolioIntent(kind="target", weights={"X": 0.95})

    # When
    result = transition_portfolio_state(
        prior_state=prior,
        intent=intent,
        decision_date=date(2026, 6, 2),
        prev_closes={"X": 100_000.0},
        opens={},
        closes={},
        cost_model=CostModel(CostConfig(commission_bps=0.0, slippage_bps=0.0)),
        adv_by_ticker={"X": 1e12},
        max_order_to_adv=0.01,
        exposure_limits=(0.95, 1.90, 0.05),
        leverage_multiples={"X": 2},
        execution=None,
        panel=None,
    )

    # Then: equity conserved, not duplicated (regression: was 1.0B cash while retaining 9,500 shares)
    assert result.equity_close == 1_000_000_000.0
    assert result.state.cash == 50_000_000.0
    assert result.state.shares == {"X": 9_500.0}
    assert result.session_return == 0.0
    assert result.fills == ()


def test_carry_forward_marks_never_prices_missing_at_zero() -> None:
    from src.execution.ledger import carry_forward_marks, is_priceless_session

    # Given
    shares = {"A": 10.0, "B": 5.0, "C": 1.0}
    closes = {"A": 200.0}
    opens = {"B": 150.0, "A": 190.0}
    prev_closes = {"C": float("nan")}

    # When
    marks = carry_forward_marks(shares, closes, opens, prev_closes)

    # Then: first finite price wins per ticker; C has none and is omitted, not zero
    assert marks == {"A": 200.0, "B": 150.0}
    assert "C" not in marks
    assert is_priceless_session(shares, {}) is True
    assert is_priceless_session(shares, {"A": 190.0}) is False
    assert is_priceless_session({}, {}) is False


def test_is_priceless_session_handles_bad_inputs() -> None:
    from src.execution.ledger import carry_forward_marks, is_priceless_session

    assert is_priceless_session({"A": "bad"}, {}) is False  # type: ignore[arg-type]
    assert is_priceless_session({"A": 0.0}, {}) is False
    assert is_priceless_session({"A": 1.0}, {"A": "bad"}) is True
    assert carry_forward_marks("bad", {"A": 1.0}) == {}  # type: ignore[arg-type]
    assert carry_forward_marks({"A": 1.0}, {"A": object()}) == {}  # type: ignore[dict-item]
    assert carry_forward_marks({"A": 1.0}, "bad", {"A": 2.0}) == {"A": 2.0}  # type: ignore[arg-type]


class _BadShares:
    def items(self):
        raise RuntimeError("boom")


def test_is_priceless_session_returns_false_for_unreadable_shares() -> None:
    from src.execution.ledger import is_priceless_session

    assert is_priceless_session(_BadShares(), {}) is False


def test_priceless_session_equity_prev_fallback_when_prev_closes_invalid() -> None:
    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import PortfolioIntent

    prior = PortfolioLedgerState(cash=25_000_000.0, shares={"X": 100.0})
    result = transition_portfolio_state(
        prior_state=prior,
        intent=PortfolioIntent(kind="hold"),
        decision_date=date(2026, 6, 2),
        prev_closes={},
        opens={},
        closes={},
        cost_model=CostModel(CostConfig(commission_bps=0.0, slippage_bps=0.0)),
        adv_by_ticker={"X": 1e12},
        max_order_to_adv=0.01,
        exposure_limits=(0.95, 1.90, 0.05),
        leverage_multiples={"X": 1},
        execution=None,
        panel=None,
    )
    assert result.equity_close == 25_000_000.0


class _BadOpens:
    def __getitem__(self, key: str) -> float:
        return 1.0

    def __iter__(self):
        raise RuntimeError("bad opens")

    def __len__(self) -> int:
        return 1


def test_is_priceless_session_tolerates_unreadable_opens_map() -> None:
    from collections.abc import Mapping

    from src.execution.ledger import is_priceless_session

    class _BadOpensMapping(Mapping[str, float]):
        def __getitem__(self, key: str) -> float:
            return 1.0

        def __iter__(self):
            raise RuntimeError("bad opens")

        def __len__(self) -> int:
            return 1

    assert is_priceless_session({"A": 1.0}, _BadOpensMapping()) is True


class _BadPriceMap:
    def __getitem__(self, key: str) -> float:
        raise RuntimeError("bad lookup")

    def __iter__(self):
        return iter(["A"])

    def __len__(self) -> int:
        return 1


def test_carry_forward_marks_skips_maps_with_bad_lookup() -> None:
    from collections.abc import Mapping

    from src.execution.ledger import carry_forward_marks

    class _BadGetMap(Mapping[str, float]):
        def __getitem__(self, key: str) -> float:
            return 1.0

        def get(self, key: str, default: object = None) -> float:
            raise RuntimeError("bad lookup")

        def __iter__(self):
            return iter(["A"])

        def __len__(self) -> int:
            return 1

    assert carry_forward_marks({"A": 1.0}, _BadGetMap(), {"A": 2.0}) == {"A": 2.0}


def test_priceless_session_falls_back_when_prev_equity_unreadable() -> None:
    from types import SimpleNamespace

    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger import transition_portfolio_state
    from src.portfolio.intent import PortfolioIntent

    prior = SimpleNamespace(
        cash=10.0,
        shares={"X": 1.0},
        equity_at_prices=lambda _prices: (_ for _ in ()).throw(RuntimeError("bad equity")),
    )
    result = transition_portfolio_state(
        prior_state=prior,
        intent=PortfolioIntent(kind="hold"),
        decision_date=date(2026, 6, 2),
        prev_closes={"X": 100.0},
        opens={},
        closes={},
        cost_model=CostModel(CostConfig(commission_bps=0.0, slippage_bps=0.0)),
        adv_by_ticker={"X": 1e12},
        max_order_to_adv=0.01,
        exposure_limits=(0.95, 1.90, 0.05),
        leverage_multiples={"X": 1},
        execution=None,
        panel=None,
    )
    assert result.equity_close == 110.0


class _WeirdShare:
    def __init__(self) -> None:
        self._calls = 0

    def __float__(self) -> float:
        self._calls += 1
        if self._calls <= 4:
            return 2.0
        raise ValueError("bad share")


def test_priceless_session_skips_bad_weight_entries() -> None:
    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import PortfolioIntent

    prior = PortfolioLedgerState(cash=50.0, shares={"Y": _WeirdShare()})
    result = transition_portfolio_state(
        prior_state=prior,
        intent=PortfolioIntent(kind="hold"),
        decision_date=date(2026, 6, 2),
        prev_closes={"Y": 10.0},
        opens={},
        closes={"Y": 10.0},
        cost_model=CostModel(CostConfig(commission_bps=0.0, slippage_bps=0.0)),
        adv_by_ticker={"Y": 1e12},
        max_order_to_adv=0.01,
        exposure_limits=(0.95, 1.90, 0.05),
        leverage_multiples={"Y": 1},
        execution=None,
        panel=None,
    )
    assert result.equity_close == 70.0
    assert result.weights_after_close == {}
