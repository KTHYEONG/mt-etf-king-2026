# ruff: noqa

def test_cash_accounting_preserves_missing_open_asset() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    result=cash_order_transition(**kw)
    assert result.equity_close == 100.0
    assert result.state.shares["A"] == 4.0
    assert result.state.cash >= 5.0
    assert all(q == int(q) for q in result.state.shares.values())


def test_cash_accounting_validates_cash_and_hold_fallback() -> None:
    from datetime import date
    import pytest
    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger_state import PortfolioLedgerState
    from src.execution.cash_accounting import cash_order_transition
    from src.portfolio.intent import HOLD_INTENT, PortfolioIntent

    kw = dict(prior_state=PortfolioLedgerState(cash=10.0, shares={"A": 1.0}), intent=PortfolioIntent(kind="target", weights={}), decision_date=date(2024, 1, 2), prev_closes={"A": 10.0}, opens={}, closes={}, cost_model=CostModel(CostConfig()), adv_by_ticker={"A": 1e6}, max_order_to_adv=0.01, exposure_limits=(0.8, 1.6, 0.05), leverage_multiples={"A": 1}, execution=None, panel=None)
    with pytest.raises(ValueError):
        cash_order_transition(**(kw | {"prior_state": PortfolioLedgerState(cash=-1.0, shares={})}))
    held = cash_order_transition(**(kw | {"intent": HOLD_INTENT}))
    assert held.state.shares == {"A": 1.0}

def test_cash_accounting_cash_sells_only_fillable_positions() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["intent"]=CASH_INTENT
    result=cash_order_transition(**kw)
    assert result.state.shares == {"A":4.0}
    assert result.state.cash == 60.0
    assert result.equity_close == 100.0

def test_cash_accounting_hold_marks_without_trading() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["intent"]=HOLD_INTENT
    kw["closes"]={"A":12.0,"B":10.0}
    result=cash_order_transition(**kw)
    assert result.state == kw["prior_state"]
    assert result.equity_close == 108.0
    assert result.diagnostics.transaction_cost == 0.0
    assert not result.fills

def test_cash_accounting_all_missing_open_preserves_marks() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["opens"]={}
    kw["closes"]={}
    result=cash_order_transition(**kw)
    assert result.state == kw["prior_state"]
    assert result.equity_close == 100.0
    assert not result.fills

def test_cash_accounting_no_borrowing_during_slow_rotation() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["prior_state"]=PortfolioLedgerState(cash=20.0,shares={"A":8.0})
    kw["opens"]={"A":10.0,"B":10.0}
    kw["adv_by_ticker"]={"A":1.0,"B":1e6}
    kw["cost_model"]=CostModel(CostConfig(commission_bps=10.0))
    result=cash_order_transition(**kw)
    assert result.state.shares["A"] == 8.0
    assert result.state.cash >= 0.05 * result.equity_close
    assert abs(result.equity_close - (100.0-result.diagnostics.transaction_cost)) < 1e-10
    assert result.state.shares.get("B",0.0) <= 1.0

def test_cash_accounting_costs_and_integer_adv_limit() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["prior_state"]=PortfolioLedgerState(cash=100.0,shares={})
    kw["intent"]=PortfolioIntent(kind="target",weights={"B":0.8})
    kw["adv_by_ticker"]={"B":2500.0}
    kw["cost_model"]=CostModel(CostConfig(commission_bps=10.0))
    result=cash_order_transition(**kw)
    assert result.state.shares == {"B":2.0}
    assert abs(result.state.cash-79.98) < 1e-10
    assert abs(result.equity_close-99.98) < 1e-10
    assert abs(result.diagnostics.transaction_cost-0.02) < 1e-10

def test_cash_accounting_missing_adv_blocks_buy() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["prior_state"]=PortfolioLedgerState(cash=100.0,shares={})
    kw["adv_by_ticker"]={}
    result=cash_order_transition(**kw)
    assert result.state.cash == 100.0
    assert result.state.shares == {}

def test_cash_accounting_rejects_invalid_financial_inputs() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    import pytest
    for changes in ({"max_order_to_adv":0.0},{"max_order_to_adv":float("nan")},{"lot_size":0},{"leverage_multiples":{"A":1}},{"prev_closes":{}},{"intent":PortfolioIntent(kind="target",weights={"B":float("nan")})},{"cost_model":CostModel(CostConfig(commission_bps=-1.0))}):
        with pytest.raises(ValueError):
            cash_order_transition(**(kw | changes))

def test_cash_accounting_fractional_compatibility() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    kw["lot_size"]=None
    kw["prior_state"]=PortfolioLedgerState(cash=100.0,shares={})
    kw["adv_by_ticker"]={"B":2500.0}
    result=cash_order_transition(**kw)
    assert result.state.shares == {"B":2.5}
    assert result.equity_close == 100.0

def test_cash_accounting_respects_real_next_open() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    import polars as pl
    from src.core.calendar import get_calendar
    from src.backtest.execution import NextOpenExecution
    kw["prior_state"]=PortfolioLedgerState(cash=100.0,shares={})
    kw["opens"]={"B":20.0}
    kw["closes"]={"B":20.0}
    kw["execution"]=NextOpenExecution(get_calendar())
    kw["panel"]=pl.DataFrame({"date":[date(2024,1,2),date(2024,1,3)],"ticker":["B","B"],"open":[10.0,20.0]})
    result=cash_order_transition(**kw)
    assert result.state.shares == {"B":4.0}
    assert all(f.execution_date == date(2024,1,3) for f in result.fills)
    assert all(f.price == 20.0 for f in result.fills)
