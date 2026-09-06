def test_p25_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, build_parser
    from src.cli.commands.backtest.families.sticky_house import _hook_house_money
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    assert 'sticky.house_money' in STICKY_ADOPTION_MODELS
    assert 'sticky.mom60_peak_lock' in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(_hook_house_money)
    assert 'house_money_ratchet_returns' in bt
    assert 'evaluate_p25_adoption_gates' in bt
    assert 'continuation_capture' in bt
    assert 'overlay_right_tail_stats' in bt
    assert 'championship_lock_returns' in bt
    dec = inspect.getsource(_OVERLAY_HOOKS['sticky.house_money'])
    assert 'house_money_should_cash' in dec
    assert 'remaining_sessions' in dec
    assert 'overlay_param' in dec
    parser = build_parser()
    dec_parser = None
    for action in parser._subparsers._group_actions:
        for name, sub in action.choices.items():
            if name == 'decide':
                dec_parser = sub
    assert dec_parser is not None
    dests = [a.dest for a in dec_parser._actions]
    assert 'capital' in dests


def test_p25_decide_and_backtest_share_alpha_and_objective() -> None:
    import inspect

    from src.cli.commands.backtest.families.sticky_house import _hook_house_money
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    backtest_source = inspect.getsource(_hook_house_money)
    decide_source = inspect.getsource(_OVERLAY_HOOKS['sticky.house_money'])

    assert 'evaluate_championship_adoption' in backtest_source
    assert 'execution_faithful_late_lock_returns' in backtest_source
    assert 'optimize_p25_overlay' in backtest_source
    assert '_BL_P25_WIRING["sticky.house_money"]' in decide_source
    assert 'restore_state' in decide_source
