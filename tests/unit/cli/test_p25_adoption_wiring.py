def test_p25_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, build_parser, cmd_backtest, cmd_decide

    assert 'P25' in STICKY_ADOPTION_MODELS
    assert 'P24' in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert 'P25' in bt
    assert 'house_money_ratchet_returns' in bt
    assert 'evaluate_p25_adoption_gates' in bt
    assert 'continuation_capture' in bt
    assert 'overlay_right_tail_stats' in bt
    assert 'championship_lock_returns' in bt
    dec = inspect.getsource(cmd_decide)
    assert 'house_money_should_cash' in dec
    assert 'remaining_sessions' in dec
    assert 'load_p25_arm' in dec
    parser = build_parser()
    dec_parser = None
    for action in parser._subparsers._group_actions:
        for name, sub in action.choices.items():
            if name == 'decide':
                dec_parser = sub
    assert dec_parser is not None
    dests = [a.dest for a in dec_parser._actions]
    assert 'capital' in dests
