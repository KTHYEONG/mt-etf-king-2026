def test_p27_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    assert "sticky.mom60_raw" in STICKY_ADOPTION_MODELS
    assert "sticky.mom60_concentrated" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(_hook_mom60_raw)
    assert "load_p27_exposure_limits" in bt
    assert "field_relative_report" in bt
    assert "oneshot_anchor_starts" in bt
    assert "oneshot_window_returns" in bt
    assert "evaluate_championship_adoption" in bt
    assert "rolling.returns" in bt
    dec = inspect.getsource(_OVERLAY_HOOKS["sticky.mom60_raw"])
    assert "overlay_should_cash" in dec
    assert "load_overlay_mode" in dec


def test_p27_decide_identity_overlay_does_not_force_cash() -> None:
    import inspect

    from src.cli.commands.decide.models import _OVERLAY_HOOKS
    from src.tournament.policy import overlay_should_cash

    src = inspect.getsource(_OVERLAY_HOOKS["sticky.mom60_raw"])
    assert "overlay_should_cash" in src
    assert "load_overlay_mode" in src
    assert overlay_should_cash("identity", 0.99, 0, 0.50, 5) is False


def test_p27_cli_independent_window_eval_wiring() -> None:
    import inspect

    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw
    from src.cli.commands.backtest.families.sticky_house import _hook_house_money
    from src.cli.commands.backtest.families.sticky_mom60 import _hook_mom60_concentrated

    p27_src = inspect.getsource(_hook_mom60_raw)
    p26_src = inspect.getsource(_hook_mom60_concentrated)
    p25_src = inspect.getsource(_hook_house_money)
    assert "oneshot_independent_window_returns" in p27_src
    assert "oneshot_independent_window_returns(" in p27_src
    assert "path_dependent=_b21_flags.path_dependent" in p27_src
    assert "path_dependent=False" not in p27_src
    assert "path_dependent=_b21_flags_p26.path_dependent" in p26_src
    assert "path_dependent=_b21_flags_p25.path_dependent" in p25_src


def test_p27_cli_incumbent_no_slow_override() -> None:
    import inspect

    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    p27_src = inspect.getsource(_hook_mom60_raw)
    assert "path_dependent_mode=('slow'" not in p27_src
    assert "path_dependent_mode=_path_mode" in p27_src or "resolve_path_dependent_mode" in p27_src
    assert "exposure_limits=_p21_alpha_limits_p27" in p27_src
    assert "resolve_exposure_limits_for_model" in p27_src


def test_sticky_shared_session_cache_wiring() -> None:
    import inspect

    import src.cli.commands.backtest._core as _core
    from src.cli.commands.backtest import _prep

    core_src = inspect.getsource(_core.run_cells) + inspect.getsource(_prep.prepare_run)
    assert "is_pd" in core_src
    assert "session_cache=shared_cache" in core_src


def test_p27_cli_gross_diagnostics_avoids_undefined_name() -> None:
    import inspect
    import re

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    p27_src = inspect.getsource(_hook_mom60_raw)
    assert 'getattr(rolling, "diagnostics"' in p27_src
    assert re.search(r"^\s*_ = diagnostics\b", p27_src, flags=re.M) is None
    assert "gross_violation_count" in p27_src
    assert "sticky.mom60_raw" in STICKY_ADOPTION_MODELS


def test_p27_hook_records_cutoff_auc_without_replacing_championship_gate() -> None:
    import inspect

    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    src = inspect.getsource(_hook_mom60_raw)
    assert "cutoff_auc_score" in src
    assert "mean_smooth_cutoff_utility" in src
    assert "CUTOFF_AUC_IS_PRODUCTION_GATE" in src
    assert 'summary["cutoff_auc_score"]' in src
    assert 'summary["cutoff_auc_smooth_mean"]' in src
    assert 'summary["cutoff_auc_is_production_gate"]' in src
    assert "evaluate_championship_adoption" in src
    assert 'summary["championship_gate_status"]' in src
    champ_at = src.index("evaluate_championship_adoption")
    auc_at = src.index("cutoff_auc_score")
    status_at = src.index('summary["championship_gate_status"]')
    assert champ_at < status_at
    assert "evaluate_attack_policy" not in src.split('summary["championship_gate_status"]')[1]


def test_p27_hook_executes_cutoff_auc_summary_fields() -> None:
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    returns = [0.61, 0.51, 0.41, 0.0]
    rolling = SimpleNamespace(returns=returns, backtest=None, diagnostics=None)
    cal = MagicMock()
    cal.sessions.return_value = []
    summary: dict[str, object] = {}
    cell = SimpleNamespace(
        summary=summary,
        model=MagicMock(),
        engine=MagicMock(),
        panel=MagicMock(),
        case_config=MagicMock(),
        horizon=36,
        simulator=MagicMock(),
        close_map={},
        lev_allowed=True,
        inv_allowed=True,
        eval_mode="identity",
        rolling=rolling,
        cal=cal,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        path_mode="default",
        shared_cache={},
        master=MagicMock(),
        rolling_exposure_limits=None,
    )
    _hook_mom60_raw(cell)
    assert "cutoff_auc_score" in summary
    assert "cutoff_auc_smooth_mean" in summary
    assert summary["cutoff_auc_is_production_gate"] is True


def test_prep_wires_kospi_championship_sleeve_map() -> None:
    import inspect

    from src.cli.commands.backtest import _prep


    src = inspect.getsource(_prep)
    assert "kospi_sleeve_feature_maps" in src
    assert "build_championship_sleeve_map" in src
    assert "championship_sleeves" in src
    assert "prep_championship_sleeves_on_engine" in src
    from datetime import date, timedelta
    from types import SimpleNamespace

    import polars as pl

    base = date(2026, 1, 2)
    idx = pl.DataFrame(
        {
            "index_name": ["KOSPI"] * 70,
            "date": [base + timedelta(days=i) for i in range(70)],
            "close": [100.0 + float(i) for i in range(70)],
        }
    )
    engine = SimpleNamespace()
    out = _prep.prep_championship_sleeves_on_engine(engine, idx)
    assert isinstance(out, dict)
    assert engine.championship_sleeves == out


def test_prep_championship_sleeves_on_engine_skips_when_index_none() -> None:
    from types import SimpleNamespace

    from src.cli.commands.backtest import _prep

    engine = SimpleNamespace()
    assert _prep.prep_championship_sleeves_on_engine(engine, None) is None
    assert not hasattr(engine, "championship_sleeves")


def test_load_index_daily_panel_missing_returns_none(tmp_path) -> None:
    from src.cli.commands.backtest import _prep
    from src.core.paths import DataPaths

    paths = DataPaths(root=tmp_path)
    assert _prep.load_index_daily_panel(paths) is None


def test_load_index_daily_panel_read_failure_returns_none(tmp_path) -> None:
    from src.cli.commands.backtest import _prep
    from src.core.paths import DataPaths

    bad = tmp_path / "normalized" / "index_daily.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_text("not-parquet", encoding="utf-8")
    paths = DataPaths(root=tmp_path)
    assert _prep.load_index_daily_panel(paths) is None


def test_prepare_run_when_index_daily_missing(tmp_path) -> None:
    import argparse
    from dataclasses import replace

    from src.cli.commands.backtest._prep import prepare_run
    from src.cli.context import build_backtest_context
    from src.core.paths import DataPaths

    args = argparse.Namespace(
        model="sticky.mom60_raw",
        start="2026-01-02",
        end="2026-01-09",
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        stress_grid=False,
    )
    ctx = build_backtest_context(args)
    ctx = replace(ctx, paths=DataPaths(root=tmp_path))
    prep = prepare_run(ctx)
    assert getattr(prep.engine, "championship_sleeves", None) is None


def test_prepare_run_regime_build_failure_still_wires_sleeves(monkeypatch) -> None:
    import argparse

    from src.cli.commands.backtest._prep import prepare_run
    from src.cli.context import build_backtest_context
    from src.features.builder import FeatureBuilder

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("regime-fail")

    monkeypatch.setattr(FeatureBuilder, "build_regime_series", _boom)
    args = argparse.Namespace(
        model="sticky.mom60_raw",
        start="2026-01-02",
        end="2026-01-09",
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        stress_grid=False,
    )
    ctx = build_backtest_context(args)
    prep = prepare_run(ctx)
    sleeves = getattr(prep.engine, "championship_sleeves", None)
    assert isinstance(sleeves, dict)
    assert len(sleeves) > 0
    assert prep.regimes is None


def test_prepare_run_wires_championship_sleeves_from_index_daily() -> None:
    import argparse

    from src.cli.commands.backtest._prep import prepare_run
    from src.cli.context import build_backtest_context

    args = argparse.Namespace(
        model="sticky.mom60_raw",
        start="2026-01-02",
        end="2026-01-09",
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        stress_grid=False,
    )
    ctx = build_backtest_context(args)
    prep = prepare_run(ctx)
    sleeves = getattr(prep.engine, "championship_sleeves", None)
    assert isinstance(sleeves, dict)
    assert len(sleeves) > 0


def test_p27_cli_gross_metrics_are_null_safe() -> None:
    import inspect

    from src.cli.commands.backtest import _core
    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    hook_src = inspect.getsource(_hook_mom60_raw)
    # The unavailable marker must survive to the artifact, never be coerced to 0
    assert "gross_viol_p27 is not None" in hook_src
    assert "effective_gross_max_p27 is not None" in hook_src
    assert "int(_exp_p27.gross_violation_count)" not in hook_src

    core_src = inspect.getsource(_core)
    assert "prefer_execution_gross_count" in core_src
    assert "_resolved_gvc is not None" in core_src
    assert "_exposure.effective_gross_max is not None" in core_src
