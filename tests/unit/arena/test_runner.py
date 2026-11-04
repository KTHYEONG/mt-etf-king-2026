import json
import logging
import runpy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena import runner
from src.arena.contest import EditionSpec, load_edition
from src.arena.evaluation import EvalConfig, evaluate
from src.arena.panel import PanelConfig
from src.arena.research import Evidence, ResearchConfig, Selection, load_research_config
from src.arena.runner import ResearchReport, StrategyRow, main, run_research, write_report
from src.arena.simulate import load_sim_config
from src.arena.strategies import candidate_grid, baselines, load_strategies_config
from src.arena.universe import load_issuer_table

NOW = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
EVAL = EvalConfig(10, 10, 200, 7, 0.95, (1.1, 1.3, 1.5, 2.0, 2.3, 3.0), 40)
SIM = load_sim_config()
ISSUERS = load_issuer_table()
EDITION = load_edition(2026)
BASE = load_strategies_config()
CFG = replace(
    BASE,
    horizons=(2, 3),
    vol_window=3,
    er_window=4,
    er_min=0.4,
    band=0.05,
    min_listed_sessions=4,
    z_window=10,
    z_threshold=1.0,
    flow_window=(5,),
    rotation_hold=(5,),
    switch_sleeves=("kospi200",),
    chop_filter=(False, True),
    sleeves={"kospi200": ("069500", "122630"), "kosdaq150": ("229200", "233740")},
    inverse={"kospi200": "252670", "kosdaq150": "251340"},
    baseline_hold=("122630",),
    coin_pair=("122630", "252670"),
    coin_seed=1,
)
RESEARCH = replace(
    load_research_config(),
    discovery_start=date(2024, 1, 2),
    holdout_start=date(2024, 9, 2),
    regime_min_sessions=20,
)
NAMES = {
    "069500": "KODEX 200",
    "122630": "KODEX 레버리지",
    "229200": "KODEX 코스닥150",
    "233740": "KODEX 코스닥150레버리지",
    "252670": "KODEX 200선물인버스2X",
    "251340": "KODEX 코스닥150선물인버스",
}
SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "ticker": pl.String(),
    "name": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "trading_value": pl.Int64(),
    "nav": pl.Float64(),
    "shares_outstanding": pl.Int64(),
    "underlying_index_name": pl.String(),
}
CANDIDATE_IDS = [s.strategy_id for s in candidate_grid(CFG)]
BASELINE_IDS = [s.strategy_id for s in baselines(CFG)]


def _bdays(n: int) -> list[date]:
    days: list[date] = []
    day = date(2024, 1, 1)
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _silver(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(20261105)
    days = _bdays(n)
    rows: list[dict[str, Any]] = []
    for ticker, name in NAMES.items():
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n))
        shares = np.maximum(1000 + np.cumsum(rng.integers(-40, 41, n)), 1)
        for day, close, count in zip(days, closes.tolist(), shares.tolist(), strict=True):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "name": name,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000,
                    "trading_value": 10**9,
                    "nav": close,
                    "shares_outstanding": count,
                    "underlying_index_name": "IDX",
                }
            )
    return pl.DataFrame(rows, schema=SCHEMA)


def _run(
    tmp_path: Path,
    rows: pl.DataFrame | None = None,
    edition: EditionSpec = EDITION,
    eval_cfg: EvalConfig = EVAL,
    run_id: str = "run",
) -> ResearchReport:
    return run_research(
        _silver() if rows is None else rows,
        edition=edition,
        issuers=ISSUERS,
        track="autonomous",
        strategies_cfg=CFG,
        research_cfg=RESEARCH,
        sim_cfg=SIM,
        eval_cfg=eval_cfg,
        panel_cfg=PANEL_CFG,
        registry=tmp_path / "registry.jsonl",
        recorded_at=NOW,
        run_id=run_id,
    )


def test_run_pending_without_band_has_a_row_per_strategy(tmp_path: Path) -> None:
    report = _run(tmp_path)
    assert report.status == "PENDING_W"
    assert report.band is None and report.selection is None
    assert [row.strategy_id for row in report.rows] == [*CANDIDATE_IDS, *BASELINE_IDS]
    assert all(row.verdict is None for row in report.rows)
    assert [row.baseline for row in report.rows] == [False] * len(CANDIDATE_IDS) + [True] * len(BASELINE_IDS)
    assert report.data_first == RESEARCH.discovery_start
    assert report.data_last == _bdays(300)[-1]


def test_run_gated_with_band_selects_a_labelled_candidate(tmp_path: Path) -> None:
    report = _run(tmp_path, edition=replace(EDITION, prize_line=(1.1, 1.3)))
    assert report.status == "GATED"
    assert report.band == (1.1, 1.3)
    for row in report.rows:
        if row.baseline:
            assert row.verdict is None
        else:
            assert row.verdict is not None
            assert row.verdict.strategy_id == row.strategy_id
    assert report.selection is not None
    assert report.selection.strategy_id in CANDIDATE_IDS
    assert report.selection.label in ("VALIDATED", "UNVALIDATED")


def test_band_is_added_to_the_exceedance_grid(tmp_path: Path) -> None:
    coarse = replace(EVAL, exceedance_grid=(1.5, 2.0, 3.0))
    report = _run(tmp_path, edition=replace(EDITION, prize_line=(1.1, 1.3)), eval_cfg=coarse)
    stress = report.rows[0].evidence.holdout_stress
    assert stress is not None
    assert 1.1 in stress.exceedance.thresholds and 1.3 in stress.exceedance.thresholds


def test_contest_window_multiple_is_reported_only_when_the_window_has_sessions(tmp_path: Path) -> None:
    covered = replace(EDITION, start=date(2024, 1, 2), end=date(2024, 12, 31))
    inside = _run(tmp_path / "inside", edition=covered)
    assert all(row.contest_window_multiple is not None for row in inside.rows)
    outside = _run(tmp_path / "outside")
    assert all(row.contest_window_multiple is None for row in outside.rows)


def test_candidates_and_baselines_register_in_separate_families(tmp_path: Path) -> None:
    report = _run(tmp_path)
    families = {json.loads(line)["strategy_family"] for line in (tmp_path / "registry.jsonl").read_text().splitlines()}
    assert any(f.startswith("autonomous/") for f in families)
    assert any(f.startswith("autonomous-baseline/") for f in families)
    candidates = [row for row in report.rows if not row.baseline]
    for row in candidates:
        assert row.evidence.holdout_dn is not None
        assert row.evidence.holdout_dn.edge.n_trials == len(CANDIDATE_IDS)
    first = candidates[0].evidence.holdout_dn
    assert first is not None
    assert first.edge.p_value_adjusted == pytest.approx(1.0 - (1.0 - first.edge.p_value) ** len(CANDIDATE_IDS))
    registered = [row for row in report.rows if row.baseline and row.evidence.holdout_dn is not None]
    for row in registered:
        assert row.evidence.holdout_dn is not None
        assert row.evidence.holdout_dn.edge.n_trials == len(registered)


def test_non_sponsor_row_for_a_used_ticker_fails_closed(tmp_path: Path) -> None:
    rows = _silver()
    first = rows["date"].min()
    rows = rows.with_columns(
        pl.when((pl.col("ticker") == "122630") & (pl.col("date") == first))
        .then(pl.lit("WON 122630 fund"))
        .otherwise(pl.col("name"))
        .alias("name")
    )
    with pytest.raises(ValueError, match="122630"):
        _run(tmp_path, rows=rows)
    assert not (tmp_path / "registry.jsonl").exists()


def test_missing_ticker_fails_closed(tmp_path: Path) -> None:
    rows = _silver().filter(pl.col("ticker") != "252670")
    with pytest.raises(ValueError, match="missing"):
        _run(tmp_path, rows=rows)
    assert not (tmp_path / "registry.jsonl").exists()


def test_empty_rows_and_unknown_track_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _run(tmp_path, rows=_silver().head(0))
    with pytest.raises(ValueError, match="is not a track"):
        run_research(
            _silver(),
            edition=EDITION,
            issuers=ISSUERS,
            track="nosuchtrack",
            strategies_cfg=CFG,
            research_cfg=RESEARCH,
            sim_cfg=SIM,
            eval_cfg=EVAL,
            panel_cfg=PANEL_CFG,
            registry=tmp_path / "registry.jsonl",
            recorded_at=NOW,
            run_id="x",
        )


def _manual_report(rows: tuple[StrategyRow, ...], **changes: Any) -> ResearchReport:
    base = {
        "run_id": "manual",
        "recorded_at": NOW,
        "edition": 2026,
        "track": "autonomous",
        "data_first": date(2024, 1, 2),
        "data_last": date(2024, 12, 31),
        "band": None,
        "status": "PENDING_W",
        "rows": rows,
        "selection": None,
    }
    base.update(changes)
    return ResearchReport(**base)


def _empty_evidence(strategy_id: str) -> Evidence:
    return Evidence(strategy_id, "trend_switch", None, None, None, None, {}, ("holdout_dn: short",))


def test_report_files_are_written_atomically_and_overwrite_cleanly(tmp_path: Path) -> None:
    report = _run(tmp_path, edition=replace(EDITION, prize_line=(1.1, 1.3)))
    out = tmp_path / "out"
    json_path, md_path = write_report(report, out)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "GATED"
    assert payload["band"] == [1.1, 1.3]
    assert len(payload["rows"]) == len(report.rows)
    text = md_path.read_text(encoding="utf-8")
    for strategy_id in [*CANDIDATE_IDS, *BASELINE_IDS]:
        assert strategy_id in text
    assert "Contest-window multiple is descriptive and in-sample" in text
    assert "KOSDAQ150 inverse leg (251340) is 1x" in text
    assert "PASS" in text or "FAIL" in text
    assert str(tmp_path / "registry.jsonl") in text
    write_report(report, out)
    assert md_path.read_text(encoding="utf-8") == text


def test_pending_report_renders_empty_views_and_pending_selection(tmp_path: Path) -> None:
    rows = (
        StrategyRow(
            "trend_switch:kospi200:plain",
            "trend_switch",
            False,
            _empty_evidence("trend_switch:kospi200:plain"),
            None,
            None,
        ),
    )
    report = _manual_report(rows)
    json_path, md_path = write_report(report, tmp_path / "pending")
    text = md_path.read_text(encoding="utf-8")
    assert "PENDING_W: prize line undecided (overview §8 O1)" in text
    assert "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | PENDING |" in text
    assert "Candidate holdout n_trials: n/a" in text
    assert json.loads(json_path.read_text(encoding="utf-8"))["selection"] is None


def test_gated_report_without_a_selection_says_so(tmp_path: Path) -> None:
    baseline_row = StrategyRow("hold:122630", "baseline_hold", True, _empty_evidence("hold:122630"), None, 1.25)
    report = _manual_report(
        (baseline_row,),
        band=(1.1, 1.3),
        status="GATED",
        selection=Selection("trend_switch:kospi200:plain", "UNVALIDATED", 0.02),
    )
    _, md_path = write_report(report, tmp_path / "gated")
    assert "Selection: trend_switch:kospi200:plain (UNVALIDATED), lift at band low 0.0200" in md_path.read_text(
        encoding="utf-8"
    )
    none_report = _manual_report((baseline_row,), band=(1.1, 1.3), status="GATED", selection=None)
    _, md_none = write_report(none_report, tmp_path / "none")
    text = md_none.read_text(encoding="utf-8")
    assert "Selection: none (no candidate has a defined lift at the band)" in text
    assert "| 1.2500 |" not in text and "BASELINE" in text


def _write_silver(root: Path, frame: pl.DataFrame) -> None:
    (root / "normalized").mkdir(parents=True)
    frame.write_parquet(root / "normalized" / "etf_daily.parquet")


def test_main_runs_end_to_end_and_writes_under_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_silver(tmp_path, _silver())
    monkeypatch.setattr(runner, "load_strategies_config", lambda: CFG)
    monkeypatch.setattr(runner, "load_research_config", lambda: RESEARCH)
    monkeypatch.setattr(runner, "load_eval_config", lambda: EVAL)

    assert main(["research", "--data-root", str(tmp_path), "--run-id", "run_a"]) == 0
    printed = capsys.readouterr().out.strip()
    report_md = tmp_path / "results" / "arena" / "run_a" / "report.md"
    assert printed == str(report_md)
    assert report_md.is_file()
    assert (tmp_path / "results" / "arena" / "registry.jsonl").is_file()
    assert (tmp_path / "results" / "arena" / "run_a" / "report.json").is_file()


def test_main_default_run_id_uses_utc_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_silver(tmp_path, _silver())
    monkeypatch.setattr(runner, "load_strategies_config", lambda: CFG)
    monkeypatch.setattr(runner, "load_research_config", lambda: RESEARCH)
    monkeypatch.setattr(runner, "load_eval_config", lambda: EVAL)
    assert main(["research", "--data-root", str(tmp_path)]) == 0
    run_dirs = [p.name for p in (tmp_path / "results" / "arena").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert len(run_dirs[0]) == len("20261105T090000Z")


def test_main_without_silver_table_logs_and_returns_one(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="src.arena.runner"):
        assert main(["research", "--data-root", str(tmp_path), "--run-id", "x"]) == 1
    assert "[SYS] research status=fail error=" in caplog.text


def test_module_entry_point_exits_with_main_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["src.arena", "research", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("src.arena.__main__", run_name="__main__")
    assert exit_info.value.code == 1
