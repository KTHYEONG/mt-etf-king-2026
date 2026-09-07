"""One-command Phase 0 feasibility audit (measurement only, no alpha change)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Final

import polars as pl

from src.research.activation_state import ACTIVATION_STATE_IS_PRODUCTION_GATE as ACTIVATION_STATE_IS_PRODUCTION_GATE
from src.research.activation_state import activation_conditional_table as activation_conditional_table
from src.research.activation_state import classify_activation_series as classify_activation_series
from src.research.activation_state import classify_activation_state as classify_activation_state
from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE as AUDIT_REGIME_IS_ACTIVATION_GATE
from src.research.audit_regime import audit_regime_label as audit_regime_label
from src.research.decision_population import PopulationError, WindowPolicy, build_decision_population
from src.research.executable_oracle import executable_open_to_open_ceiling
from src.research.feasibility_metrics import (
    CaptureReport,
    GapReport,
    align_returns_by_entry_date,
    capture_intersection,
    label_comparator_compliance,
    peak_return_from_return_space_giveback,
    peak_return_from_wealth_giveback,
    return_space_gap,
)
from src.universe.manifest_validate import validate_deployment_manifest

CAPTURE_THRESHOLD: Final[float] = 0.50


def run_feasibility_audit(
    *,
    calendar_sessions: Sequence[date],
    panel: pl.DataFrame,
    horizon: int,
    output_dir: Path,
    oracle_opens: pl.DataFrame,
    strategy_windows: Mapping[str, pl.DataFrame],
    comparator_gross: Mapping[str, Mapping[str, float]],
    champion_gross_max: float,
    p38_terminal_by_start: Mapping[date, float] | None = None,
    manifest: frozenset[str] | None = None,
    enforce_horizon_36: bool = True,
    kospi_mom60_by_date: Mapping[date, float | None] | None = None,
) -> dict[str, object]:
    if enforce_horizon_36 and horizon != 36:
        raise PopulationError(f"feasibility audit requires horizon=36, got {horizon}")
    calendar_pop = build_decision_population(calendar_sessions=calendar_sessions, panel=panel, horizon=horizon, policy=WindowPolicy.CALENDAR_NAIVE)
    panel_pop = build_decision_population(calendar_sessions=calendar_sessions, panel=panel, horizon=horizon, policy=WindowPolicy.PANEL_NAIVE)
    executable_pop = build_decision_population(calendar_sessions=calendar_sessions, panel=panel, horizon=horizon, policy=WindowPolicy.EXECUTABLE)
    ceiling = executable_open_to_open_ceiling(oracle_opens, executable_pop.windows)
    oracle_by_decision = {row["decision_date"]: float(row["ceil_exec_raw"]) for row in ceiling.iter_rows(named=True)}
    oracle_seq = tuple(oracle_by_decision[w.decision_date] for w in executable_pop.windows)
    strategies: dict[str, dict[str, object]] = {}
    sticky_aligned: tuple[float, ...] | None = None
    for name, frame in strategy_windows.items():
        starts = list(frame["window_start"].to_list())
        terminal = [float(v) for v in frame["terminal_return"].to_list()]
        aligned = align_returns_by_entry_date(starts=starts, returns=terminal, population=executable_pop)
        if "peak_return" in frame.columns:
            peak_raw = [float(v) for v in frame["peak_return"].to_list()]
            peak = align_returns_by_entry_date(starts=starts, returns=peak_raw, population=executable_pop)
        elif "wealth_giveback" in frame.columns:
            wealth_raw = [float(v) for v in frame["wealth_giveback"].to_list()]
            wealth = align_returns_by_entry_date(starts=starts, returns=wealth_raw, population=executable_pop)
            peak = tuple(peak_return_from_wealth_giveback(t, g) for t, g in zip(aligned, wealth, strict=True))
        elif "giveback" in frame.columns:
            giveback_raw = [float(v) for v in frame["giveback"].to_list()]
            giveback = align_returns_by_entry_date(starts=starts, returns=giveback_raw, population=executable_pop)
            peak = tuple(peak_return_from_return_space_giveback(t, g) for t, g in zip(aligned, giveback, strict=True))
        else:
            peak = aligned
        capture: CaptureReport = capture_intersection(aligned, oracle_seq, threshold=CAPTURE_THRESHOLD)
        gap: GapReport = return_space_gap(oracle_seq, aligned, peak)
        if name == "sticky.mom60_raw":
            sticky_aligned = aligned
        strategies[name] = {"n": len(aligned), "capture_rate": capture.rate, "mean_selection_timing": gap.mean_selection_timing, "mean_giveback": gap.mean_giveback, "mean_total": gap.mean_total}
    b1_gross = comparator_gross["baseline.mom20_top1"]
    b1_role = label_comparator_compliance(effective_gross_max=float(b1_gross["effective_gross_max"]), gross_violation_count=int(b1_gross["gross_violation_count"]), champion_gross_max=float(champion_gross_max))
    p38_aligned = None if p38_terminal_by_start is None else align_returns_by_entry_date(starts=list(p38_terminal_by_start.keys()), returns=list(p38_terminal_by_start.values()), population=executable_pop)
    p38_status = "MISSING_ARTIFACT" if p38_terminal_by_start is None else "AVAILABLE"
    p38_mean = (sum(p38_aligned) / len(p38_aligned)) if p38_aligned else None
    if p38_aligned is not None:
        p38_capture: CaptureReport = capture_intersection(p38_aligned, oracle_seq, threshold=CAPTURE_THRESHOLD)
        p38_gap: GapReport = return_space_gap(oracle_seq, p38_aligned, p38_aligned)
        strategies["adaptive_specialists.p38"] = {
            "n": len(p38_aligned),
            "capture_rate": p38_capture.rate,
            "mean_selection_timing": p38_gap.mean_selection_timing,
            "mean_giveback": p38_gap.mean_giveback,
            "mean_total": p38_gap.mean_total,
        }
    present = frozenset(panel["ticker"].unique().to_list()) if "ticker" in panel.columns else frozenset()
    manifest_result = validate_deployment_manifest(manifest=manifest, present_tickers=present, issuer_by_ticker=dict.fromkeys(present, "UNKNOWN"), allowed_issuers=frozenset())
    output_dir.mkdir(parents=True, exist_ok=True)
    populations = {"calendar_naive": calendar_pop, "panel_naive": panel_pop, "executable": executable_pop}
    md_lines = ["# Window Population Reconciliation", "", "| Policy | Raw Sessions | Eligible Windows | Excluded | Reason |", "| --- | --- | --- | --- | --- |"]
    for policy_name, pop in populations.items():
        excluded = len(pop.exclusions) if pop.exclusions else len(pop.phantom)
        reason = ", ".join(sorted({e.reason for e in pop.exclusions})) or "phantoms kept"
        md_lines.append(f"| {policy_name} | {len(calendar_sessions)} | {pop.n_windows} | {excluded} | {reason} |")
    (output_dir / "window_population_reconciliation.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    metrics = {"horizon": horizon, "b1_role": b1_role, "p38_status": p38_status, "p38_mean_terminal": p38_mean, "manifest_status": manifest_result.status, "audit_regime_is_activation_gate": AUDIT_REGIME_IS_ACTIVATION_GATE, "activation_state_is_production_gate": ACTIVATION_STATE_IS_PRODUCTION_GATE, "activation_as_of": None, "audit_regime_unlabeled_fallback": audit_regime_label(mom60=None, mom20=None, rv20=None, dd60=None), "strategies": strategies, "populations": {k: {"eligible_windows": v.n_windows, "excluded": len(v.exclusions)} for k, v in populations.items()}}
    if kospi_mom60_by_date is not None and len(executable_pop.windows) > 0:
        last_decision = executable_pop.windows[-1].decision_date
        pit_map = {day: value for day, value in kospi_mom60_by_date.items() if day <= last_decision}
        snap = classify_activation_state(decision_date=last_decision, mom60_by_date=pit_map)
        metrics["activation_as_of"] = {"date": last_decision.isoformat(), "state": str(snap.state), "mom60": snap.mom60, "q1": snap.q1, "q2": snap.q2, "n_history": snap.n_history}
    if kospi_mom60_by_date is not None and sticky_aligned is not None:
        snaps = classify_activation_series(kospi_mom60_by_date)
        state_by_decision = {item.as_of: str(item.state) for item in snaps}
        activation_states = [state_by_decision.get(w.decision_date, "UNCERTAIN") for w in executable_pop.windows]
        activation_table = activation_conditional_table(states=activation_states, terminal_returns=sticky_aligned, oracle_returns=oracle_seq)
        activation_table.write_csv(output_dir / "p27_regime_activation_table.csv")
    (output_dir / "2026_championship_feasibility_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    table_rows = [{"strategy": name, "metric": metric, "value": str(value)} for name, vals in strategies.items() for metric, value in vals.items()]
    pl.DataFrame(table_rows, schema={"strategy": pl.String, "metric": pl.String, "value": pl.String}, strict=False).write_csv(output_dir / "2026_championship_feasibility_tables.csv")
    return {"b1_role": b1_role, "p38_status": p38_status, "p38_mean_terminal": p38_mean, "manifest_status": manifest_result.status, "strategies": strategies}
