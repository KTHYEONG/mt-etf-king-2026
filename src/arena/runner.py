"""Research runner: registered candidates and baselines, causal evidence, prize-line gate and report files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from src.arena.contest import EditionSpec, load_edition
from src.arena.evaluation import EvalConfig, EvaluationReport, load_eval_config, report_to_dict
from src.arena.panel import Panel, PanelConfig, PanelSourceError, build_panel, load_etf_silver, load_panel_config
from src.arena.proxy import ProxyCalibration, load_proxy_config, load_stock_silver, splice_proxy
from src.arena.registry import count_trials
from src.arena.research import (
    Evidence,
    GateVerdict,
    ResearchConfig,
    Selection,
    Strategy,
    gate,
    gather_evidence,
    leave_one_year_out,
    load_research_config,
    select,
)
from src.arena.simulate import SimConfig, load_sim_config, simulate
from src.arena.strategies import (
    StrategiesConfig,
    baselines,
    candidate_grid,
    load_strategies_config,
    strategy_tickers,
)
from src.arena.universe import IssuerTable, load_issuer_table, resolve_issuer
from src.core.atomic_io import atomic_write_text
from src.core.config import ConfigError
from src.core.paths import DataPaths

logger = logging.getLogger(__name__)

Status = Literal["GATED", "PENDING_W"]
_RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"
_REGISTRY_NAME = "registry.jsonl"
_BASELINE_SUFFIX = "-baseline"
_STRESS_LOW = 2.0
_STRESS_HIGH = 2.3
_NOTE_CONTEST = "Contest-window multiple is descriptive and in-sample; it is not a selection criterion."
_NOTE_HAIRCUT = (
    "haircut re-applied with the final family size K at report time (registry lines keep sequential counts)."
)
_NOTE_KOSDAQ = "The KOSDAQ150 inverse leg (251340) is 1x, so the KOSDAQ pair is asymmetric."
_NOTE_PROXY = (
    "Pre-listing rows of proxied ETFs are synthetic: daily-reset multiple x KRX underlying return minus drag, "
    "with capacity = underlying traded value / |multiple|. They are evidence about the vehicle, "
    "not about the listed product's history."
)


@dataclass(frozen=True)
class StrategyRow:
    strategy_id: str
    family: str
    baseline: bool
    evidence: Evidence
    verdict: GateVerdict | None
    contest_window_multiple: float | None


@dataclass(frozen=True)
class ResearchReport:
    run_id: str
    recorded_at: datetime
    edition: int
    track: str
    data_first: date
    data_last: date
    band: tuple[float, float] | None
    status: Status
    rows: tuple[StrategyRow, ...]
    selection: Selection | None
    data_variant: Literal["listed", "proxy"] = "listed"
    proxy: tuple[ProxyCalibration, ...] = ()
    trial_tags: Mapping[str, str] = field(default_factory=dict)


def _check_sponsors(rows: pl.DataFrame, tickers: frozenset[str], issuers: IssuerTable, edition: EditionSpec) -> None:
    used = rows.filter(pl.col("ticker").is_in(sorted(tickers))).select("ticker", "name").unique()
    offenders = sorted(
        {
            ticker
            for ticker, name in used.iter_rows()
            if resolve_issuer(name, issuers, edition.sponsors).status != "SPONSOR"
        }
    )
    if offenders:
        raise ValueError(f"tickers with non-sponsor or unknown issuer rows: {offenders}")


def _restate(view: str, report: EvaluationReport, family: str, registry: Path) -> EvaluationReport:
    """Re-apply the Sidak haircut with the registry's final family size K for `family/view`."""
    trials = count_trials(registry, f"{family}/{view}")
    edge = report.edge
    adjusted = replace(edge, n_trials=trials, p_value_adjusted=1.0 - (1.0 - edge.p_value) ** trials)
    return replace(report, edge=adjusted)


def _restate_evidence(evidence: Evidence, family: str, registry: Path) -> Evidence:
    def maybe(view: str, report: EvaluationReport | None) -> EvaluationReport | None:
        return None if report is None else _restate(view, report, family, registry)

    return replace(
        evidence,
        discovery_dn=maybe("discovery_dn", evidence.discovery_dn),
        holdout_dn=maybe("holdout_dn", evidence.holdout_dn),
        full_dn=maybe("full_dn", evidence.full_dn),
        holdout_stress=maybe("holdout_stress", evidence.holdout_stress),
        regimes={
            regime: _restate(f"regime_{regime}", report, family, registry)
            for regime, report in evidence.regimes.items()
        },
    )


def _contest_multiple(strategy: Strategy, panel: Panel, sim_cfg: SimConfig, edition: EditionSpec) -> float | None:
    targets = strategy.targets(panel)
    if targets.is_empty():
        return None
    result = simulate(panel, targets, sim_cfg, capital_krw=edition.initial_capital_krw)
    in_window = np.array([edition.start <= day <= edition.end for day in result.sessions], dtype=bool)
    picked = result.net_returns[in_window]
    if picked.size == 0:
        return None
    return float(np.prod(1.0 + picked))


def run_research(
    rows: pl.DataFrame,
    *,
    edition: EditionSpec,
    issuers: IssuerTable,
    track: str,
    strategies_cfg: StrategiesConfig,
    research_cfg: ResearchConfig,
    sim_cfg: SimConfig,
    eval_cfg: EvalConfig,
    panel_cfg: PanelConfig,
    registry: Path,
    recorded_at: datetime,
    run_id: str,
    trial_tags: Mapping[str, str] | None = None,
    proxy: tuple[ProxyCalibration, ...] = (),
) -> ResearchReport:
    """Evaluate every registered candidate and baseline and gate the candidates when the prize line is set."""
    if track not in edition.tracks:
        raise ValueError(f"track {track!r} is not a track of edition {edition.edition}")
    candidates = candidate_grid(strategies_cfg)
    references = baselines(strategies_cfg)
    tickers = strategy_tickers((*candidates, *references)) | {research_cfg.regime_reference}
    _check_sponsors(rows, tickers, issuers, edition)
    if rows.is_empty():
        raise ValueError("rows must not be empty")
    end: date = rows.select(pl.col("date").max()).item()
    panel = build_panel(rows, sorted(tickers), research_cfg.discovery_start, end, panel_cfg)
    if panel.missing_tickers:
        raise ValueError(f"tickers missing from rows: {list(panel.missing_tickers)}")

    band = edition.prize_line
    eval_used = eval_cfg
    if band is not None:
        grid = tuple(sorted(set(eval_cfg.exceedance_grid) | set(band)))
        eval_used = replace(eval_cfg, exceedance_grid=grid)

    jobs = [(strategy, False) for strategy in candidates] + [(strategy, True) for strategy in references]
    evidence_rows: list[tuple[Strategy, bool, Evidence, float | None]] = []
    for index, (strategy, baseline) in enumerate(jobs, start=1):
        logger.info("[ALGO] research strategy=%s i=%d/%d", strategy.strategy_id, index, len(jobs))
        evidence = gather_evidence(
            strategy,
            panel,
            sim_cfg,
            eval_used,
            research_cfg,
            track=f"{track}{_BASELINE_SUFFIX}" if baseline else track,
            capital_krw=edition.initial_capital_krw,
            registry=registry,
            recorded_at=recorded_at,
            trial_tags=trial_tags,
        )
        multiple = _contest_multiple(strategy, panel, sim_cfg, edition)
        evidence_rows.append((strategy, baseline, evidence, multiple))

    evidence_rows = [
        (
            strategy,
            baseline,
            _restate_evidence(evidence, f"{track}{_BASELINE_SUFFIX}" if baseline else track, registry),
            multiple,
        )
        for strategy, baseline, evidence, multiple in evidence_rows
    ]
    verdicts: dict[str, GateVerdict] = {}
    selection: Selection | None = None
    status: Status = "PENDING_W"
    if band is not None:
        status = "GATED"
        candidate_evidence = [ev for _, baseline, ev, _ in evidence_rows if not baseline]
        for evidence in candidate_evidence:
            family = [other for other in candidate_evidence if other.family == evidence.family]
            verdicts[evidence.strategy_id] = gate(evidence, research_cfg, band, family)
        selection = select(list(verdicts.values()))

    report_rows = tuple(
        StrategyRow(
            strategy_id=evidence.strategy_id,
            family=evidence.family,
            baseline=baseline,
            evidence=evidence,
            verdict=verdicts.get(evidence.strategy_id),
            contest_window_multiple=multiple,
        )
        for _, baseline, evidence, multiple in evidence_rows
    )
    return ResearchReport(
        run_id=run_id,
        recorded_at=recorded_at,
        edition=edition.edition,
        track=track,
        data_first=panel.sessions[0],
        data_last=panel.sessions[-1],
        band=band,
        status=status,
        rows=report_rows,
        selection=selection,
        data_variant="proxy" if proxy else "listed",
        proxy=proxy,
        trial_tags=dict(trial_tags or {}),
    )


def _optional(report: EvaluationReport | None) -> dict[str, Any] | None:
    return None if report is None else report_to_dict(report)


def _evidence_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "strategy_id": evidence.strategy_id,
        "family": evidence.family,
        "discovery_dn": _optional(evidence.discovery_dn),
        "holdout_dn": _optional(evidence.holdout_dn),
        "full_dn": _optional(evidence.full_dn),
        "holdout_stress": _optional(evidence.holdout_stress),
        "regimes": {regime: report_to_dict(report) for regime, report in evidence.regimes.items()},
        "missing": list(evidence.missing),
    }


def _verdict_dict(verdict: GateVerdict | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "strategy_id": verdict.strategy_id,
        "passed": verdict.passed,
        "criteria": [asdict(criterion) for criterion in verdict.criteria],
        "lift_at_band": None if verdict.lift_at_band is None else list(verdict.lift_at_band),
    }


def _optional_date(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def _calibration_dict(calibration: ProxyCalibration) -> dict[str, Any]:
    payload = asdict(calibration)
    payload["listed_first"] = calibration.listed_first.isoformat()
    payload["proxy_first"] = _optional_date(calibration.proxy_first)
    payload["proxy_last"] = _optional_date(calibration.proxy_last)
    return payload


def _report_dict(report: ResearchReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "recorded_at": report.recorded_at.isoformat(),
        "edition": report.edition,
        "track": report.track,
        "data_first": report.data_first.isoformat(),
        "data_last": report.data_last.isoformat(),
        "band": None if report.band is None else list(report.band),
        "status": report.status,
        "data_variant": report.data_variant,
        "trial_tags": dict(report.trial_tags),
        "proxy": [_calibration_dict(calibration) for calibration in report.proxy],
        "rows": [
            {
                "strategy_id": row.strategy_id,
                "family": row.family,
                "baseline": row.baseline,
                "evidence": _evidence_dict(row.evidence),
                "verdict": _verdict_dict(row.verdict),
                "contest_window_multiple": row.contest_window_multiple,
            }
            for row in report.rows
        ],
        "selection": None
        if report.selection is None
        else {
            "strategy_id": report.selection.strategy_id,
            "label": report.selection.label,
            "lift_at_band_lo": report.selection.lift_at_band_lo,
        },
    }


def _p_text(report: EvaluationReport | None) -> str:
    return "n/a" if report is None else f"{report.edge.p_value_adjusted:.4f}"


def _exceedance_text(report: EvaluationReport | None, threshold: float) -> str:
    if report is None or threshold not in report.exceedance.thresholds:
        return "n/a"
    index = report.exceedance.thresholds.index(threshold)
    return f"{report.exceedance.p_strategy[index]:.3f}/{report.exceedance.p_twin[index]:.3f}"


def _verdict_text(row: StrategyRow) -> str:
    if row.baseline:
        return "BASELINE"
    if row.verdict is None:
        return "PENDING"
    return "PASS" if row.verdict.passed else "FAIL"


def _row_line(row: StrategyRow) -> str:
    evidence = row.evidence
    lyo = leave_one_year_out(evidence.full_dn).value
    extreme = evidence.regimes.get("extreme")
    multiple = "n/a" if row.contest_window_multiple is None else f"{row.contest_window_multiple:.3f}"
    cells = [
        row.strategy_id,
        row.family,
        _p_text(evidence.discovery_dn),
        _p_text(evidence.holdout_dn),
        "n/a" if lyo is None else f"{lyo:.6f}",
        _exceedance_text(evidence.holdout_stress, _STRESS_LOW),
        _exceedance_text(evidence.holdout_stress, _STRESS_HIGH),
        _exceedance_text(extreme, _STRESS_HIGH),
        multiple,
        _verdict_text(row),
    ]
    return "| " + " | ".join(cells) + " |"


def _selection_text(report: ResearchReport) -> str:
    if report.selection is not None:
        sel = report.selection
        return f"Selection: {sel.strategy_id} ({sel.label}), lift at band low {sel.lift_at_band_lo:.4f}"
    if report.status == "PENDING_W":
        return "PENDING_W: prize line undecided (overview §8 O1)"
    return "Selection: none (no candidate has a defined lift at the band)"


def _variant_text(report: ResearchReport) -> str:
    if report.data_variant == "listed":
        return "listed"
    tag = report.trial_tags.get("panel", "n/a")
    execution = report.trial_tags.get("execution", "n/a")
    return f"proxy, tag {tag}, execution {execution}"


def _proxy_row(calibration: ProxyCalibration) -> str:
    first, last = _optional_date(calibration.proxy_first), _optional_date(calibration.proxy_last)
    cells = [
        calibration.etf,
        calibration.underlying,
        f"{calibration.multiple:g}",
        f"{first or 'n/a'} to {last or 'n/a'}",
        str(calibration.proxy_sessions),
        calibration.listed_first.isoformat(),
        str(calibration.overlap_sessions),
        f"{calibration.close_corr:.6f}",
        f"{calibration.close_te_bps:.2f}",
        f"{calibration.close_mean_bps:.2f}",
        f"{calibration.open_te_bps:.2f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _proxy_section(calibrations: tuple[ProxyCalibration, ...]) -> list[str]:
    return [
        "## Proxy",
        "",
        "| etf | underlying | multiple | proxy span | proxy sessions | listed since | overlap | "
        "close corr | close TE bp | close mean bp | open TE bp |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *(_proxy_row(calibration) for calibration in calibrations),
        "",
    ]


def _markdown(report: ResearchReport, registry_path: Path) -> str:
    band = "undecided (overview §8 O1)" if report.band is None else f"[{report.band[0]}, {report.band[1]}]"
    trials = [
        row.evidence.holdout_dn.edge.n_trials
        for row in report.rows
        if not row.baseline and row.evidence.holdout_dn is not None
    ]
    proxied = report.data_variant == "proxy"
    notes = [_NOTE_CONTEST, _NOTE_KOSDAQ, _NOTE_HAIRCUT, *([_NOTE_PROXY] if proxied else [])]
    lines = [
        f"# Arena research {report.run_id}",
        "",
        f"- Status: {report.status}",
        f"- Band: {band}",
        f"- Data: {report.data_first.isoformat()} to {report.data_last.isoformat()}",
        f"- Data variant: {_variant_text(report)}",
        f"- Registry: {registry_path}",
        f"- Candidate holdout n_trials: {max(trials) if trials else 'n/a'}",
        "",
        "| id | family | discovery p_adj | holdout p_adj | leave-one-year-out min | "
        f"stress P(M>={_STRESS_LOW}) strategy/twin | stress P(M>={_STRESS_HIGH}) strategy/twin | "
        f"extreme P(M>={_STRESS_HIGH}) strategy/twin | contest-window multiple | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_row_line(row) for row in report.rows)
    lines.extend(["", _selection_text(report), ""])
    if proxied:
        lines.extend(_proxy_section(report.proxy))
    lines.extend(["## Notes", *(f"- {note}" for note in notes), ""])
    return "\n".join(lines)


def write_report(report: ResearchReport, out_dir: Path) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` atomically into `out_dir`; returns (json_path, md_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    payload = json.dumps(_report_dict(report), ensure_ascii=False, indent=2, allow_nan=False)
    atomic_write_text(json_path, payload + "\n")
    atomic_write_text(md_path, _markdown(report, out_dir.parent / _REGISTRY_NAME))
    return json_path, md_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.arena")
    commands = parser.add_subparsers(dest="command", required=True)
    research = commands.add_parser("research", help="run the pre-registered research grid")
    research.add_argument("--data-root", type=Path, required=True)
    research.add_argument("--edition", type=int, default=2026)
    research.add_argument("--track", default="autonomous")
    research.add_argument("--run-id", default=None)
    research.add_argument("--proxy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `research` subcommand; returns 0 with the report path on stdout, or 1 after logging the failure."""
    args = _parser().parse_args(argv)
    recorded_at = datetime.now(UTC)
    run_id: str = args.run_id or recorded_at.strftime(_RUN_ID_FORMAT)
    try:
        edition = load_edition(args.edition)
        issuers = load_issuer_table()
        strategies_cfg = load_strategies_config()
        research_cfg = load_research_config()
        sim_cfg = load_sim_config()
        eval_cfg = load_eval_config()
        panel_cfg = load_panel_config()
        paths = DataPaths(args.data_root)
        rows = load_etf_silver(paths, panel_cfg)
        tags: Mapping[str, str] | None = None
        proxy: tuple[ProxyCalibration, ...] = ()
        if args.proxy:
            proxy_cfg = load_proxy_config()
            stock = load_stock_silver(paths, panel_cfg, sorted({product.underlying for product in proxy_cfg.products}))
            rows, proxy = splice_proxy(rows, stock, proxy_cfg)
            sim_cfg = replace(sim_cfg, execution=proxy_cfg.execution)
            tags = {"panel": proxy_cfg.tag, "execution": proxy_cfg.execution}
        report = run_research(
            rows,
            edition=edition,
            issuers=issuers,
            track=args.track,
            strategies_cfg=strategies_cfg,
            research_cfg=research_cfg,
            sim_cfg=sim_cfg,
            eval_cfg=eval_cfg,
            panel_cfg=panel_cfg,
            registry=paths.results("arena") / _REGISTRY_NAME,
            recorded_at=recorded_at,
            run_id=run_id,
            trial_tags=tags,
            proxy=proxy,
        )
        _, md_path = write_report(report, paths.results(f"arena/{run_id}"))
    except (ConfigError, PanelSourceError, ValueError) as exc:
        logger.error("[SYS] research status=fail error=%s", exc)
        return 1
    sys.stdout.write(f"{md_path}\n")
    return 0
