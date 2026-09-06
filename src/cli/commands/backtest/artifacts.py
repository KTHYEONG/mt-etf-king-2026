"""Backtest artifact attachment and emission (P4).

ADR-P36: the P36 attainability-artifact wiring that lived in
``_attach_p36_backtest_artifacts`` now runs here, renamed off the task id.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping

from src.cli.context import BacktestContext, BacktestResult

logger = logging.getLogger(__name__)


def attach_backtest_artifacts(ctx: BacktestContext, result: BacktestResult) -> Mapping[str, object]:
    """Enrich each cell's meta/summary with attainability artifacts.

    Returns a mapping of run_id to the enriched ``{"meta": ..., "summary": ...}``
    payloads for the emission step.
    """
    from src.tournament.attainability import attainability_curve, enrich_backtest_run_artifacts, window_opportunities

    artifacts: dict[str, object] = {}
    for bundle in result.cells:
        meta = dict(bundle.meta)
        summary = dict(bundle.summary)
        thresholds = [0.30, 0.40, 0.50, 0.60]
        _opps = window_opportunities([], [], {}, int(bundle.horizon))
        attainability_curve(_opps, thresholds)
        meta, summary = enrich_backtest_run_artifacts(
            meta,
            summary,
            calendar=ctx.calendar,
            panel=bundle.panel,
            engine=bundle.engine,
            model=bundle.model,
            case_config=bundle.case_config,
            rolling=bundle.rolling,
            horizon=int(bundle.horizon),
            shared_cache=bundle.shared_cache,
            leverage_allowed=bundle.leverage_allowed,
            inverse_allowed=bundle.inverse_allowed,
        )
        artifacts[bundle.run_id] = {"meta": meta, "summary": summary}
    return artifacts


def emit_backtest_artifacts(
    ctx: BacktestContext,
    result: BacktestResult,
    artifacts: Mapping[str, object],
) -> None:
    """Persist result files, forensics side files, traces and tail reports."""
    from src.reporting.results import write_backtest_result

    cells_by_id = {bundle.run_id: bundle for bundle in result.cells}
    for run_id, payload in artifacts.items():
        bundle = cells_by_id.get(run_id)
        if bundle is None:
            continue
        meta = payload["meta"] if isinstance(payload, dict) else bundle.meta
        summary = payload["summary"] if isinstance(payload, dict) else bundle.summary
        assert isinstance(meta, dict)
        assert isinstance(summary, dict)
        _write_success = False
        try:
            write_backtest_result(
                ctx.paths,
                run_id=run_id,
                meta=meta,
                summary=summary,
                daily=bundle.daily,
                trades=bundle.trades,
                windows=bundle.windows_df,
            )
            if bundle.daily is not None and bundle.trades is not None:
                logger.info(
                    f"[EVAL] artifacts run_id={run_id} daily_rows={bundle.daily.height} trade_rows={bundle.trades.height}"
                )
            _write_success = True
        except FileExistsError:
            logger.warning(f"[SYS] backtest result exists run_id={run_id} skipping overwrite")
            _write_success = False
        except Exception as exc2:
            logger.warning(f"[SYS] backtest result write failed run_id={run_id} error={exc2!r}")
            _write_success = False
        if _write_success and bundle.forensics_payload is not None:
            try:
                import json as _json_opt

                out_dir = ctx.paths.results(run_id) / "p25_optimization.json"
                out_dir.write_text(_json_opt.dumps(bundle.forensics_payload, indent=2), encoding="utf-8")
                logger.info(f"[EVAL] forensics optimizer wrote {out_dir}")
            except Exception as _e_opt_write:
                logger.warning(f"[EVAL] forensics write failed {_e_opt_write!r}")
        # trace write after successful result write
        if _write_success and ctx.trace:
            try:
                from src.core.trace import InMemoryTraceSink as _T2  # noqa: N814
                from src.reporting.trace_store import frames_from_sink, write_trace_artifacts

                _sink_for_write = bundle.trace_sink
                if _sink_for_write is None:
                    try:
                        _sink_for_write = _T2()
                    except Exception:
                        _sink_for_write = None
                if getattr(bundle.rolling, "backtest", None) is None and _sink_for_write is not None:
                    with contextlib.suppress(Exception):
                        # extra run with cell case_config and trace
                        _extra = bundle.engine.run(bundle.model, bundle.panel, bundle.case_config, trace=_sink_for_write)
                if _sink_for_write is not None:
                    try:
                        _sessions_df, _candidates_df, _gates_list = frames_from_sink(_sink_for_write)
                    except Exception:
                        import polars as _pl

                        _sessions_df = _pl.DataFrame({"decision_date": [], "n_universe": []})
                        _candidates_df = _pl.DataFrame({"decision_date": [], "ticker": []})
                        _gates_list = []
                    try:
                        dest = ctx.paths.trace(run_id)
                        write_trace_artifacts(dest, sessions=_sessions_df, candidates=_candidates_df, gates=_gates_list)
                    except OSError as _oe:
                        logger.warning(f"[SYS] trace write failed { _oe!r}")
                    except Exception as _e2:
                        logger.warning(f"[SYS] trace write failed {_e2!r}")
            except OSError as _oe_outer:
                logger.warning(f"[SYS] trace write failed {_oe_outer!r}")
            except Exception as _e_outer:
                logger.warning(f"[SYS] trace write failed {_e_outer!r}")
            # forensics wiring: tail_miss_report when --trace and --forensics
            if _write_success and ctx.forensics:
                try:
                    from src.reporting.tail_forensics import summarise_tail_miss_windows as _summ_tmf
                    from src.reporting.tail_forensics import write_tail_miss_report as _write_tmf

                    try:
                        _wdf_for = bundle.windows_df
                    except (AttributeError, NameError):
                        _wdf_for = None
                    try:
                        _sdf_for = _sessions_df
                    except NameError:
                        import polars as _pl_for2  # noqa: F401

                        _sdf_for = _pl_for2.DataFrame()
                    try:
                        _cdf_for = _candidates_df
                    except NameError:
                        import polars as _pl_for3  # noqa: F401

                        _cdf_for = _pl_for3.DataFrame()
                    if _wdf_for is not None:
                        if getattr(_wdf_for, "height", 0) == 0:
                            with contextlib.suppress(Exception):
                                import polars as _pl_load  # noqa: F401

                                wp = ctx.paths.results(run_id) / "windows.parquet"
                                if wp.exists():
                                    _wdf_for = _pl_load.read_parquet(str(wp))
                        try:
                            report_for = _summ_tmf(_wdf_for, _cdf_for, _sdf_for, threshold=0.40, near_miss_lo=0.20)
                            _write_tmf(ctx.paths.results(run_id), report_for)
                        except Exception as _e_for_inner:
                            logger.warning(f"[SYS] tail forensics write failed {_e_for_inner!r}")
                    else:
                        try:
                            import polars as _pl_load2  # noqa: F401

                            wp2 = ctx.paths.results(run_id) / "windows.parquet"
                            if wp2.exists():
                                _wdf_load = _pl_load2.read_parquet(str(wp2))
                                report_for2 = _summ_tmf(_wdf_load, _cdf_for, _sdf_for, threshold=0.40, near_miss_lo=0.20)
                                _write_tmf(ctx.paths.results(run_id), report_for2)
                        except Exception as _e_for_inner2:
                            logger.warning(f"[SYS] tail forensics write failed {_e_for_inner2!r}")
                except Exception as _e_for:
                    logger.warning(f"[SYS] tail forensics failed {_e_for!r}")


__all__ = ["attach_backtest_artifacts", "emit_backtest_artifacts"]
