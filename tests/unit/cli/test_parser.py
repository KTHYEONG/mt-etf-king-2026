# ruff: noqa
from __future__ import annotations


def test_parser_accepts_adaptive_specialists_mode() -> None:
    from src.cli.parser import build_parser

    args = build_parser().parse_args(["champion-research", "--start", "2018-01-02", "--end", "2026-08-27", "--candidate-mode", "adaptive_specialists"])
    assert args.candidate_mode == "adaptive_specialists"

def test_parser_accepts_explicit_frontier_artifact_paths() -> None:
    from src.cli.parser import build_parser
    args=build_parser().parse_args(["champion-research","--start","2018-01-02","--end","2026-08-27","--candidate-mode","capacity_frontier","--data-root","data","--output","results/championship_frontier"])
    assert args.candidate_mode == "capacity_frontier"
    assert args.data_root == "data"
    assert args.output == "results/championship_frontier"
