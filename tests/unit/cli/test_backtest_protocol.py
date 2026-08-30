"""SCENARIO-11-03"""
import pytest

from src.cli import build_parser


def test_SCENARIO_11_03_cli_protocol_defaults() -> None:  # noqa: N802
    """SCENARIO-11-03"""
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "B1", "--start", "2024-01-02", "--end", "2024-01-08"])
    assert args.protocol == "single"
    assert args.stress_grid is False

    args_grid = parser.parse_args(
        ["backtest", "--model", "B1", "--start", "2024-01-02", "--end", "2024-01-08", "--protocol", "grid"]
    )
    assert args_grid.protocol == "grid"

    args_stress = parser.parse_args(
        ["backtest", "--model", "B1", "--start", "2024-01-02", "--end", "2024-01-08", "--stress-grid"]
    )
    assert args_stress.stress_grid is True
    assert args_stress.protocol == "single"


@pytest.mark.parametrize("scenario_id", ["SCENARIO-11-03"])
def test_SCENARIO_hyphen_wrapper_protocol(scenario_id: str) -> None:  # noqa: N802
    test_SCENARIO_11_03_cli_protocol_defaults()
