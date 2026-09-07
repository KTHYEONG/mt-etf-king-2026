from argparse import Namespace

from src.cli.commands.feasibility_audit import cmd_feasibility_audit
from src.cli.main import SUBCOMMANDS as MAIN_SUBCOMMANDS
from src.cli.parser import SUBCOMMANDS, build_parser


def test_feasibility_audit_cli_registered_and_rejects_non_36_horizon() -> None:
    parser = build_parser()
    args = parser.parse_args(["feasibility-audit", "--start", "2018-01-02", "--end", "2026-08-27", "--horizon", "35"])
    assert args.horizon == 35
    assert "feasibility-audit" in SUBCOMMANDS
    assert "feasibility-audit" in MAIN_SUBCOMMANDS
    assert cmd_feasibility_audit(Namespace(horizon=35, start="2018-01-02", end="2026-08-27", p27_run=None, b1_run=None, p38_promotion=None, output="docs/research", data_root="data")) == 1


def test_feasibility_audit_cli_success_writes_outputs(tmp_path) -> None:
    import json
    from datetime import date

    import polars as pl

    from src.core.calendar import get_calendar

    sessions = get_calendar().sessions(date(2024, 1, 2), date(2026, 3, 31))
    assert len(sessions) >= 80
    tickers = ["A", "B"]
    panel = pl.DataFrame(
        {
            "date": [d for d in sessions for _ in tickers],
            "ticker": tickers * len(sessions),
            "name": ["삼성 ETF" if t == "A" else "타사 ETF" for d in sessions for t in tickers],
            "open": [100.0 + float(i) for i in range(len(sessions)) for _ in tickers],
            "close": [101.0 + float(i) for i in range(len(sessions)) for _ in tickers],
            "trading_value": [200_000_000.0] * (len(sessions) * len(tickers)),
            "is_tradable": [True] * (len(sessions) * len(tickers)),
        }
    )
    data_root = tmp_path / "data"
    (data_root / "normalized").mkdir(parents=True)
    panel.write_parquet(data_root / "normalized" / "etf_daily.parquet")
    p27_dir = tmp_path / "p27"
    p27_dir.mkdir()
    pl.DataFrame(
        {"window_start": sessions, "terminal_return": [0.05] * len(sessions), "giveback": [0.01] * len(sessions)}
    ).write_parquet(p27_dir / "windows.parquet")
    b1_dir = tmp_path / "b1"
    b1_dir.mkdir()
    pl.DataFrame({"window_start": sessions, "terminal_return": [0.02] * len(sessions)}).write_parquet(b1_dir / "windows.parquet")
    p38_path = tmp_path / "promotion.json"
    p38_path.write_text(json.dumps({"terminal_returns": [0.03] * len(sessions)}), encoding="utf-8")
    out = tmp_path / "research"
    rc = cmd_feasibility_audit(
        Namespace(
            horizon=36,
            start="2024-01-02",
            end="2026-03-31",
            p27_run=str(p27_dir),
            b1_run=str(b1_dir),
            p38_promotion=str(p38_path),
            output=str(out),
            data_root=str(data_root),
        )
    )
    assert rc == 0
    assert (out / "window_population_reconciliation.md").exists()
    assert (out / "2026_championship_feasibility_metrics.json").exists()
    assert (out / "2026_championship_feasibility_tables.csv").exists()
    metrics = json.loads((out / "2026_championship_feasibility_metrics.json").read_text(encoding="utf-8"))
    assert metrics["b1_role"] == "NON_COMPLIANT_REFERENCE"
    assert metrics["p38_status"] == "AVAILABLE"


def test_feasibility_audit_cli_empty_panel_returns_1(tmp_path) -> None:
    import polars as pl

    data_root = tmp_path / "data"
    (data_root / "normalized").mkdir(parents=True)
    pl.DataFrame({"date": [], "ticker": [], "name": [], "open": [], "trading_value": [], "is_tradable": []}).write_parquet(data_root / "normalized" / "etf_daily.parquet")
    rc = cmd_feasibility_audit(
        Namespace(
            horizon=36,
            start="2026-01-02",
            end="2026-01-08",
            p27_run=None,
            b1_run=None,
            p38_promotion=None,
            output=str(tmp_path / "research"),
            data_root=str(data_root),
        )
    )
    assert rc == 1
