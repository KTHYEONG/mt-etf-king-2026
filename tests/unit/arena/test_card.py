import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena import card as card_module
from src.arena.card import (
    Action,
    Card,
    CardConfig,
    build_card,
    history_row,
    load_card_config,
    render_card,
    resolve_strategy,
    write_card,
)
from src.arena.contest import load_edition
from src.arena.evaluation import EvalConfig, load_eval_config
from src.arena.panel import load_panel_config
from src.arena.prize_line import PrizeLineConfig, load_prize_line_config
from src.arena.segment import load_segment_config
from src.arena.simulate import load_sim_config
from src.arena.strategies import load_strategies_config
from src.core.config import ConfigError, load_config

CARD = load_card_config()
STRATEGIES = load_strategies_config()
SIM = load_sim_config()
SEGMENT = load_segment_config()
EVAL = load_eval_config()
PRIZE = load_prize_line_config()
PANEL = load_panel_config()
EDITION = load_edition(2026)
TICKERS = {
    "0193T0": "KODEX SK하이닉스레버리지2X",
    "0197X0": "KODEX SK하이닉스인버스2X",
    "091160": "KODEX 반도체",
    "069500": "KODEX 200",
}
VOTE = ("0193T0", "091160", "069500")
RUN = date(2026, 10, 6)
EXPECTED = date(2026, 10, 5)
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


class _Calendar:
    """Weekday-only session calendar with the three queries build_card uses."""

    def __init__(self, sessions: list[date]) -> None:
        self._sessions = sorted(sessions)
        self._set = frozenset(sessions)

    def is_session(self, day: date) -> bool:
        return day in self._set

    def previous_session(self, day: date, offset: int = 1) -> date:
        index = bisect_left(self._sessions, day) - offset
        if index < 0:
            raise ValueError(f"no session {offset} before {day}")
        return self._sessions[index]

    def session_count(self, start: date, end: date) -> int:
        return bisect_right(self._sessions, end) - bisect_left(self._sessions, start)


def _weekdays(start: date, end: date) -> list[date]:
    days: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


DAYS = _weekdays(date(2025, 1, 1), date(2026, 12, 31))
CALENDAR = _Calendar(DAYS)


def _rows(seed: int = 20261106) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for ticker, name in TICKERS.items():
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.012, len(DAYS)))
        opens = np.concatenate([[closes[0]], closes[:-1]])
        for day, open_, close in zip(DAYS, opens.tolist(), closes.tolist(), strict=True):
            records.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "name": name,
                    "open": open_,
                    "high": max(open_, close) * 1.01,
                    "low": min(open_, close) * 0.99,
                    "close": close,
                    "volume": 1000,
                    "trading_value": 10**9,
                    "nav": close,
                    "shares_outstanding": 1000,
                    "underlying_index_name": "IDX",
                }
            )
    return pl.DataFrame(records, schema=SCHEMA)


def _shock(rows: pl.DataFrame, day: date, factor: float) -> pl.DataFrame:
    hit = (pl.col("date") == day) & pl.col("ticker").is_in(VOTE)
    return rows.with_columns(close=pl.when(hit).then(pl.col("close") * factor).otherwise(pl.col("close")))


def _card(
    rows: pl.DataFrame,
    run_date: date = RUN,
    *,
    cfg: CardConfig = CARD,
    equity: int | None = None,
    locked: bool = False,
    prize_cfg: PrizeLineConfig = PRIZE,
    eval_cfg: EvalConfig = EVAL,
) -> Card:
    return build_card(
        rows,
        run_date=run_date,
        cfg=cfg,
        strategies_cfg=STRATEGIES,
        sim_cfg=SIM,
        segment_cfg=SEGMENT,
        eval_cfg=eval_cfg,
        prize_cfg=prize_cfg,
        panel_cfg=PANEL,
        edition=EDITION,
        calendar=CALENDAR,
        equity_krw=equity,
        locked=locked,
    )


BASE_SECTION = dict(load_config("arena")["card"])


def test_shipped_config_loads_and_validates() -> None:
    assert CARD.strategy_id == "segment_fade:skhynix:vote3:long"
    assert CARD.edition == 2026
    assert CARD.w_fixed is None
    assert CARD.output_dir == "card"


@pytest.mark.parametrize(
    "section",
    [
        {**BASE_SECTION, "extra": 1},
        {key: value for key, value in BASE_SECTION.items() if key != "line_k"},
        {**BASE_SECTION, "line_k": 0.0},
        {**BASE_SECTION, "w_fixed": 1.0},
        {**BASE_SECTION, "output_dir": "a/b"},
        {**BASE_SECTION, "output_dir": ".."},
        {**BASE_SECTION, "edition": True},
        {**BASE_SECTION, "strategy_id": ""},
    ],
)
def test_invalid_card_section_raises(monkeypatch: pytest.MonkeyPatch, section: dict[str, Any]) -> None:
    monkeypatch.setattr(card_module, "load_config", lambda name: {"card": section})
    with pytest.raises(ConfigError):
        load_card_config()


def test_resolve_strategy_unknown_id_raises_and_known_baseline_resolves() -> None:
    with pytest.raises(ConfigError, match="is not a registered arena strategy"):
        resolve_strategy("segment_fade:skhynix:vote9:long", STRATEGIES)
    assert resolve_strategy("hold:0193T0", STRATEGIES).strategy_id == "hold:0193T0"


def test_segment_up_day_actions() -> None:
    card = _card(_shock(_rows(), EXPECTED, 1.05))
    assert card.status == "TRADE"
    assert card.signal_session == EXPECTED
    assert card.holding_now == "0193T0"
    assert card.actions == (
        Action(CARD.open_window, "0193T0", "0197X0"),
        Action(CARD.close_window, "0197X0", "0193T0"),
    )


def test_segment_down_day_open_noop() -> None:
    card = _card(_shock(_rows(), EXPECTED, 0.95))
    assert card.status == "TRADE"
    assert card.actions == (
        Action(CARD.open_window, None, None),
        Action(CARD.close_window, None, None),
    )
    assert "할 일 없음 (계좌 확인만)" in render_card(card)


def test_stale_data_gives_no_trade_naming_the_date() -> None:
    stale = _rows().filter(pl.col("date") <= date(2026, 10, 2))
    card = _card(stale)
    assert card.status == "NO_TRADE"
    assert all(action.sell is None and action.buy is None for action in card.actions)
    assert any("2026-10-02" in warning for warning in card.warnings)


def test_rows_on_or_after_run_date_do_not_change_the_card() -> None:
    rows = _rows()
    corrupt = rows.with_columns(
        close=pl.when(pl.col("date") >= RUN).then(pl.col("close") * 50.0).otherwise(pl.col("close")),
        volume=pl.when(pl.col("date") >= RUN).then(pl.lit(0)).otherwise(pl.col("volume")),
    )
    assert _card(rows) == _card(corrupt)


def test_non_session_run_date_is_closed() -> None:
    card = _card(_rows(), date(2026, 10, 3))
    assert card.status == "CLOSED"
    assert card.actions == ()
    assert any("휴장" in warning for warning in card.warnings)


def test_outside_window_still_shows_actions_with_provisional_line() -> None:
    card = _card(_rows(), date(2026, 8, 3))
    assert card.status == "OUT_OF_WINDOW"
    assert card.session_index is None
    assert len(card.actions) == 2
    assert card.line_w is not None
    assert card.line_basis is not None and card.line_basis.endswith("임시")


def test_lock_trigger_formula_with_fixed_line() -> None:
    card = _card(_rows(), cfg=replace(CARD, w_fixed=1.3))
    assert card.line_w == 1.3
    lock_w = 1.3**EVAL.lock_level_scale
    assert card.lock_w == pytest.approx(lock_w)
    assert card.lock_trigger_krw == math.floor(1_000_000_000 * lock_w * (1.0 + EVAL.lock_trigger_buffer))
    assert EVAL.lock_trigger_buffer == 0.01


def test_lock_level_sits_above_the_central_line() -> None:
    card = _card(_rows(), cfg=replace(CARD, w_fixed=1.3))
    flat = _card(_rows(), cfg=replace(CARD, w_fixed=1.3), eval_cfg=replace(EVAL, lock_level_scale=1.0))
    assert EVAL.lock_level_scale > 1.0
    assert card.lock_w is not None and card.lock_w > card.line_w
    assert flat.lock_trigger_krw == math.floor(1_000_000_000 * 1.3 * (1.0 + EVAL.lock_trigger_buffer))
    assert "잠금 배수" in render_card(card)


def test_line_is_frozen_at_edition_start() -> None:
    rows = _rows()
    before = _card(rows)
    assert before.line_basis is not None and before.line_basis.startswith("k=1")
    later = rows.with_columns(
        close=pl.when(pl.col("date") > EDITION.start).then(pl.col("close") * 3.0).otherwise(pl.col("close"))
    )
    after = _card(later)
    assert before.line_w is not None
    assert after.line_w == before.line_w


def test_locked_flag_sells_holding_to_cash() -> None:
    card = _card(_rows(), locked=True)
    assert card.status == "LOCKED"
    assert card.actions == (Action(CARD.open_window, "0193T0", None),)


def test_line_reached_warns_without_auto_lock() -> None:
    card = _card(_rows(), equity=10**10)
    assert card.status == "TRADE"
    assert any("--locked" in warning for warning in card.warnings)


def test_daily_baseline_is_a_single_noop_open_action() -> None:
    card = _card(_rows(), cfg=replace(CARD, strategy_id="hold:0193T0"))
    assert card.status == "TRADE"
    assert card.actions == (Action(CARD.open_window, None, None),)


def test_render_has_no_rank_lines() -> None:
    text = render_card(_card(_rows()))
    assert text.startswith("# 2026-10-06 (화) 매매 카드")
    for banned in ("순위", "1위", "상금권"):
        assert banned not in text


def test_write_card_overwrites_today_and_appends_history_once(tmp_path: Path) -> None:
    card = _card(_rows())
    out = tmp_path / "card"
    assert write_card(card, out, force=False) == out / "today.md"
    write_card(card, out, force=False)
    lines = (out / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == history_row(card)
    assert (out / "today.md").read_text(encoding="utf-8") == render_card(card)
    write_card(card, out, force=True)
    assert len((out / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_non_numeric_line_k_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_module, "load_config", lambda name: {"card": {**BASE_SECTION, "line_k": "1.0"}})
    with pytest.raises(ConfigError, match="finite number"):
        load_card_config()


def test_missing_card_mapping_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_module, "load_config", lambda name: {})
    with pytest.raises(ConfigError, match="must define a card mapping"):
        load_card_config()


def test_no_rows_before_signal_session_gives_no_trade() -> None:
    card = _card(_rows().filter(pl.col("date") > EXPECTED))
    assert card.status == "NO_TRADE"
    assert any("데이터 없음" in warning for warning in card.warnings)


def test_missing_strategy_ticker_gives_no_trade() -> None:
    card = _card(_rows().filter(pl.col("ticker") != "091160"))
    assert card.status == "NO_TRADE"
    assert any("091160" in warning for warning in card.warnings)


def test_negative_equity_input_raises() -> None:
    with pytest.raises(ValueError, match="equity_krw must be >= 0"):
        _card(_rows(), equity=-1)


def test_nan_reference_sigma_in_window_gives_no_trade() -> None:
    rows = _rows().filter(~((pl.col("ticker") == "069500") & (pl.col("date") < date(2026, 8, 3))))
    card = _card(rows, prize_cfg=replace(PRIZE, reference="069500"))
    assert card.status == "NO_TRADE"
    assert card.line_w is None
    assert any("W 계산 불가" in warning for warning in card.warnings)


def test_empty_model_window_gives_no_equity_warning() -> None:
    late_edition = replace(EDITION, start=RUN)
    card = build_card(
        _rows(),
        run_date=RUN,
        cfg=CARD,
        strategies_cfg=STRATEGIES,
        sim_cfg=SIM,
        segment_cfg=SEGMENT,
        eval_cfg=EVAL,
        prize_cfg=PRIZE,
        panel_cfg=PANEL,
        edition=late_edition,
        calendar=CALENDAR,
        equity_krw=None,
        locked=False,
    )
    assert card.equity_krw is None
    assert any("모델 잔고 없음" in warning for warning in card.warnings)


def test_daily_strategy_without_decision_gives_no_trade() -> None:
    rows = _rows().filter(~((pl.col("ticker") == "0193T0") & (pl.col("date") < date(2026, 9, 28))))
    card = _card(rows, cfg=replace(CARD, strategy_id="hold:0193T0"))
    assert card.status == "NO_TRADE"
    assert any("결정 없음" in warning for warning in card.warnings)


def test_locked_with_stale_data_flags_unknown_holding() -> None:
    card = _card(_rows().filter(pl.col("date") <= date(2026, 10, 2)), locked=True)
    assert card.status == "LOCKED"
    assert card.actions == (Action(CARD.open_window, None, None),)
    assert any("보유 확인 불가" in warning for warning in card.warnings)


def test_cash_overnight_close_sells_to_cash_and_renders() -> None:
    cash_card = _card(_shock(_rows(), EXPECTED, 1.05), cfg=replace(CARD, strategy_id="segment_fade:skhynix:vote3:cash"))
    assert cash_card.actions[1] == Action(CARD.close_window, "0197X0", None)
    assert "KODEX SK하이닉스인버스2X(0197X0) 전량 매도 (현금)" in render_card(cash_card)



def test_render_covers_closed_no_trade_and_locked_states() -> None:
    closed = render_card(_card(_rows(), date(2026, 10, 3)))
    assert "휴장일: 할 일 없음" in closed
    no_trade = render_card(_card(_rows().filter(pl.col("date") <= date(2026, 10, 2))))
    assert "현 보유 유지, 매매 없음" in no_trade
    locked = render_card(_card(_rows(), locked=True))
    assert "전량 현금 유지: KODEX SK하이닉스레버리지2X(0193T0) 전량 매도" in locked
    flat = render_card(_card(_rows().filter(pl.col("date") <= date(2026, 10, 2)), locked=True))
    assert "전량 현금 유지 (보유 없음)" in flat
