"""Morning trading card: one causal decision per session for the arena strategy selected in `configs/arena.yaml`."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import polars as pl

from src.arena.contest import EditionSpec
from src.arena.evaluation import EvalConfig
from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.prize_line import PrizeLineConfig, reference_sigma, vol_scaled_line
from src.arena.research import Strategy, contest_window_multiple
from src.arena.segment import SegmentConfig
from src.arena.simulate import SimConfig
from src.arena.strategies import StrategiesConfig, baselines, candidate_grid, strategy_tickers
from src.core.atomic_io import atomic_write_text
from src.core.config import ConfigError, load_config

Status = Literal["TRADE", "NO_TRADE", "LOCKED", "CLOSED", "OUT_OF_WINDOW"]
EquitySource = Literal["input", "model", "none"]

_CARD_KEYS: Final = frozenset(
    {"strategy_id", "edition", "lock_k", "w_fixed", "open_window", "close_window", "output_dir"}
)
_PANEL_LOOKBACK_DAYS: Final = 400
_TODAY_FILE: Final = "today.md"
_HISTORY_FILE: Final = "history.jsonl"
_WEEKDAYS: Final = ("월", "화", "수", "목", "금", "토", "일")
_LOCK_MESSAGE: Final = "라인 도달: 계좌 확인 후 --locked 로 실행 (자동 전환 없음)"
_PROVISIONAL_MESSAGE: Final = "W 임시: 편집 시작 전이라 마지막 세션 sigma 기준"
_HOLD_MESSAGE: Final = "보유 확인 불가: 잠금 전 계좌에서 직접 확인"


@dataclass(frozen=True)
class CardConfig:
    strategy_id: str
    edition: int
    lock_k: float
    w_fixed: float | None
    open_window: str
    close_window: str
    output_dir: str


@dataclass(frozen=True)
class Action:
    window: str
    sell: str | None
    buy: str | None


@dataclass(frozen=True)
class Card:
    run_date: date
    signal_session: date
    strategy_id: str
    status: Status
    actions: tuple[Action, ...]
    holding_now: str | None
    session_index: int | None
    sessions_total: int | None
    line_w: float | None
    lock_trigger_krw: int | None
    equity_krw: int | None
    equity_source: EquitySource
    warnings: tuple[str, ...]
    names: Mapping[str, str]
    capital_krw: int
    line_basis: str | None


class CalendarLike(Protocol):
    def is_session(self, day: date) -> bool: ...

    def previous_session(self, day: date, offset: int = 1) -> date: ...

    def session_count(self, start: date, end: date) -> int: ...


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"{where} must be a finite number")
    return float(value)


def load_card_config(name: str = "arena") -> CardConfig:
    """Load and validate the `card` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("card")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a card mapping")
    if set(section) != _CARD_KEYS:
        raise ConfigError(f"card keys must be exactly {sorted(_CARD_KEYS)}")
    edition = section["edition"]
    if type(edition) is not int:
        raise ConfigError("card.edition must be an integer")
    lock_k = _number(section["lock_k"], "card.lock_k")
    if not lock_k > 0.0:
        raise ConfigError("card.lock_k must be > 0")
    w_fixed: float | None = None
    if section["w_fixed"] is not None:
        w_fixed = _number(section["w_fixed"], "card.w_fixed")
        if not w_fixed > 1.0:
            raise ConfigError("card.w_fixed must be null or > 1.0")
    output_dir = _text(section["output_dir"], "card.output_dir")
    if any(token in output_dir for token in ("/", "\\", "..")):
        raise ConfigError("card.output_dir must be a single path part")
    return CardConfig(
        strategy_id=_text(section["strategy_id"], "card.strategy_id"),
        edition=edition,
        lock_k=lock_k,
        w_fixed=w_fixed,
        open_window=_text(section["open_window"], "card.open_window"),
        close_window=_text(section["close_window"], "card.close_window"),
        output_dir=output_dir,
    )


def resolve_strategy(strategy_id: str, strategies_cfg: StrategiesConfig) -> Strategy:
    """Return the single arena strategy (candidate or baseline) with this id; raises ConfigError otherwise."""
    matches = [
        strategy
        for strategy in (*candidate_grid(strategies_cfg), *baselines(strategies_cfg))
        if strategy.strategy_id == strategy_id
    ]
    if len(matches) != 1:
        raise ConfigError(f"card.strategy_id {strategy_id} is not a registered arena strategy")
    return matches[0]


def prize_line_multiple(
    panel: Panel,
    edition: EditionSpec,
    cfg: CardConfig,
    prize_cfg: PrizeLineConfig,
    horizon: int,
) -> float:
    """Return the prize line W; raises ValueError when the reference sigma is undefined at the reference session."""
    if cfg.w_fixed is not None:
        return cfg.w_fixed
    sigmas = reference_sigma(panel, prize_cfg)
    index = panel.sessions.index(edition.start) if _line_frozen(panel, edition) else len(panel.sessions) - 1
    return vol_scaled_line(float(sigmas[index]), horizon, cfg.lock_k)


def _line_frozen(panel: Panel, edition: EditionSpec) -> bool:
    return edition.start in panel.sessions


def _names(visible: pl.DataFrame) -> dict[str, str]:
    latest = visible.sort("date").group_by("ticker").agg(pl.col("name").last())
    return {str(ticker): str(name) for ticker, name in latest.iter_rows()}


def _fresh_panel(
    visible: pl.DataFrame,
    expected: date,
    tickers: frozenset[str],
    edition: EditionSpec,
    panel_cfg: PanelConfig,
    warnings: list[str],
) -> Panel | None:
    if visible.is_empty():
        warnings.append(f"데이터 없음: 기준 세션 {expected.isoformat()} 이전 행이 없습니다")
        return None
    earliest: date = visible.select(pl.col("date").min()).item()
    lookback_start = edition.start - timedelta(days=_PANEL_LOOKBACK_DAYS)
    start = earliest if lookback_start >= expected else max(earliest, lookback_start)
    panel = build_panel(visible, sorted(tickers), start, expected, panel_cfg)
    if panel.missing_tickers:
        warnings.append(f"데이터 누락: {', '.join(panel.missing_tickers)}")
        return None
    last = panel.sessions[-1] if panel.sessions else None
    if last != expected:
        warnings.append(
            f"데이터 지연: 마지막 세션 {last.isoformat() if last else '없음'}, 기준 세션 {expected.isoformat()}"
        )
        return None
    return panel


def _hold_of(row: Mapping[str, Any]) -> str | None:
    return str(row["ticker"]) if float(row["weight"]) > 0.0 else None


def _move(source: str | None, destination: str | None) -> tuple[str | None, str | None]:
    if source == destination:
        return None, None
    return source, destination


def _segment_plan(targets: pl.DataFrame, expected: date, cfg: CardConfig) -> tuple[str | None, tuple[Action, ...]]:
    prior = targets.filter((pl.col("leg") == "overnight") & (pl.col("decision_date") < expected))
    holding = _hold_of(prior.row(prior.height - 1, named=True)) if prior.height else None
    today = targets.filter(pl.col("decision_date") == expected)
    intraday = today.filter(pl.col("leg") == "intraday")
    overnight = today.filter(pl.col("leg") == "overnight")
    intraday_to = _hold_of(intraday.row(0, named=True))
    overnight_to = _hold_of(overnight.row(0, named=True))
    actions = (
        Action(cfg.open_window, *_move(holding, intraday_to)),
        Action(cfg.close_window, *_move(intraday_to, overnight_to)),
    )
    return holding, actions


def _daily_plan(targets: pl.DataFrame, expected: date, cfg: CardConfig) -> tuple[str | None, tuple[Action, ...]] | None:
    upto = targets.filter(pl.col("decision_date") <= expected).sort("decision_date")
    if upto.is_empty():
        return None
    target = _hold_of(upto.row(upto.height - 1, named=True))
    prior = upto.filter(pl.col("decision_date") < expected)
    holding = _hold_of(prior.row(prior.height - 1, named=True)) if prior.height else None
    return holding, (Action(cfg.open_window, *_move(holding, target)),)


def _decision(
    strategy: Strategy, panel: Panel, expected: date, cfg: CardConfig
) -> tuple[str | None, tuple[Action, ...]] | None:
    targets = strategy.targets(panel)
    if getattr(strategy, "execution_model", "daily") == "segment":
        return _segment_plan(targets, expected, cfg)
    return _daily_plan(targets, expected, cfg)


def _lock_trigger(capital_krw: int, line_w: float | None, buffer: float) -> int | None:
    if line_w is None:
        return None
    return math.floor(capital_krw * line_w * (1.0 + buffer))


def build_card(
    rows: pl.DataFrame,
    *,
    run_date: date,
    cfg: CardConfig,
    strategies_cfg: StrategiesConfig,
    sim_cfg: SimConfig,
    segment_cfg: SegmentConfig,
    eval_cfg: EvalConfig,
    prize_cfg: PrizeLineConfig,
    panel_cfg: PanelConfig,
    edition: EditionSpec,
    calendar: CalendarLike,
    equity_krw: int | None,
    locked: bool,
) -> Card:
    """Return the card for `run_date` using only rows dated on or before the previous session; never raises on stale data."""
    if equity_krw is not None and equity_krw < 0:
        raise ValueError("equity_krw must be >= 0")
    strategy = resolve_strategy(cfg.strategy_id, strategies_cfg)
    expected = calendar.previous_session(run_date)
    visible = rows.filter(pl.col("date") <= expected)
    names = _names(visible) if not visible.is_empty() else {}
    capital = edition.initial_capital_krw
    if not calendar.is_session(run_date):
        return Card(
            run_date=run_date,
            signal_session=expected,
            strategy_id=cfg.strategy_id,
            status="CLOSED",
            actions=(),
            holding_now=None,
            session_index=None,
            sessions_total=None,
            line_w=None,
            lock_trigger_krw=None,
            equity_krw=None,
            equity_source="none",
            warnings=(f"휴장일 {run_date.isoformat()}: 매매 없음",),
            names=names,
            capital_krw=capital,
            line_basis=None,
        )

    warnings: list[str] = []
    tickers = strategy_tickers((strategy,)) | {prize_cfg.reference}
    panel = _fresh_panel(visible, expected, tickers, edition, panel_cfg, warnings)
    in_window = edition.start <= run_date <= edition.end
    horizon = calendar.session_count(edition.start, edition.end)

    line_w: float | None = None
    line_basis: str | None = None
    if panel is not None:
        try:
            line_w = prize_line_multiple(panel, edition, cfg, prize_cfg, horizon)
        except ValueError as exc:
            warnings.append(f"W 계산 불가: {exc}")
        if cfg.w_fixed is not None:
            line_basis = f"고정 {cfg.w_fixed:.2f}"
        elif line_w is not None:
            provisional = not _line_frozen(panel, edition)
            line_basis = f"k={cfg.lock_k:g}" + (" 임시" if provisional else "")
            if provisional:
                warnings.append(_PROVISIONAL_MESSAGE)
    lock_trigger = _lock_trigger(capital, line_w, eval_cfg.lock_trigger_buffer)

    session_index = calendar.session_count(edition.start, run_date) if in_window else None
    session_total = horizon if in_window else None

    equity: int | None = None
    source: EquitySource = "none"
    if equity_krw is not None:
        equity, source = equity_krw, "input"
    elif in_window and panel is not None:
        multiple = contest_window_multiple(strategy, panel, sim_cfg, edition, segment_cfg)
        if multiple is None:
            warnings.append("모델 잔고 없음: 대회 기간 세션이 없습니다")
        else:
            equity, source = math.floor(capital * multiple), "model"

    decision = None if panel is None else _decision(strategy, panel, expected, cfg)
    holding = decision[0] if decision is not None else None
    failure: str | None = None
    if panel is None:
        failure = "데이터 점검 실패"
    elif decision is None:
        failure = f"{expected.isoformat()} 결정 없음"
    elif in_window and line_w is None:
        failure = "W 계산 불가"

    if locked:
        status: Status = "LOCKED"
        actions: tuple[Action, ...] = (Action(cfg.open_window, holding, None),)
        if failure is not None:
            warnings.append(_HOLD_MESSAGE)
    elif failure is not None:
        status = "NO_TRADE"
        actions = (Action(cfg.open_window, None, None),)
        warnings.append(f"매매 없음: {failure}")
    else:
        assert decision is not None
        status = "TRADE" if in_window else "OUT_OF_WINDOW"
        actions = decision[1]

    if equity is not None and lock_trigger is not None and equity >= lock_trigger and not locked:
        warnings.append(_LOCK_MESSAGE)

    return Card(
        run_date=run_date,
        signal_session=expected,
        strategy_id=cfg.strategy_id,
        status=status,
        actions=actions,
        holding_now=holding,
        session_index=session_index,
        sessions_total=session_total,
        line_w=line_w,
        lock_trigger_krw=lock_trigger,
        equity_krw=equity,
        equity_source=source,
        warnings=tuple(warnings),
        names=names,
        capital_krw=capital,
        line_basis=line_basis,
    )


def _label(ticker: str, names: Mapping[str, str]) -> str:
    return f"{names.get(ticker, ticker)}({ticker})"


def _action_text(action: Action, names: Mapping[str, str]) -> str:
    if action.sell is None and action.buy is None:
        return "할 일 없음"
    if action.sell is not None and action.buy is not None:
        return f"{_label(action.sell, names)} 전량 매도 → {_label(action.buy, names)} 전량 매수"
    if action.sell is not None:
        return f"{_label(action.sell, names)} 전량 매도 (현금)"
    assert action.buy is not None
    return f"{_label(action.buy, names)} 전량 매수 (현금에서)"


def _todo_lines(card: Card) -> list[str]:
    if card.status == "CLOSED":
        return ["휴장일: 할 일 없음"]
    if card.status == "NO_TRADE":
        return [f"1. {card.actions[0].window} 현 보유 유지, 매매 없음"]
    if card.status == "LOCKED":
        sell = card.actions[0].sell
        detail = f": {_label(sell, card.names)} 전량 매도" if sell is not None else " (보유 없음)"
        return [f"1. {card.actions[0].window} 전량 현금 유지{detail}"]
    lines = [f"1. {card.actions[0].window} {_action_text(card.actions[0], card.names)}"]
    if len(card.actions) > 1:
        close = card.actions[1]
        trigger = "W 미정" if card.lock_trigger_krw is None else f"{card.lock_trigger_krw:,}원"
        close_text = _action_text(close, card.names) if close.sell or close.buy else "할 일 없음 (계좌 확인만)"
        lines.append(
            f"2. {close.window} 계좌 확인: {trigger} 이상이면 → 전량 매도, 대회 끝까지 현금 (목표 도달). "
            f"아니면 → {close_text}"
        )
    return lines


def _status_lines(card: Card) -> list[str]:
    capital = card.capital_krw
    if card.equity_krw is None:
        account = "- 계좌: 확인 불가"
    else:
        multiple = card.equity_krw / capital
        source = {"input": "입력", "model": "모델", "none": "없음"}[card.equity_source]
        account = f"- 계좌: {card.equity_krw:,}원 ({source}), 배수 {multiple:.3f}"
    lines = [account]
    if card.line_w is not None:
        basis = f" ({card.line_basis})" if card.line_basis else ""
        lines.append(f"- 상금 라인 W: {card.line_w:.3f}{basis}")
        if card.equity_krw is not None:
            gap = card.line_w - card.equity_krw / capital
            lines.append(f"- W까지: {gap:.3f}배" if gap > 0.0 else "- W 도달: 계좌 확인 필요")
    if card.lock_trigger_krw is not None:
        lines.append(f"- 잠금 기준: {card.lock_trigger_krw:,}원")
    lines.append(f"- 상태: {card.status}")
    lines.append(f"- 전략: {card.strategy_id}")
    lines.append(f"- 신호 세션: {card.signal_session.isoformat()}")
    return lines


def render_card(card: Card) -> str:
    """Return the card as Korean markdown in the fixed section order; contains no leaderboard or rank lines."""
    weekday = _WEEKDAYS[card.run_date.weekday()]
    lines = [f"# {card.run_date.isoformat()} ({weekday}) 매매 카드", ""]
    if card.session_index is not None and card.sessions_total is not None:
        remaining = card.sessions_total - card.session_index
        lines.append(f"세션 {card.session_index}/{card.sessions_total} · 남은 {remaining}")
    else:
        lines.append("대회 기간 밖 (참고용)")
    lines.extend(["", "## 오늘 할 일", *_todo_lines(card), "", "## 현황", *_status_lines(card)])
    if card.warnings:
        lines.extend(["", "## 경고", *(f"- {warning}" for warning in card.warnings)])
    return "\n".join(lines) + "\n"


def history_row(card: Card) -> dict[str, Any]:
    """Return the JSON-serialisable history record of one card."""
    return {
        "run_date": card.run_date.isoformat(),
        "signal_session": card.signal_session.isoformat(),
        "strategy_id": card.strategy_id,
        "status": card.status,
        "actions": [asdict(action) for action in card.actions],
        "holding_now": card.holding_now,
        "equity_krw": card.equity_krw,
        "equity_source": card.equity_source,
        "lock_trigger_krw": card.lock_trigger_krw,
        "line_w": card.line_w,
    }


def write_card(card: Card, out_dir: Path, *, force: bool) -> Path:
    """Overwrite `today.md` and append the history row once per run date (twice with `force`); returns the `today.md` path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    history = out_dir / _HISTORY_FILE
    existing = history.read_text(encoding="utf-8").splitlines() if history.is_file() else []
    run_key = card.run_date.isoformat()
    recorded = any(json.loads(line).get("run_date") == run_key for line in existing if line.strip())
    today = out_dir / _TODAY_FILE
    atomic_write_text(today, render_card(card))
    if force or not recorded:
        row = json.dumps(history_row(card), ensure_ascii=False, allow_nan=False, sort_keys=True)
        atomic_write_text(history, "\n".join([*existing, row]) + "\n")
    return today
