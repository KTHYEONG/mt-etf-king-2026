# 02. System Layers — 8계층 아키텍처

## 1. 계층 다이어그램

```
┌─────────────────────────────────────────────┐
│  L0  core/          settings · calendar     │  ← 모든 계층의 기반
│                     paths · logging         │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L1  data/          providers → bronze    │  raw 불변 · 타입 fail-closed
│                     → silver (Parquet)      │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L2  universe/      instrument master       │  "오늘 실제로 살 수 있는가"
│                     PIT universe            │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L3  features/      momentum · trend · vol  │  PIT guard 강제
│                     flow · breadth · regime │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L4  alpha/         baselines · leadership  │  Signal
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L5  portfolio/     sizing · constraints    │  Signal ≠ Portfolio
│                     state machine           │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L6  backtest/      next-open execution     │  look-ahead 차단
│                     costs · metrics         │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L7  tournament/    rolling-36D · replay  │  분포 산출
│                     bootstrap · MC          │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  L8  reporting/     daily decision          │  근거 포함 출력
└─────────────────────────────────────────────┘
```

**Alpha 모델만 만드는 프로젝트가 되어서는 안 됩니다.** L1~L7 이 먼저 제대로 되어 있어야 L4 를 믿을 수 있습니다.

---

## 2. 모듈 맵 (`src/`)

```
src/
├── cli.py                    # 단일 진입점 (SUBCOMMANDS 레지스트리)
│
├── core/
│   ├── settings.py           # Settings, get_settings (SecretStr)
│   ├── sops_env.py           # .env.enc sops 복호화 (메모리 전용)
│   ├── calendar.py           # TradingCalendar (XKRX)
│   ├── paths.py              # DataPaths (bronze/silver/gold/state)
│   └── logging_setup.py      # [SYS][DATA][ALGO][EVAL] 태그
│
├── data/
│   ├── providers/
│   │   ├── base.py           # MarketDataProvider Protocol
│   │   ├── krx.py            # KRXOpenAPIProvider
│   │   └── ratelimit.py      # RateLimiter, QuotaLedger
│   ├── bronze.py             # BronzeStore (write-once JSON)
│   ├── backfill.py           # BackfillPlanner, run_backfill
│   ├── schema.py             # DatasetSchema, 디코더
│   ├── silver.py             # SilverBuilder
│   └── validation.py         # PanelValidator
│
├── universe/
│   ├── instruments.py        # InstrumentMaster, resolve_leverage
│   ├── taxonomy.py           # index_key + leverage_family + theme
│   ├── families.py           # LeverageFamily (IDX_IND_NM 기반)
│   ├── provider.py           # PointInTimeUniverse
│   └── tournament.py         # TournamentRules, UNKNOWN sentinel
│
├── features/
│   ├── pit.py                # assert_pit, align_session_grid
│   ├── momentum.py           # mom_{3,5,10,20,40,60}
│   ├── trend.py              # MA ratio, breakout, drawdown
│   ├── volatility.py         # rv, ATR, downside
│   ├── flow.py               # creation_flow, disparity
│   ├── breadth.py            # market + cluster breadth
│   ├── crosssec.py           # percentile rank, acceleration
│   ├── regime.py             # 5-state classifier
│   └── builder.py            # FeatureBuilder
│
├── alpha/
│   ├── base.py               # AlphaModel Protocol
│   ├── baselines.py          # B0~B5
│   ├── leadership.py         # SectorLeadershipModel (Stage 4)
│   └── ml/                   # Stage 5 — AlphaModel 구현체
│       ├── dataset.py        # (X, y, group), 라벨 = 단면 rank
│       ├── splits.py         # PurgedWalkForward(horizon, embargo)
│       ├── ranker.py         # LightGBMRankerAlpha
│       ├── train.py          # fold 루프, nested early stopping
│       └── registry.py       # 모델 아티팩트 버전 관리
│
├── portfolio/
│   ├── selection.py          # ClusterAwareSelection (family dedup 우선)
│   ├── sizing.py             # weights_from_scores, confidence
│   ├── constraints.py        # normalize_weights, gross_exposure_gate
│   └── state.py              # PositionState machine
│
├── backtest/
│   ├── engine.py             # BacktestEngine
│   ├── execution.py          # NextOpenExecution
│   ├── costs.py              # CostModel, CostConfig.grid
│   └── metrics.py            # compound_returns, window_returns
│
├── tournament/
│   ├── simulator.py          # TournamentSimulator (rolling-36D)
│   ├── distribution.py       # ReturnDistribution, bootstrap
│   ├── replay.py             # TournamentReplay (2025)
│   ├── policy.py             # AggressionPolicy
│   ├── exposure.py           # ExposureSelector (배수 선택)
│   └── montecarlo.py         # CompetitorField (stress only)
│
└── reporting/
    └── dashboard.py          # DailyDecision, render_dashboard
```

---

## 3. 데이터 저장 구조

```
data/
├── raw/                      # bronze — write-once JSON envelope
│   └── krx/
│       └── etp/etf_bydd_trd/2026/20260827.json
├── normalized/               # silver — Parquet
│   ├── etf_daily.parquet
│   ├── index_daily.parquet
│   └── stock_daily.parquet
├── features/                   # gold — feature Parquet
│   └── etf_features.parquet
├── competition_history/          # 제1·2회 대회 공개 데이터 (문서/Parquet)
└── state/                      # quota ledger 등
    └── krx_quota.json
```

| 계층 | 형식 | 불변성 |
| --- | --- | --- |
| Bronze | JSON envelope | write-once (재수집 시 revision 파일) |
| Silver | Parquet (zstd) | validation PASS 후에만 갱신 |
| Gold | Parquet (zstd) | feature 재빌드 시 멱등 |

---

## 4. 일일 데이터 흐름

```
KRX Open API
    │  basDd=YYYYMMDD (1 call = 1 session 전체 단면)
    ▼
BronzeStore (raw JSON, 해석 없음)
    ▼
SilverBuilder (타입 디코드 + validation)
    ▼
InstrumentMaster + PointInTimeUniverse
    ▼
FeatureBuilder (PIT guard)
    ▼
AlphaModel.score → Portfolio → NextOpenExecution
    ▼
BacktestEngine / TournamentSimulator / DailyDecision
```

---

## 5. CLI 진입점

모든 기능은 `mt-etf` 단일 CLI 로 노출됩니다. `SUBCOMMANDS` 레지스트리에 단계별로 등록합니다.

| Subcommand | 계층 | 역할 |
| --- | --- | --- |
| `config-check` | L0 | 설정·경로 검증 |
| `calendar` | L0 | XKRX 세션 조회 |
| `ingest` | L1 | KRX backfill |
| `normalize` | L1 | bronze → silver |
| `universe` | L2 | PIT universe (`--mode structural\|deployment`) |
| `features` | L3 | silver → gold features |
| `backtest` | L6~7 | baseline/strategy 백테스트 |
| `replay` | L7 | 2025 대회 day-by-day 재현 |
| `train` | L4 | ML ranker 학습 (purged walk-forward) |
| `decide` | L8 | 일일 의사결정 + 근거 |

---

## 6. 계층 간 의존 규칙

1. **하위 계층은 상위를 모른다.** `data/` 는 `alpha/` 를 import 하지 않습니다.
2. **상위 계층은 하위 Protocol 에만 의존한다.** `AlphaModel` 은 `FeatureBuilder` 구현 세부를 모릅니다.
3. **look-ahead 차단은 여러 계층에 분산된다.** 각 계층이 자기 책임만 fail-closed 로 처리합니다.

| 계층 | look-ahead 방어 |
| --- | --- |
| data | raw 보존, `""` → None |
| universe | `date ≤ t` 슬라이스, ADV 캐시 |
| features | `assert_pit`, session grid |
| backtest | `decision_date` ≠ `execution_date` |

---

## 7. 두 종류의 백테스트

혼합 금지. 리포트에서 항상 분리 표기합니다.

| | Deployment | Structural |
| --- | --- | --- |
| universe | 현재 상장·유동성 충족 ETF | 각 시점 실제 존재 ETF |
| 기간 | 2024-01 ~ 현재 | 2018-01 ~ 현재 |
| 목적 | **이번 대회 전략 검증** | 아이디어 구조적 존재 여부 |
| survivorship bias | 의도적 허용 | 없음 |

## 8. Refactor R2-R5 Module Map (Updated)

```
src/
├── strategies/
│   ├── ids.py                # semantic StrategyId constants
│   ├── registry.py           # LEGACY_ALIASES, resolve_strategy_id, STRATEGIES
│   ├── baselines/
│   │   ├── core.py           # BuyAndHoldBaseline, B0-B5, M07 shims
│   │   └── portfolio.py      # P08-P19 portfolio factories
│   └── sticky/
│       ├── config.py         # load_overlay_mode, read_sticky_yaml_block (semantic-first)
│       ├── model.py          # StickyLeaderModel, StickyLeaderConfig
│       ├── overlays.py       # impulse/crash/abs-mom/same-leader overlays
│       └── factories.py      # P20-P28b factories, FACTORY_REGISTRY
├── cli/
│   ├── dispatch.py           # family_of, normalize_cli_model_arg, STICKY_*_HANDLERS
│   ├── constants.py          # CHAMPION_STRATEGY semantic
│   ├── main.py               # build_parser + SUBCOMMANDS
│   └── commands/
│       ├── backtest.py       # cmd_backtest (normalize semantic)
│       ├── decide.py         # cmd_decide
│       ├── config.py         # config-check, calendar
│       ├── data.py           # ingest, normalize
│       ├── universe.py       # universe
│       ├── features.py       # features
│       ├── replay.py         # replay
│       └── storage.py        # storage-migrate
└── tournament/
    ├── distribution_core.py  # ReturnDistribution, evaluate_adoption_gates
    ├── overlay_returns.py    # locked_window etc, execution_faithful
    ├── objective_core.py     # ObjectiveGateConfig, evaluate_objective_gates
    ├── championship.py       # evaluate_championship_adoption, field_relative_report
    └── adoption_reports.py   # p15/p16/p24/p25 reports
```
