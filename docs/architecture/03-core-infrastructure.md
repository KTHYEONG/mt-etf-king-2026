# 03. Core Infrastructure — L0 계층

`src/core/` 는 프로젝트 전체가 공유하는 **기반 인프라**입니다. 데이터·alpha·백테스트 어디서든 동일한 설정·달력·경로·로깅을 사용합니다.

---

## 1. 모듈 구성

| 모듈 | 책임 |
| --- | --- |
| `settings.py` | 환경변수·YAML 로드, `Settings` 싱글톤 |
| `calendar.py` | XKRX 거래일, `next_session` / `prev_session` |
| `paths.py` | `DataPaths` — bronze/silver/gold/state 루트 |
| `logging_setup.py` | 구조화 로그, `[SYS][DATA][ALGO][EVAL]` 태그 |

---

## 2. Settings

### 2.1 로드 순서

```
환경변수 (KRX_AUTH_KEY 등)
  → configs/base.yaml
  → configs/{env}.yaml (선택)
  → CLI override (선택)
```

### 2.2 핵심 필드

| 필드 | 용도 |
| --- | --- |
| `krx.auth_key` | `SecretStr` — 로그·repr 에 노출 금지 |
| `krx.base_url` | `https://data-dbg.krx.co.kr/svc/apis` |
| `krx.rate_limit_rps` | API 호출 속도 제한 |
| `data.root` | `data/` 루트 (repo 내 상대 경로) |
| `calendar.exchange` | `"XKRX"` 고정 |
| `tournament.capital` | 10억 (1,000,000,000 KRW) |
| `tournament.start_date` / `end_date` | 2026 대회 기간 |

### 2.3 `get_settings()`

- `@lru_cache` 싱글톤
- 테스트에서는 `get_settings.cache_clear()` + env override 로 격리

---

## 3. TradingCalendar

`exchange_calendars` 기반 XKRX 달력입니다.

```python
cal = TradingCalendar()
cal.is_session(date(2026, 9, 21))   # True
cal.sessions_in_range(start, end)   # 36 sessions for 2026 tournament
cal.next_session(d)                 # 다음 거래일
cal.prev_session(d)                 # 이전 거래일
```

**모든 시계열 정렬·체결·rolling window 는 이 달력을 통해서만** 날짜를 해석합니다. `pd.bdate_range` 나 주말 제거 로직을 직접 쓰지 않습니다.

---

## 4. DataPaths

```python
paths = DataPaths(root=settings.data.root)
paths.bronze("krx/etp/etf_bydd_trd", "2026/20260827.json")
paths.silver("etf_daily.parquet")
paths.gold("etf_features.parquet")
paths.state("krx_quota.json")
```

- 디렉터리 자동 생성 (`mkdir -p` 동등)
- bronze 경로는 `{dataset}/{year}/{YYYYMMDD}.json` 규칙

---

## 5. Logging

### 5.1 태그 체계

| 태그 | 사용 계층 | 예시 |
| --- | --- | --- |
| `[SYS]` | core, CLI | 설정 로드, 시작/종료 |
| `[DATA]` | data | ingest, normalize, validation |
| `[ALGO]` | features, alpha, portfolio | feature 빌드, score |
| `[EVAL]` | backtest, tournament | 백테스트 결과, 분포 |

### 5.2 원칙

- `auth_key` 값은 **절대 로그에 남기지 않음**
- validation CRITICAL 은 `ERROR` + exit code ≠ 0
- 일일 `decide` 출력은 사람이 읽을 수 있는 요약 + JSON artifact

---

## 6. CLI Spine

`src/cli.py` 는 `SUBCOMMANDS` dict 에 subcommand handler 를 등록합니다.

```python
SUBCOMMANDS: dict[str, Callable] = {
    "config-check": cmd_config_check,
    "calendar": cmd_calendar,
    # ... 단계별 추가
}
```

### 6.1 `config-check`

시작 전 필수 검증:

- `KRX_AUTH_KEY` 존재
- `data/` 쓰기 가능
- XKRX calendar 로드 성공
- 대회 기간 36 sessions 확인

### 6.2 구현 순서 (spec 01)

1. `settings` + `paths` + `calendar`
2. `logging_setup`
3. `cli` — `config-check`, `calendar` 만 먼저
4. 이후 spec 02~08 에서 subcommand 점진 추가

---

## 7. 패키지 Skeleton

현재 `src/` 에는 계층별 `__init__.py` 만 존재합니다. spec 01 구현 시 `core/` 모듈부터 채웁니다.

```
src/
├── __init__.py
├── cli.py          # (구현 예정)
└── core/
    ├── __init__.py
    ├── settings.py
    ├── calendar.py
    ├── paths.py
    └── logging_setup.py
```
