# 03. Core Infrastructure — L0 계층

`src/core/` 는 프로젝트 전체가 공유하는 **기반 인프라**입니다. 데이터·alpha·백테스트 어디서든 동일한 설정·달력·경로·로깅을 사용합니다.

---

## 1. 모듈 구성

| 모듈 | 책임 |
| --- | --- |
| `settings.py` | 환경변수·`.env.enc`(sops) 로드, `Settings` 싱글톤 |
| `sops_env.py` | sops 복호화, `SopsDotEnvSettingsSource` |
| `calendar.py` | XKRX 거래일, `next_session` / `previous_session` |
| `paths.py` | `DataPaths` — bronze/silver/gold/state 루트 |
| `logging_setup.py` | 구조화 로그, `[SYS][DATA][ALGO][EVAL]` 태그 |

---

## 2. Settings

### 2.1 비밀(Secrets) 로드 — `.env.enc` + sops

API 키 등 비밀은 **평문 `.env` 파일을 만들지 않습니다.** 저장소에 커밋된 `.env.enc` 를 런타임에 sops 로 메모리 복호화합니다.

```
프로세스 환경변수 (KRX_OPENAPI_KEY 등)     ← 최우선
  → .env.enc (sops -d, stdout → 메모리)   ← 기본
```

| 항목 | 값 |
| --- | --- |
| 암호화 파일 | `.env.enc` (dotenv 형식, sops+age) |
| 복호화 도구 | `sops` CLI (PATH 필수) |
| age 키 | `~/.config/sops/age/keys.txt` (로컬, git 제외) |
| 경로 오버라이드 | `MT_ETF_ENV_ENC=/path/to/.env.enc` |
| 평문 `.env` | **생성·사용 안 함** (`.gitignore` 대상) |

구현 (`src/core/sops_env.py`):

1. `decrypt_env_enc()` — `sops -d --input-type dotenv --output-type dotenv` 를 subprocess 로 호출, **stdout 만** 사용 (디스크 미기록)
2. `SopsDotEnvSettingsSource` — 복호화 결과를 pydantic-settings dotenv 소스로 주입
3. `decrypt_env_enc` 는 `@lru_cache` — 프로세스당 1회 복호화

복호화 실패(sops 미설치, age 키 없음, 파일 없음) 시 빈 dict → `krx_openapi_key` 누락 → `ValidationError` (fail-closed).

### 2.2 비시크릿 설정 로드 (향후)

```
configs/base.yaml
  → configs/{env}.yaml (선택)
  → CLI override (선택)
```

현재 spec 01 구현 범위에서는 YAML merge 는 미연결. `Settings` 는 secrets + 경로/쿼터 등 pydantic 필드만 담당.

### 2.3 핵심 필드

| 필드 | 용도 |
| --- | --- |
| `krx_openapi_key` | `SecretStr` — 로그·repr 에 노출 금지 |
| `fred_api` / `ecos_api` / `opendart_api_key` | `SecretStr \| None` |
| `krx_base_url` | `https://data-dbg.krx.co.kr/svc/apis` |
| `data_root` | `data/` 루트 |
| `log_root` | `logs/` 루트 |
| `daily_call_quota` | KRX 일일 호출 상한 (기본 8,000) |
| `calendar_name` | `"XKRX"` 고정 |

### 2.4 `get_settings()`

- `@lru_cache` 싱글톤
- 테스트에서는 `clear_settings_caches()` + env override 로 격리
- `get_secret_value()` 는 API 클라이언트 **생성 시 1회**만 호출; 로그·repr·예외에 원문 금지

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

- `Settings` 로드 성공 (`.env.enc` sops 복호화 또는 환경변수)
- credential 존재 여부 **boolean 만** 출력 (`krx_openapi_key=True` 등)
- `DataPaths` probe 성공
- XKRX calendar 로드 성공

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
├── cli.py
└── core/
    ├── __init__.py
    ├── settings.py
    ├── sops_env.py       # sops 복호화 + Settings dotenv 소스
    ├── calendar.py
    ├── paths.py
    └── logging_setup.py
```
