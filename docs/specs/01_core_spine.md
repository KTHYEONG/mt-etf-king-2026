# 01. Core Spine — settings · calendar · paths · logging · CLI

**선행**: 없음 (최초 spec)
**상위**: [00_architecture.md](00_architecture.md)

---

## 1. Diagnosis & Invariants

### 1.1 왜 이것이 첫 spec 인가

이 시스템의 모든 look-ahead 방지 로직은 결국 **"t 시점에 무엇이 관측 가능했는가"** 라는 단일 질문으로 환원되고, 그 답은 **거래일 캘린더**가 정의한다. 캘린더가 계층마다 흩어지면(`pandas.bdate_range`, 하드코딩 휴일 목록, API 응답의 날짜 등) 계층 간 세션 정의가 어긋나고, 이는 조용한 off-by-one look-ahead 로 나타난다. 따라서 **캘린더를 단일 진입점으로 고정**하는 것이 1순위다.

### 1.2 핵심 불변식

**INV-CAL (단일 캘린더 원천)**
세션 날짜를 생성·이동·판정하는 코드는 `TradingCalendar` 하나뿐이다. 다른 계층은 `date` 산술을 직접 수행하지 않는다.

$$
\texttt{next\_session}(\texttt{previous\_session}(d)) = d \quad \forall d \in \text{Sessions}
$$

$$
\texttt{session\_count}(a, b) = |\{s \in \text{Sessions} : a \le s \le b\}|
$$

검증된 앵커 값(XKRX):

| 질의 | 값 |
| --- | --- |
| `session_count(2026-09-21, 2026-11-13)` | **36** (2026 대회) |
| `session_count(2025-09-22, 2025-11-14)` | **35** (2025 대회) |
| `session_count(2018-01-01, 2026-08-27)` | 2,125 |
| `session_count(2010-01-04, 2026-08-27)` | 4,101 |
| `is_session(2026-08-15)` | False (광복절, 토) |
| `is_session(2026-08-17)` | False (대체휴일, 월) |
| `is_session(2026-08-27)` | True |
| `previous_session(2026-08-18)` | **2026-08-14** |

이 표는 §2.7의 실측 API 커버리지와 정확히 일치한다 — `basDd=20260815` 호출이 1,163행을 반환하면서도 전 가격 필드가 `""` 였던 사실(TRAP-1)과 XKRX 의 비세션 판정이 서로를 교차 검증한다.

**INV-SECRET (비밀 비노출)**
API 키는 `SecretStr` 로만 보관하고 `repr`/`str`/로그/예외 메시지 어디에도 원문이 나타나지 않는다. `.env.enc` 는 sops 로 암호화되어 있고 평문 `.env` 는 `.gitignore` 대상이다.

**INV-PATH (루트 이탈 차단)**
모든 데이터 경로는 `data_root` 하위로 resolve 되어야 한다. endpoint 문자열이 외부에서 들어오므로 `../` 주입 가능성을 fail-closed 로 차단한다.

**INV-LOG (태그 스키마)**
DEBUG 이상의 구조화 로그는 `[SYS]|[DATA]|[ALGO]|[EVAL]` 4개 태그만 사용하고 `key=value` 평문 형식을 지킨다(`.agents/rules/logging.md`). `print()` 금지.

### 1.3 복잡도

- `sessions(a,b)`: XKRX 세션 인덱스 이진 탐색 → $O(\log n + k)$
- `TradingCalendar` 인스턴스는 프로세스당 1회 생성 후 캐시. XKRX 로딩은 ~1s 이므로 재생성 금지.

---

## 2. Architecture & Mitigation

### 2.1 계층 분리

```
src/core/settings.py       ← 환경변수 → 타입 검증된 Settings (pydantic-settings)
src/core/calendar.py       ← XKRX 래핑, 세션 산술의 유일한 원천
src/core/paths.py          ← bronze/silver/gold 경로 결정 + traversal guard
src/core/logging_setup.py  ← 태그 스키마 로거 구성
src/cli.py                 ← subcommand dispatcher (이후 모든 spec의 wiring 앵커)
```

`cli.py` 는 **확장점**이다. spec 02~08 은 각자의 핸들러를 `SUBCOMMANDS` 레지스트리에 등록한다. 이 구조 덕분에 새 기능이 추가되어도 `main()` 은 변하지 않는다.

### 2.2 Fail-closed 처리

| 실패 | 처리 |
| --- | --- |
| `KRX_OPENAPI_KEY` 없음 | `Settings` 생성 시 `ValidationError`. 기본값·빈 문자열 대체 금지 |
| 캘린더 범위 밖 날짜 질의 | `ValueError` (조용히 clamp 금지) |
| `previous_session` 이 캘린더 시작 이전 | `ValueError` |
| endpoint 경로에 `..` 또는 절대경로 | `ValueError` |
| 로그 디렉터리 생성 실패 | 콘솔 핸들러만으로 계속 (관측성 손실 < 실행 중단) |

### 2.3 CLI 설계

```
mt-etf config-check                       # 설정·경로 검증 (비밀 미출력)
mt-etf calendar --start … --end …         # 세션 수/목록 출력
```

두 명령 모두 실사용 도구이자 `Settings`/`TradingCalendar`/`DataPaths` 의 진짜 호출자다 — 고아 구현(orphaned implementation)이 발생하지 않는다.

### 2.4 Caller wiring

`pyproject.toml` 의 `[project.scripts]` 에 `mt-etf = "src.cli:main"` 를 등록한다. 이것이 시스템 전체의 단일 진입점 계약이다.

---

## 3. Assumptions

- **A-1**: `exchange_calendars` 의 `XKRX` 가 KRX 휴장일의 신뢰 가능한 원천이다. 근거: 2026-08-15/08-17 비세션 판정이 KRX API 의 공백 가격 응답과 일치함을 실측 확인.
- **A-2**: 캘린더 로드 범위는 `2009-01-01 ~ 현재+2년`. 데이터 시작(2010-01-04)보다 1년 여유를 두어 warmup lookback 이 범위를 벗어나지 않게 한다.
- **A-3**: `data_root` 기본값 `data/`, `log_root` 기본값 `logs/` (둘 다 repo 내부 — AGENTS.md §4 프로젝트 전용 경로 정책).

---

## 4. Execution Target

```bash
uv run pytest tests/unit/core tests/unit/test_cli.py -q --tb=short
uv run python tools/agent_skills/lean_check.py --spec docs/specs/01_core_spine_contract.json
```

## 5. 테스트 규약

`scenario_id` 는 테스트 docstring 에 **문자열 그대로** 포함한다(계약 검증기가 리터럴을 탐색한다).

```python
def test_tournament_2026_has_36_sessions() -> None:
    """SCENARIO-01-01: 2026 대회 구간의 세션 수는 정확히 36이어야 한다."""
```
