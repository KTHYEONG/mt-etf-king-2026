# 03. Silver Panel — typed decode · canonical schema · fail-closed validation

**선행**: [02_krx_ingestion](02_krx_ingestion.md)
**상위**: [00_architecture.md](00_architecture.md)

---

## 1. Diagnosis & Invariants

### 1.1 이 계층이 막아야 하는 3가지 재앙

**재앙 1 — `""` 를 0으로 읽는 것.**
KRX 는 전 필드를 문자열로 주고 결측을 `""` 로 표현한다. `float("")` 는 예외지만, 흔한 방어 코드 `pl.col(x).cast(pl.Float64, strict=False)` 또는 `fillna(0)` 은 **가격 0** 을 만든다. 가격 0은 다음 날 수익률을 $+\infty$ 로 만들고, 그 종목이 momentum 랭킹 1위가 되어 포트폴리오 전체를 먹는다.

**INV-03-1**: 빈 문자열은 `None` 으로만 디코딩한다. 그리고 "빈 문자열(정상 결측)"과 "파싱 불가 문자열(데이터 오류)"을 **구분**한다 — 후자는 조용히 `None` 이 되면 안 되고 오류로 승격된다.

**재앙 2 — 휴장일을 거래일로 착각하는 것.**
`basDd=20260815` 는 1,163행을 반환하되 `TDD_CLSPRC` 가 전부 `""` 다. 행 개수 기반 판정은 통과한다.

**INV-03-2 (거래일 판정식)**: 세션 여부는 행 개수가 아니라 **유효 종가 비율**로 정의한다.

$$
\rho(d) = \frac{|\{r : \texttt{TDD\_CLSPRC}_r \neq \texttt{""}\}|}{|\{r\}|}, \qquad
\text{session}(d) \iff \rho(d) \ge \rho_{\min}
$$

$\rho_{\min} = 0.5$ (config). 실측: 거래일 $\rho = 1.0$, 휴장일 $\rho = 0.0$ — 임계값 근처에 관측이 없어 판정이 매우 안정적이다.

그리고 이 판정은 **XKRX 캘린더와 교차 검증**된다. 두 원천이 불일치하면 조용히 한쪽을 택하지 않고 `CRITICAL` 로 승격한다. 두 독립 원천이 어긋났다는 건 둘 중 하나에 대한 우리의 가정이 틀렸다는 뜻이고, 그 상태로 만든 백테스트는 신뢰할 수 없다.

**재앙 3 — 필드명 충돌.**
`/etp/etf_bydd_trd` 와 `/sto/stk_bydd_trd` 의 `ISU_CD` 는 **6자리 단축코드**지만, `/sto/stk_isu_base_info` 의 `ISU_CD` 는 **ISIN**(`KR7095570008`)이고 단축코드는 `ISU_SRT_CD` 다. 범용 `ISU_CD → ticker` 리네이밍은 마스터 조인을 전부 깨뜨린다.

**INV-03-3**: 필드 매핑은 **endpoint 별로 명시 선언**한다. 공통 리네이밍 규칙 금지.

### 1.2 회계 항등식 (무료 무결성 검증)

실측으로 정확히 성립함을 확인했다 (451060 / 2026-08-27):

$$
\texttt{MKTCAP} = \texttt{LIST\_SHRS} \times \texttt{TDD\_CLSPRC}
$$
$$
\texttt{INVSTASST\_NETASST\_TOTAMT} \approx \texttt{LIST\_SHRS} \times \texttt{NAV} \quad (\text{상대오차} \; 4\times10^{-8})
$$

이는 **공짜로 얻는 강력한 무결성 체크**다. 파싱 오류·자릿수 밀림·행 오정렬이 있으면 즉시 깨진다.

**INV-03-4**: 4개 필드가 모두 존재하는 행에 대해 상대오차 $\le 10^{-6}$ (시총), $\le 10^{-4}$ (순자산, NAV 반올림 허용) 를 요구한다. 위반 비율이 임계 초과면 `CRITICAL`.

### 1.3 괴리율 이상치 — robust 통계만 사용

실측 |disparity| 분포: p50 = **34bp**, p95 = 184bp, p99 = 292bp.
그러나 `265690 ACE 러시아MSCI(합성)` 은 종가 8,535 / NAV 48.38 → **+1,754,158bp**, `ACC_TRDVAL = 0`.

이 한 종목이 산술평균을 34bp에서 1,524bp로 끌어올린다. **평균 기반 임계값(예: mean ± 3σ)은 이 종목 때문에 정상 종목까지 통과시킨다.**

**INV-03-5 (INV-12 구체화)**: 이상치 판정은 median/MAD 로만 한다. 그리고 다음은 **삭제가 아니라 `is_tradable=False` 플래그**로 처리한다 — 행을 지우면 패널에 구멍이 생겨 momentum lookback 이 조용히 짧아진다.

| 조건 | 의미 |
| --- | --- |
| `trading_value == 0` | 당일 거래 부재 → 체결 불가 |
| `close is None` | 가격 미형성 |
| `\|disparity\| > max_abs_disparity` (기본 0.20) | NAV/가격 괴리 비정상 (거래정지·해외 청산 등) |
| `high < low` 또는 `close ∉ [low, high]` | OHLC 정합성 붕괴 |

### 1.4 복잡도

- 디코드: 행당 $O(1)$, 전체 $O(N)$. 2018~현재 ETF 패널 $N \approx 2{,}125 \times 1{,}100 \approx 2.3\text{M}$ 행.
- Polars lazy + `zstd` Parquet. 목표: 전체 재빌드 90초 이내, 증분 빌드 5초 이내.
- **주의**: bronze JSON 2,125개를 매번 전량 재파싱하면 I/O 가 지배한다. `--since` 증분 경로를 반드시 제공한다.

---

## 2. Architecture & Mitigation

```
 bronze/*.json ──► schema.py ──────► silver.py ──────► normalized/*.parquet
                  (필드맵·디코더)    (조립·증분)            │
                        │                                  ▼
                        └──────────► validation.py ──► ValidationReport
                                     (세션판정·항등식·        │
                                      robust 이상치)          ▼
                                                     CRITICAL → 빌드 실패
```

### 2.1 정규 스키마

**`etf_daily`** (핵심 테이블)

| 컬럼 | 타입 | 원본 | 비고 |
| --- | --- | --- | --- |
| `date` | Date | `BAS_DD` | |
| `ticker` | Utf8 | `ISU_CD` | 6자리 단축코드 |
| `name` | Utf8 | `ISU_NM` | |
| `close` `open` `high` `low` | Float64 | `TDD_*PRC` | nullable |
| `volume` | Int64 | `ACC_TRDVOL` | |
| `trading_value` | Int64 | `ACC_TRDVAL` | |
| `nav` | Float64 | `NAV` | |
| `market_cap` | Int64 | `MKTCAP` | |
| `net_assets` | Int64 | `INVSTASST_NETASST_TOTAMT` | AUM |
| `shares_outstanding` | Int64 | `LIST_SHRS` | **설정좌수 — flow 계산의 핵심** |
| `underlying_index_name` | Utf8 | `IDX_IND_NM` | 클러스터 키 |
| `underlying_index_close` | Float64 | `OBJ_STKPRC_IDX` | tracking difference 용 |
| `is_tradable` | Boolean | 파생 | validation 산출 |

**`index_daily`**: `date, index_class, index_name, close, open, high, low, volume, trading_value, market_cap`
`index_class` 는 endpoint 로부터 부여 (`KOSPI`/`KOSDAQ`/`KRX`). 실측상 `idx/*` 응답의 `IDX_CLSS` 도 동일 값을 담고 있으나, 첫 행(`코스피 (외국주포함)`)처럼 `CLSPRC_IDX` 가 `""` 인 행이 존재하므로 지수 레벨은 nullable 로 둔다.

**`stock_daily`**: `date, ticker, name, market, sector_type, close, open, high, low, volume, trading_value, market_cap, shares_outstanding`
KOSPI(944) + KOSDAQ(1,823) 를 하나의 테이블로 union 하며 `market` 으로 구분한다. **용도는 오직 market breadth 계산**이다.

### 2.2 디코더 계약

```
decode_optional_float("")          -> None
decode_optional_float("34970")     -> 34970.0
decode_optional_float("1,234.5")   -> 1234.5      # 쉼표 방어
decode_optional_float("-")         -> ValueError  # 조용한 None 금지
decode_optional_int("")            -> None
decode_optional_int("34970.0")     -> ValueError  # 정수 필드에 소수 = 스키마 오류
```

`""` 만 `None` 이고, 그 외 파싱 실패는 **전부 예외**다. 이것이 INV-03-1 의 핵심 — "관대한 파서"는 데이터 오류를 조용한 결측으로 바꿔 하류에서 발견 불가능하게 만든다.

### 2.3 검증 게이트

| ID | 게이트 | 심각도 | 판정 |
| --- | --- | --- | --- |
| V1 | 세션 분류 $\rho \ge 0.5$ | — | 비세션 날짜는 패널에서 제외 |
| V2 | 세션 판정 vs XKRX 불일치 | **CRITICAL** | 두 독립 원천 모순 |
| V3 | `(date, ticker)` 중복 | **CRITICAL** | 조인 폭발 |
| V4 | `market_cap = shares × close` 상대오차 > 1e-6 인 행 비율 > 0.1% | **CRITICAL** | 파싱/정렬 오류 |
| V5 | `net_assets ≈ shares × nav` 상대오차 > 1e-4 인 행 비율 > 1% | WARN | NAV 반올림 여유 |
| V6 | 캘린더 세션 대비 패널 결손일 존재 | WARN | 수집 미완 → 재수집 유도 |
| V7 | OHLC 정합성 위반 행 | WARN + `is_tradable=False` | |
| V8 | robust 괴리율 임계 초과 행 | WARN + `is_tradable=False` | |
| V9 | `trading_value == 0` | INFO + `is_tradable=False` | |

`CRITICAL` 이 하나라도 있으면 `SilverBuilder.build` 는 Parquet 을 쓰지 않고 실패한다 — 오염된 패널이 디스크에 남아 이후 모든 연구를 오염시키는 것보다 낫다.

### 2.4 증분 빌드

`build(dataset, start, end, mode)` 에서 `mode="incremental"` 이면 기존 Parquet 의 `max(date)` 이후 bronze 세션만 파싱해 append 한다. `mode="full"` 은 전량 재빌드. 재빌드 결과는 항상 동일해야 한다(**멱등성**).

---

## 3. Assumptions

- **A-1**: `ISU_CD` 는 ETF/주식 시세 endpoint 에서 6자리 단축코드다(실측). 단, 신규 종목에 `0131A0`, `0142D0` 처럼 **영문자를 포함한 코드**가 존재하므로 숫자 전용 검증을 하면 안 된다 → `^[0-9A-Z]{6}$`.
- **A-2**: `index_class` 는 endpoint 에서 부여한다. 응답의 `IDX_CLSS` 는 교차 검증용으로만 쓴다.
- **A-3**: `stock_daily` 는 breadth 전용이므로 `sector_type`(`SECT_TP_NM`) 이 `""` 여도 정상이다(KOSPI 다수가 공란).

---

## 4. Execution Target

```bash
uv run pytest tests/unit/data -q --tb=short
uv run python tools/agent_skills/lean_check.py --spec docs/specs/03_silver_panel_contract.json

uv run mt-etf normalize --dataset etf_daily --mode full
```
