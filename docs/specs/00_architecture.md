# 00. Tournament Quant Research System — Master Architecture

> 머니투데이 제3회 ETF 투자왕(2026-09-21 ~ 2026-11-13) 우승 확률 극대화를 위한 연구 시스템 설계.
> 본 문서는 하위 spec 01~08의 상위 계약(umbrella contract)이며, 모든 하위 spec은 여기 정의된 **Global Invariant(INV-x)** 를 위반할 수 없다.

---

## 1. Objective Function — 무엇을 최적화하는가

### 1.1 목적 정의

일반 퀀트의 `maximize Sharpe` / `minimize MDD` 가 **아니다.**

$$
\text{maximize} \quad \mathbb{E}[\text{Prize}] = 1000 \cdot P(\text{rank}{=}1) + 500 \cdot P(\text{rank}{=}2) + 100 \cdot P(\text{category top}=1)
$$

참가자 분포 $F$ 를 모르므로 $P(\text{rank}{=}1)$ 을 직접 추정하지 않는다. 대신 **구조적으로 등가인 대리 목적**을 쓴다.

$N$ 명의 경쟁자 수익률이 i.i.d. $F$ 라면 우승 조건은 $R > \max_{i \le N} R_i$ 이고, 우승 임계값 $R^\*$ 는 $F$ 의 $(1 - 1/N)$ 분위수다. 즉:

$$
P(\text{rank}{=}1) = \int P(R > \theta) \, dG_N(\theta), \qquad G_N(\theta) = F(\theta)^N
$$

따라서 **$\theta$ 격자에 대한 exceedance probability $P(R_{36d} > \theta)$ 곡선 전체**가 우리가 추정해야 할 대상이며, 단일 스칼라 지표(mean, Sharpe)로 압축하면 목적 함수의 정보가 파괴된다.

**INV-OBJ**: 모든 전략 평가는 스칼라가 아니라 **36거래일 수익률 분포 전체**를 산출해야 한다. 요약 지표는 분포에서 파생될 뿐 대체할 수 없다.

### 1.2 의사결정 관련 구간 (decision-relevant zone)

2025년 제2회 대회 관측값(5주차 선두 +72.28%, 최종 우승 +47.82%)을 **단일 표본 앵커**로 사용한다. 이는 분포 추정치가 아니라 임계값의 자릿수(order of magnitude)를 잡기 위한 참조점이다.

- $\theta \in [30\%, 60\%]$ 를 **대상(1위) 관련 구간**으로 본다.
- $\theta \in [20\%, 40\%]$ 를 **최우수상(2위) 관련 구간**으로 본다.
- $\theta < 15\%$ 구간의 확률 개선은 상금 기대값을 거의 올리지 못한다.

**결과적 설계 함의**: 평균 수익률을 1%p 올리는 개선보다, $P(R > 40\%)$ 를 1%p 올리는 개선이 우선한다. 단, §1.3 제약을 만족해야 한다.

### 1.3 Tail 최적화의 반대 제약

`5% 확률 +100% / 95% 확률 -30%` 같은 전략은 tail은 크지만 우승 확률이 높지 않을 수 있다. 우승은 **max order statistic 초과** 문제이므로, 경쟁자 상위 tail도 같이 두꺼워지는 강세장에서 우리만 하방으로 빠지면 상금 기대값은 0이다.

**INV-TAIL**: 전략 채택 기준은 `P(R > θ)` 단독이 아니라 다음 3개를 동시에 본다.
1. $P(R > \theta)$ for $\theta \in \{10, 20, 30, 40, 50\}\%$
2. 분포 하위 꼬리: $q_{05}$, worst 5% 평균, MDD
3. 시장 조건부 성능: 상승장/횡보장/하락장 sub-sample에서의 $P(R > \theta)$

### 1.4 대회 기간은 40일이 아니라 **36 거래일**

XKRX 캘린더 기준 2026-09-21 ~ 2026-11-13 = **36 sessions** (검증 완료). 2025년 대회(2025-09-22 ~ 2025-11-14)는 **35 sessions**.

**INV-HORIZON**: 대회 기간은 상수 `40`이 아니라 `TradingCalendar.session_count(start, end)` 로 계산한다. rolling window 길이, label horizon, aggression schedule 전부 이 값에서 파생된다.

---

## 2. 경험적으로 검증된 데이터 사실 (Empirical Ground Truth)

아래는 **추정이 아니라 실제 API 호출로 확인한 사실**이다. 하위 spec의 모든 계약은 이 사실 위에 세워진다.

### 2.1 KRX Open API 표면

| 항목 | 확인된 값 |
| --- | --- |
| Base URL | `https://data-dbg.krx.co.kr/svc/apis` |
| 인증 | HTTP header `AUTH_KEY: <key>` (`.env.enc` 의 `KRX_OPENAPI_KEY`) |
| 메서드 | GET(query) / POST(json body) 모두 200 |
| 응답 루트 | `{"OutBlock_1": [...]}` |
| 값 타입 | **전 필드 string**. 결측은 `0`이 아니라 `""` |
| 파라미터 | `basDd` (YYYYMMDD) **단 하나만 동작** |
| 미구독 endpoint | HTTP 401 + `{"respMsg": "Unauthorized API Call", "respCode": "401"}` |
| 없는 endpoint | HTTP 404 + `respMsg` |
| 처리량 | 순차 5콜 3.08s ≈ **0.62 s/call** |

### 2.2 치명적 함정 3가지

**TRAP-1 — 휴장일이 빈 배열이 아니다.**
`basDd=20260815`(광복절) 호출 결과: `n_rows=1163`, 그러나 `TDD_CLSPRC=""`, OHLCV 전부 `""`. 반면 `LIST_SHRS`, `IDX_IND_NM` 은 값이 채워져 있다.
→ `len(rows) > 0` 로 거래일을 판정하면 **휴장일이 정상 거래일로 통과**한다. `float("")` 는 예외, `fillna(0)` 은 가격 0 이라는 재앙.

**TRAP-2 — 미래/불량 날짜가 에러가 아니다.**
`basDd=20261231`, `20090101`, `basDd="bad"` 모두 HTTP **200 + `OutBlock_1: []`**. 에러 신호가 상태 코드에 없다.

**TRAP-3 — 미지원 파라미터가 조용히 무시된다.**
`{"basDd": "20260827", "isuCd": "069500"}` → 필터링 없이 1163행 전체 반환. `{"strtDd","endDd"}` → 0행.
→ 서버측 필터링에 의존 금지. 범위 질의는 **존재하지 않는다**. 1 call = 1 session 전체 단면.

### 2.3 ETF 일별매매정보 (`/etp/etf_bydd_trd`) 실제 필드

```
BAS_DD, ISU_CD(6자리 단축코드), ISU_NM, TDD_CLSPRC, CMPPREVDD_PRC, FLUC_RT,
NAV, TDD_OPNPRC, TDD_HGPRC, TDD_LWPRC, ACC_TRDVOL, ACC_TRDVAL, MKTCAP,
INVSTASST_NETASST_TOTAMT, LIST_SHRS, IDX_IND_NM, OBJ_STKPRC_IDX, CMPPREVDD_IDX, FLUC_RT_IDX
```

**ISU_CD 의미 충돌**: ETF/주식 시세 endpoint 에서 `ISU_CD` 는 6자리 단축코드지만, `/sto/stk_isu_base_info` 에서 `ISU_CD` 는 **ISIN**(`KR7095570008`)이고 단축코드는 `ISU_SRT_CD` 다. → endpoint별 명시적 매핑 필수, 범용 리네이밍 금지.

**회계 항등식 검증 완료** (451060, 2026-08-27):
- `MKTCAP = LIST_SHRS × TDD_CLSPRC` → 12,600,000 × 34,970 = 440,622,000,000 ✓
- `INVSTASST_NETASST_TOTAMT ≈ LIST_SHRS × NAV` → 12,600,000 × 35,057.03 = 441,718,578,000 vs 441,718,559,589 (오차 4e-8) ✓

### 2.4 이 필드 조합이 열어주는 3개의 고유 신호

**(a) 진짜 자금유입 (creation/redemption flow)** — next.md §25의 `AUM change ≠ capital inflow` 문제를 정확히 해결한다.

$$
\Delta \text{AUM}_t = \underbrace{\Delta L_t \cdot \text{NAV}_t}_{\text{순수 자금흐름}} + \underbrace{L_{t-1} \cdot \Delta \text{NAV}_t}_{\text{성과효과}}, \qquad L_t = \texttt{LIST\_SHRS}_t
$$

실측(2026-08-26→27): 1163개 중 **289개**에서 설정좌수 변화 발생, 총 설정 1.744조원 / 환매 0.671조원. 가격효과와 완전히 분리된 실제 flow 관측 가능.

**(b) 괴리율(disparity)** = $(\texttt{TDD\_CLSPRC} - \texttt{NAV}) / \texttt{NAV}$
실측 분포: median 2.09bp, |disparity| p50=34bp, p95=184bp, p99=292bp.
**이상치 사례**: `265690 ACE 러시아MSCI(합성)` 종가 8,535 / NAV 48.38 → **+17,542%**, `ACC_TRDVAL=0`. 단일 종목이 평균을 1,524bp로 오염시킨다.
→ robust 통계(median/MAD)만 사용하고, `ACC_TRDVAL == 0` 또는 |disparity| 임계 초과 종목은 **fail-closed 배제**.

**(c) 추종지수 링크** `IDX_IND_NM` + `OBJ_STKPRC_IDX`
공식·point-in-time 한 ETF→기초지수 매핑이 매일 제공된다. tracking difference($R_{ETF} - R_{IDX}$)를 직접 계산 가능하고, **동일 지수 = 동일 베팅**이라는 결정론적 중복제거 키를 얻는다(next.md §38의 상관계수 추정 불필요).

### 2.5 universe 구조 실측 (2026-08-27)

| 지표 | 값 |
| --- | --- |
| ETF 종목 수 | 1,163 |
| distinct `IDX_IND_NM` | 880 (**89.4%가 singleton**) |
| 레버리지(이름 기준) | 72 |
| 인버스(이름 기준) | 41 |
| 합성 `(합성` | 99 |
| 환헤지 `(H)` | 63 |
| 액티브 | 322 |
| 운용사 브랜드 | KODEX 241, TIGER 231, RISE 142, ACE 112, PLUS 84, SOL 77, KIWOOM 70, HANARO 50, 1Q 27, KoAct 25 … |

**핵심 해석**: `IDX_IND_NM` 은 880종/1163개로 너무 세분화되어 **sector 분류가 아니다.** 이는 완벽한 *중복제거 키*이지 *테마 그룹*이 아니다. → 2단계 taxonomy 필요(§4.3).

### 2.6 유동성 — 이 프로젝트의 진짜 병목

초기자금 10억원 기준, 일거래대금 분위수:

| 분위 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 거래대금(원) | 4.9M | 36M | **292M** | 2.16B | 10.5B | 24.1B | 401B |

`max_order_to_adv` 제약별 **거래 가능 ETF 수**:

| 주문/ADV 상한 | 1% | 2% | 5% | 10% |
| --- | --- | --- | --- | --- |
| 10억 전액 집중 시 가능 종목 | **26** | 41 | 65 | 121 |

중앙값 ETF의 하루 전체 거래대금(2.92억)이 우리 자본(10억)의 **29%** 에 불과하다. 즉 유동성 필터는 부가 조건이 아니라 **universe를 1,163 → 26~121로 줄이는 1차 축소기**다.

**INV-LIQ**: universe 축소는 (1) 존재 → (2) warmup → (3) **유동성** → (4) 대회 적격성 순서로 적용하며, 유동성 파라미터는 `{1,2,5,10}%` 스트레스 그리드로 항상 동반 보고한다.

### 2.7 데이터 커버리지 & 수집 예산

ETF 패널 종목 수 추이: 2010-01-04 **50** → 2015 **172** → 2020 **450** → 2023 **666** → 2025-09 **1,019** → 2026-08 **1,163**.

→ 현재 존재하는 ETF만으로 2015년을 백테스트하면 **survivorship bias가 지배적**이다. next.md §18의 Deployment/Structural backtest 분리는 선택이 아니라 필수.

| 구간 | XKRX sessions |
| --- | --- |
| 2010-01-04 ~ 2026-08-27 | 4,101 |
| 2018-01-01 ~ 2026-08-27 | 2,125 |

수집 비용: 1 session = 1 call/endpoint. 6개 endpoint 전량 backfill = **24,606 calls**. 0.62s/call 순차 ≈ 4.2시간, 동시성 6 적용 시 ≈ 45분. 단, 일일 호출 쿼터(비공식 10,000/day 보고)가 미확인이므로 **resumable + quota-aware** 설계 필수.

### 2.8 구독 가능 endpoint 실측

| endpoint | 상태 | 용도 |
| --- | --- | --- |
| `/etp/etf_bydd_trd` | ✅ 200 (1,163행) | **핵심** ETF 패널 |
| `/sto/stk_bydd_trd` | ✅ 200 (944행) | KOSPI 종목 — market breadth |
| `/sto/ksq_bydd_trd` | ✅ 200 (1,823행) | KOSDAQ 종목 — market breadth |
| `/sto/stk_isu_base_info` | ✅ 200 | 주식 `LIST_DD` (ETF는 없음) |
| `/sto/ksq_isu_base_info` | ✅ 200 | 동일 |
| `/idx/kospi_dd_trd` | ✅ 200 (51행) | KOSPI 시리즈 지수 — regime |
| `/idx/kosdaq_dd_trd` | ✅ 200 (40행) | KOSDAQ 시리즈 지수 — regime |
| `/idx/krx_dd_trd` | ✅ 200 (40행) | KRX 통합 지수 |
| `/idx/bon_dd_trd` | ✅ 200 (3행) | 채권지수 — risk-off 판정 |
| `/drv/fut_bydd_trd` | ✅ 200 (385행) | 선물(`SPOT_PRC`, 미결제약정) |
| `/etp/etn_bydd_trd`, `/etp/elw_bydd_trd` | ❌ 401 | 미구독(불필요) |
| `/idx/drvprod_dd_trd`, `/gen/oil_bydd_trd` | ❌ 401 | 미구독 |
| **`/etp/etf_isu_base_info`** | ❌ **404 (미존재)** | ETF 기본정보 API 자체가 없음 |
| **`/etp/etf_pdf`** | ❌ **404 (미존재)** | **PDF(구성종목) API 자체가 없음** |

### 2.9 위 사실이 강제하는 2개의 아키텍처 결정

**DEC-A — ETF 상장일은 패널에서 유도한다.**
ETF용 base_info API가 존재하지 않으므로 `listing_date` 를 조회할 방법이 없다. 대신 일별 패널의 종목 등장/소멸로 `first_seen_date` / `last_seen_date` 를 유도한다. 실측 검증: 2026-08-26 → 08-27 사이 `489010` 이 패널에서 사라짐(상장폐지 이벤트 포착 성공).
이 방식은 오히려 **정의상 point-in-time 이 보장**된다 — 마스터 테이블의 사후 수정 위험이 없다.

**DEC-B — 구성종목 breadth 는 primary source 로 불가능하다.**
PDF API가 없으므로 next.md §26(ETF 내부 구성종목 breadth)은 KRX Open API만으로 구현 불가. 세 가지 선택지 중:
1. **market-level breadth** (KOSPI 944 + KOSDAQ 1,823 = 2,767종목) → **완전 가능. 즉시 채택** (regime 판정용).
2. **cluster-level breadth** (같은 테마에 속한 ETF 집합 내부의 % above MA20 등) → **완전 가능. 채택** (sector leadership 품질 판정용).
3. constituent-level breadth (PDF 필요) → pykrx(웹 스크래핑) 의존. **Stage 3 이후 optional adapter 로 유예.**

---

## 3. 계층 아키텍처

```
                    ┌───────────────────────────────────────┐
 L0  core/          │ settings · calendar · paths · logging │  ← 모든 계층의 기반
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L1  data/          │ providers → bronze(raw) → silver(정규) │  raw 불변 · 타입 fail-closed
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L2  universe/      │ instrument master · PIT universe      │  "오늘 실제로 살 수 있는가"
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L3  features/      │ momentum trend vol flow breadth regime │  PIT guard 강제
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L4  alpha/         │ baselines · sector leadership · rank   │  Signal
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L5  portfolio/     │ cluster dedup · sizing · constraints   │  Signal ≠ Portfolio
                    │ state machine (exit / re-entry)        │
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L6  backtest/      │ next-open execution · costs · metrics  │  look-ahead 차단
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L7  tournament/    │ rolling-36D · bootstrap · MC · replay  │  분포 산출
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
 L8  reporting/     │ daily decision · rationale            │
                    └───────────────────────────────────────┘
```

### 3.1 모듈 맵

```
src/
├── cli.py                       # 단일 진입점 (subcommand dispatcher)
├── core/
│   ├── settings.py              # Settings, get_settings
│   ├── calendar.py              # TradingCalendar (XKRX)
│   ├── paths.py                 # DataPaths (bronze/silver/gold)
│   └── logging_setup.py         # configure_logging ([SYS][DATA][ALGO][EVAL])
├── data/
│   ├── providers/
│   │   ├── base.py              # MarketDataProvider Protocol
│   │   └── krx.py               # KRXOpenAPIProvider, KRXEndpoint
│   ├── bronze.py                # BronzeStore (raw JSON 불변 저장)
│   ├── backfill.py              # BackfillPlanner (resumable, quota-aware)
│   ├── schema.py                # 필드 매핑 · 타입 계약
│   ├── silver.py                # SilverBuilder (Parquet 패널)
│   └── validation.py            # PanelValidator (fail-closed 게이트)
├── universe/
│   ├── instruments.py           # InstrumentMaster (PIT 속성 유도)
│   ├── taxonomy.py              # 2단계 분류: index_key + theme
│   ├── provider.py              # PointInTimeUniverse
│   └── tournament.py            # TournamentRules (미확정 규칙 = Unknown)
├── features/
│   ├── pit.py                   # assert_pit — look-ahead 차단 게이트
│   ├── momentum.py trend.py volatility.py flow.py breadth.py regime.py
│   ├── crosssec.py              # percentile rank · z-score · acceleration
│   └── builder.py               # FeatureBuilder (조립)
├── alpha/
│   ├── base.py                  # AlphaModel Protocol
│   ├── baselines.py             # B0~B5
│   └── leadership.py            # SectorLeadershipModel   (spec 07)
├── portfolio/
│   ├── selection.py sizing.py constraints.py
│   └── state.py                 # 6-state machine          (spec 08)
├── backtest/
│   ├── engine.py execution.py costs.py metrics.py
├── tournament/
│   ├── simulator.py             # rolling-36D
│   ├── distribution.py          # exceedance curve · bootstrap
│   ├── replay.py                # 2025 대회 day-by-day 재현
│   ├── montecarlo.py policy.py  # (spec 08)
└── reporting/
    └── dashboard.py
```

---

## 4. Global Invariants

하위 spec의 모든 계약은 아래를 위반할 수 없다. 위반 시 `/implement` 는 즉시 `/spec` 으로 escalate 한다.

| ID | 불변식 | 강제 위치 |
| --- | --- | --- |
| **INV-1** | 결측은 `None` 이다. `""` → `0` 변환 절대 금지. | `data/schema.py` 디코더 |
| **INV-2** | 거래일 판정은 행 개수가 아니라 **`TDD_CLSPRC` 비공백 비율**로 한다. XKRX 세션 목록과 불일치 시 fail-closed. | `data/validation.py` |
| **INV-3** | `data/raw/**` 는 write-once. 재수집은 새 파일로만. | `data/bronze.py` |
| **INV-4** | 모든 feature 함수는 `decision_date` 를 받고, `date > decision_date` 행이 계산에 기여하지 않음을 보장한다. | `features/pit.py` |
| **INV-5** | signal은 `close(t)` 로 계산하고 체결은 `open(t+1)`. 동일 세션 종가 체결 금지. | `backtest/execution.py` |
| **INV-6** | universe는 point-in-time. `first_seen ≤ t ≤ last_seen` 그리고 warmup 충족. | `universe/provider.py` |
| **INV-7** | $\sum w_i + w_{cash} = 1.0 \pm 10^{-6}$ | `portfolio/constraints.py` |
| **INV-8** | 미확정 대회 규칙은 추측하지 않는다. `Unknown` sentinel + 시나리오 파라미터로만 표현. | `universe/tournament.py` |
| **INV-9** | 레버리지 ETF 수익률은 `기초 × 배수` 로 합성하지 않는다. 실제 ETF 가격 시계열만 사용. | `features/*`, `backtest/*` |
| **INV-10** | 모든 신규 전략은 baseline B0~B5 대비 동일 프로토콜로 비교된 결과 없이는 채택 불가. | `tournament/simulator.py` |
| **INV-11** | 가중치·임계값은 코드 상수가 아니라 `configs/*.yaml` 파라미터. | 전 계층 |
| **INV-12** | robust 통계(median/MAD)만 이상치 판정에 사용. 평균 기반 임계값 금지. | `data/validation.py` |

---

## 5. 두 종류의 백테스트 (혼합 금지)

| | Deployment Backtest | Structural Backtest |
| --- | --- | --- |
| universe | 현재 상장·유동성 충족 ETF | 각 시점에 실제 존재했던 ETF |
| 기간 | 2024-01 ~ 현재 | 2018-01 ~ 현재 (2010까지 확장 가능) |
| 목적 | **이번 대회에 쓸 전략의 검증** | 아이디어가 구조적으로 존재했는가 |
| survivorship bias | 있음(의도적으로 허용) | 없음 |
| 사용처 | 파라미터 최종 결정 | 아이디어 채택/기각 |

**INV-BT**: 두 결과를 하나의 표에 합산하지 않는다. 리포트에서 항상 분리 표기한다.

---

## 6. 미확정 항목 리스크 레지스터

HTS(코스콤 모의투자) 배포 전까지 **알 수 없는 것**. 전부 config parameter 로만 표현하며 기본값은 `Unknown`.

| ID | 항목 | 현재 상태 | 시스템 대응 |
| --- | --- | --- | --- |
| R-1 | 부문별 허용 ETF 목록 | 미공개 | `configs/tournament_2026.yaml: universe.manifest_path = null` → null이면 전체 ETF, 경고 로그 |
| R-2 | 자율형 레버리지/인버스 허용 여부 | **불명확** (보도자료는 "레버리지·테마 등 고변동 상품보다 건전한 전략 유도" 취지만 언급) | `leverage_allowed: Unknown`. Unknown이면 레버리지 포함/제외 **두 시나리오 모두** 산출 |
| R-3 | 수수료·세금 | 미확정 | `commission_bps: null` → null이면 {0, 1.5, 3, 15}bp 그리드 |
| R-4 | 분배금 처리 (PR vs TR) | 미확정 | 기본 PR, TR 시나리오 병행 |
| R-5 | 수익률 계산식(미실현손익 포함 여부) | 미확정 | 평가금액 기준 기본 |
| R-6 | 주문 종류·체결 방식 | 미확정 | next-open 시장가 기본, LP 스프레드 시나리오 |
| R-7 | 종목당 최대 보유 비중 | 미확정 | `max_weight: null` → null이면 1.0, 그리드 {0.3,0.5,1.0} |
| R-8 | 현금 100% 허용 여부 | 미확정 | 허용 가정, 미허용 시나리오 병행 |
| R-9 | API 일일 호출 쿼터 | 비공식 10,000/day | `daily_call_quota` 설정값 + 소진 시 resumable 중단 |
| R-10 | 참가자 수 및 수익률 분포 | 미지 | MC는 aggressive/normal/weak 3 시나리오 **stress test 용도로만** |

**INV-8 재확인**: 위 어느 항목도 코드에 하드코딩하지 않는다. `Unknown` 은 "임의 기본값"이 아니라 "**두 시나리오 모두 산출**"을 트리거하는 값이다.

---

## 7. 실행 계획 — 잔여 24일 (기준일 2026-08-28)

대회 시작 2026-09-21, 참가 접수 마감 2026-09-16.

| 주차 | 기간 | Spec | 산출물 | 완료 판정 |
| --- | --- | --- | --- | --- |
| **W1** | 08/28–09/03 | 01, 02, 03 | core spine · KRX 수집 · silver 패널 | 2018-01-01~현재 ETF/지수/주식 패널 Parquet 생성, validation 전부 PASS |
| **W2** | 09/04–09/10 | 04, 05 | PIT universe · feature engine | 임의 날짜 t 에 대해 universe + 전체 feature 프레임이 look-ahead 없이 생성 |
| **W3** | 09/11–09/16 | 06 | backtest · rolling-36D · baseline B0~B5 · **2025 replay** | B0~B5 의 36일 수익률 분포와 `P(R>θ)` 곡선 산출. 2025 대회 replay 리포트 완성 |
| **W4** | 09/17–09/20 | 07, 08 | leadership engine · portfolio policy · daily decision CLI | 매일 `mt-etf decide --date` 로 포트폴리오 + 근거 출력. 전략 **freeze** |
| **대회중** | 09/21–11/13 | 08(overlay) | tournament aggression overlay | 실 순위 입력 기반 risk_multiplier 조정 |

**우선순위 원칙**: W3까지 끝나면 최소한 "검증된 baseline"으로 참가할 수 있다. W4가 지연되면 baseline으로 출전하고 대회 중 개선한다. **ML(Stage 5)은 이번 대회 범위에서 제외**한다 — 24일 안에 walk-forward 검증까지 끝낼 수 없고, next.md §34의 기준(rule baseline 초과 입증)을 만족시킬 시간이 없다.

### 7.1 backfill 우선순위 (쿼터 제약 대응)

1. `etp/etf_bydd_trd` 2018-01-01~현재 (2,125 calls) — **없으면 아무것도 못 함**
2. `idx/kospi_dd_trd`, `idx/kosdaq_dd_trd` 동일 구간 (4,250 calls) — regime
3. `sto/stk_bydd_trd`, `sto/ksq_bydd_trd` 동일 구간 (4,250 calls) — market breadth
4. 2010~2017 확장 (백그라운드, structural backtest 용)

---

## 8. 채택/기각 결정 게이트

next.md §68의 검증 순서를 실행 가능한 게이트로 고정한다.

```
Hypothesis → Feature → Simple Strategy → Structural BT → Deployment BT
          → Rolling-36D 분포 → Robustness grid → 2025 Replay → Accept/Reject
```

**기각 기준 (하나라도 해당하면 폐기)**
- G-1: Structural backtest 에서 `P(R>30%)` 가 B1(Top1 20D momentum) 대비 개선 없음
- G-2: 파라미터 ±30% 섭동에서 `P(R>30%)` 가 50% 이상 붕괴 (과적합)
- G-3: 유동성 `max_order_to_adv` 를 1%→5% 로 바꿀 때 성과 순위가 뒤집힘 (허상 유동성)
- G-4: 2025 replay 에서 대회 시작 전 데이터만으로 주도 섹터를 포착하지 못함
- G-5: worst 5% 시나리오에서 -40% 이하

**INV-10 재확인**: "in-sample 성능이 좋아졌다"는 채택 사유가 될 수 없다.

---

## 9. 하위 Spec 목록 및 의존 순서

| # | Spec | 상태 | 선행 |
| --- | --- | --- | --- |
| 01 | [core_spine](01_core_spine.md) | contract 확정 | — |
| 02 | [krx_ingestion](02_krx_ingestion.md) | contract 확정 | 01 |
| 03 | [silver_panel](03_silver_panel.md) | contract 확정 | 02 |
| 04 | [pit_universe](04_pit_universe.md) | contract 확정 | 03 |
| 05 | [feature_engine](05_feature_engine.md) | contract 확정 | 04 |
| 06 | [research_harness](06_research_harness.md) | contract 확정 | 05 |
| 07 | [leadership_engine](07_leadership_engine.md) | **blueprint only** | 06 결과 |
| 08 | [portfolio_tournament](08_portfolio_tournament.md) | **blueprint only** | 06 결과 |

**07·08의 contract 를 지금 확정하지 않는 이유**: 두 spec의 파라미터(테마 가중치, 집중도, state 전이 임계값)는 06의 실측 분포 결과에 의존한다. 결과를 보기 전에 숫자를 계약으로 박으면 INV-11("가중치는 hypothesis 일 뿐")과 AGENTS.md의 *Invariant Logic Over Magic Numbers* 를 정면으로 위반한다. 06 완료 시점에 `/spec` 을 재실행하여 확정한다.

**pre-impl 게이트 순서**: `lean_check --pre-impl` 은 wiring caller file 의 실존을 요구한다. greenfield 이므로 spec 01만 지금 통과하고, 02~06은 각각 선행 spec 구현 직후에 통과한다. 이는 설계 결함이 아니라 순차 빌드의 정상 상태다.
