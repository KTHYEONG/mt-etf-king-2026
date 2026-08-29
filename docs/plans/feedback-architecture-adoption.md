# Feedback 반영 계획 — Architecture 차이와 작업 순서

> **근거 문서**: [docs/architecture/](../architecture/) (01, 07, 08, 10), [docs/architecture/mt-data-report.md](../architecture/mt-data-report.md), [docs/results/06_research_harness_krx_baseline.md](../results/06_research_harness_krx_baseline.md)  
> **작성 기준일**: 2026-08-28  
> **대상 독자**: 구현·연구 방향을 한 번에 맞추기 위한 계획서 (상세 설계는 `docs/architecture/`, 실행 계약은 `docs/specs/`)

---

## 1. 한 줄 요약

외부 피드백 검토(2026-08-28)는 **“무엇을 검증해야 하는가”** 를 맞게 정의했다. 내용은 본 문서와 `docs/architecture/`에 반영되었다. 기존 `docs/architecture/`는 **데이터·PIT·baseline harness** 까지는 잘 맞지만, **대회 목적함수(giveback)·regime 연결·문서-코드 정합** 이 빠져 있고, **07 leadership을 너무 일찍·너무 넓게** 열려 있다.

**지금 할 일**은 feedback 전체를 코드에 넣는 것이 아니라, **preflight로 측정기와 문서를 고친 뒤**, 06 실측을 통과한 가설만 단계적으로 구현하는 것이다.

---

## 2. Feedback이 말하는 프로젝트 (수용)

### 2.1 목적

| 일반 퀀트 | 이 프로젝트 (feedback + architecture 공통) |
| --- | --- |
| CAGR, Sharpe, MDD | **Terminal return**, right-tail, **peak-to-final giveback**, rank preservation |
| 장기 안정 수익 | **36거래일** 모의투자 대회에서 상위권·우승 확률 |

Horizon: feedback은 “약 40거래일”이라 표현하지만, **캘린더 기준 2026 대회 = 36 sessions** (`TradingCalendar`). 구현은 36이 진실이다.

### 2.2 Core Hypothesis (한 문장)

> 8주 안의 시장·섹터 leadership을 탐지하고, 적합한 ETF vehicle로 집중한 뒤, leadership 약화·regime transition을 감지해 수익을 보존하고 필요 시 재진입하는 전략이 tournament에서 우월한가?

이 가설 자체는 **수용**한다. 다만 **아직 코드로 증명되지 않았으므로** 바로 production 전략으로 올리지 않는다.

### 2.3 계층 분리 (feedback §3, §34)

feedback이 그린 파이프라인과 기존 architecture의 **의도는 동일**하다.

```text
Point-in-Time Data
  → Market / Regime (+ transition risk)
  → Leadership Lifecycle
  → Cross-sectional Ranking
  → Vehicle Selection          ← alpha와 분리
  → Concentrated Portfolio     ← signal과 분리
  → Exit / Watch / Re-entry
  → Tournament Risk Budget     ← 대회 전용 overlay
  → 36D Terminal Performance Evaluation
```

기존 `docs/architecture/02-system-layers.md` 의 L0~L8 구조와 **충돌하지 않는다**. 차이는 **어느 블록이 이미 구현됐고, 어느 블록이 문서만 있는지** 이다.

### 2.4 과거 대회 데이터 역할 (feedback §2, §20)

| 데이터 | 역할 | 수용 규칙 |
| --- | --- | --- |
| **제1회 (2024, ~4개월)** | Hypothesis generator | 아이디어·feature 후보만. **파라미터 calibration·Core 채택 근거 금지** |
| **제2회 (2025, 8주)** | Tournament case study | replay·진단. **단독 최적화·기각 게이트 금지** |
| **다년 rolling 36D** | 실제 검증 | 전략 채택의 **유일한 정량 근거** |

`mt-data-report.md`의 관찰(leadership, leverage vehicle, rotation, giveback, re-entry)은 **검증할 질문 목록**으로 수용한다. 우승자 매매 **복제**는 하지 않는다.

---

## 3. 기존 Architecture와 무엇이 같은가

아래는 **feedback과 architecture 모두 동의**하며, **이미 spec 01~06으로 구현된 부분**이다.

| 영역 | 상태 | 근거 |
| --- | --- | --- |
| KRX bronze → silver → gold | ✅ 구현 | ingest, normalize, features CLI |
| PIT universe, sponsor deployment | ✅ 구현 | `universe/provider.py`, `configs/sponsor_brands.yaml` |
| Feature groups (mom, trend, vol, flow, breadth, regime) | ✅ 구현 | `src/features/*` |
| Next-open 체결, 비용·유동성 grid | ✅ 구현 | `backtest/`, `tournament/harness.py` |
| B0~B5 baseline + rolling 36D 분포 | ✅ 구현 + 실측 | `docs/results/06_research_harness_krx_baseline.md` |
| Alpha ≠ Portfolio (Protocol) | ✅ 구현 | `AlphaModel.score` → sizing은 별도 |
| LeverageFamily, 실제 ETF 가격 | ✅ 설계·일부 구현 | `universe/families.py`, INV-14 |
| Deployment / Structural 분리 | ✅ 설계 | harness `deployment` mode |
| ML은 gated 후보 | ✅ 문서 | `architecture/12-ml-layer.md` |

**결론**: “처음부터 다시 짠다”가 아니다. **데이터·harness·baseline spine은 유지**한다.

---

## 4. 기존 Architecture와 무엇이 다른가 (Gap)

### 4.1 목적함수·메트릭

| 항목 | 기존 architecture | feedback 요구 | 현재 코드 | **바뀌어야 할 것** |
| --- | --- | --- | --- | --- |
| 핵심 지표 | P(R>θ), CVaR, MDD, RTS | + **giveback**, turnover, transition loss | giveback **없음** | `peak_to_final_giveback` + 분포 요약 |
| 2025 replay | G-4: 실패 시 **기각** 경향 | case study, **과적합 방지** | synthetic 테스트만 | G-4 → **경고/진단**으로 완화 |
| 평가 창 | 36D (명시) | “40D” 표현 혼용 | 36D ✅ | 문서만 통일 |

### 4.2 Alpha / Leadership

| 항목 | 기존 architecture/07 | feedback | 06 실측 | **바뀌어야 할 것** |
| --- | --- | --- | --- | --- |
| B0~B5 정의 | Top-5, risk parity, creation flow 등 | M0~M9 계단식 비교 | **코드**: Top1/Top3 mom, theme, MA필터 | **문서를 코드에 맞춤** |
| Sector leadership | cluster score + 가중치 | lifecycle 6~7 상태 | B4(theme 평균) **실패** | 07은 **대표 ETF + rs/accel/breadth + dedup**만 |
| Feature 폭 | momentum, breadth, flow, breakout… | P0/P1 구분 | B3(MA) 실패, flow 데이터 V5 27% | flow/breakout **기본 제외** |
| 상태 기계 | CASH/LONG (architecture/07) | DISCOVERY→…→RECOVERY | 미구현 | 07: **6-state** / 08: **포지션** state |

### 4.3 Regime

| 항목 | 기존 | 현재 코드 | **바뀌어야 할 것** |
| --- | --- | --- | --- |
| Regime classifier | `features/regime.py` 존재 | `BacktestEngine`이 **항상 `regime=None`** | 엔진에 **일별 regime 주입** |
| B5 | Regime-gated theme | B5 ≡ B4 (실측) | preflight 후 **재측정** |
| Transition risk | feedback §10 강조 | 미구현 | **07 이후** (regime 연결 후) |

### 4.4 Vehicle / Portfolio / Tournament

| 항목 | architecture 문서 | feedback | 코드 | **시점** |
| --- | --- | --- | --- | --- |
| ExposureSelector | §5.2 상세 | vehicle layer 분리 | **없음** | **08** |
| Confidence sizing | 있음 | conf 기반 집중 | TOP1/EQUAL_K만 | **08** |
| Re-entry / WATCH | 연구 질문 #10 | P1, 우승자 강조 | 미구현 | **08** |
| Tournament aggression | policy.py | giveback·rank·days | 미구현, **기본 off** | **08** |
| Daily decision report | §1.7 형식 | 매일 출력 스펙 | CLI `decide` 미완 | **08 이후** |

### 4.5 데이터·인프라 (장기)

| feedback 항목 | architecture | 지금 | 판단 |
| --- | --- | --- | --- |
| Cross-market (US, FX) | P1 언급 | 미구현 | **수용하되 07 이후** |
| Competition history DB | §27 스키마 | 마크다운만 | 연구용, **07 블로커 아님** |
| PDF holdings breadth | DEC-B 불가 | cluster breadth로 대체 | **이미 architecture 결정과 일치** |
| 2018 backfill | roadmap 권장 | 2024~만 실측 | **병렬 운영 과제**, n_eff 확대용 |

### 4.6 문서·계약 정합

| 문제 | 영향 |
| --- | --- |
| INV 번호가 `00_architecture` vs `architecture/10` 에서 **다른 의미** | implement 시 잘못된 불변식 참조 |
| `architecture/11` “src 스켈레톤, CLI 미구현” | **현재 상태와 불일치** |
| spec `06_research_harness*.md` 가 git에서 제거됨 | ADR·results로 대체됨 — roadmap에 반영 필요 |

---

## 5. 수용 / 보류 / 기각 매트릭스

### 5.1 수용 — 지금 방향에 포함

- Terminal return·right-tail·**giveback** 중심 평가
- Leadership + regime + **vehicle 분리** + concentrated portfolio + re-entry + tournament overlay **로드맵**
- 제1회=가설, 제2회=사례, rolling 36D=채택 근거
- Baseline-first (B0~B5), 실패 시 빠른 기각
- `index_key` L1 dedup (상관계수 클러스터링 대신)
- Cluster-level breadth (PDF 대체)
- 레버리지는 **실제 ETF 가격**, alpha와 분리
- 후원사 deployment universe

### 5.2 보류 — 원칙만 수용, 구현은 뒤

| 항목 | 보류 이유 | 예상 단계 |
| --- | --- | --- |
| Leadership 6-state 본체 | regime·giveback 미연결 | Phase 3 (`spec 07`) |
| Vehicle / ExposureSelector | alpha 검증 전 | Phase 4 (`spec 08`) |
| Tournament policy / MC | overlay는 마지막 | Phase 4 |
| Cross-market features | KRX spine 우선 | Phase 5+ |
| Competition history ingest | 연구 보조 | Phase 5+ |
| LightGBM ranker | rule baseline 미확정 | Phase 5+, **기각이 기본** |
| 2018 backfill | 운영·표본 확대 | Phase 2 병렬 |

### 5.3 기각 — Core에 넣지 않음

| feedback/관찰 | 이유 |
| --- | --- |
| 장기 contrarian / 제1회 우승 전략 복제 | 8주와 구조 불일치 |
| Event/catalyst **독립 alpha** | 가격 모델 보조로만 |
| Strategy zoo (reversal, NLP, intraday, RL) | P3 또는 범위 외 |
| Theme **평균** momentum (B4 방식) | 06에서 이미 열위 |
| MA20 단독 필터 (B3 방식) | 06에서 tail·median 열위 |
| 2025 replay 실패 → 전략 폐기 | 단일 사례 과적합 |
| flow 성분 기본 포함 | V5 회계 항등식 ~27% |

---

## 6. 목표 아키텍처 vs 현재 위치

```text
[완료] L0 core ─ L1 data ─ L2 universe ─ L3 features ─ L6 backtest ─ L7 harness
         │
[지금] preflight ─ regime 주입, giveback, 문서 정렬
         │
[다음] L4 leadership (MVP) ─ rs/accel/breadth, dedup, 6-state  → A-1 게이트
         │
[그다음] L5 portfolio + L7 overlay ─ sizing, re-entry, vehicle, aggression
         │
[선택] L4 ml, cross-market, competition DB
```

feedback의 전체 그림(§3 다이어그램)은 **목표 상태**이고, **현재는 “harness까지 완료, alpha 고도화 직전”** 이다.

---

## 7. 작업 순서 (실행 체크리스트)

### Phase 0 — 이미 완료 (spec 01~06)

- [x] Core, KRX ingest, silver, universe, features
- [x] Backtest engine, B0~B5, rolling 36D, harness grid
- [x] KRX 2024~2026 실측 baseline 표 (`docs/results/06_research_harness_krx_baseline.md`)

### Phase 1 — Preflight ✅ 완료

**ADR**: `ADR_20260828_07_PREFLIGHT`

| # | 작업 | 산출 | 완료 기준 |
| --- | --- | --- | --- |
| 1.1 | `build_regime_series` + `BacktestEngine(regimes=...)` | regime이 B5에 전달 | B5 ≠ B4 (STRONG_RISK_OFF에서 B5 현금) |
| 1.2 | `peak_to_final_giveback` + `ReturnDistribution` 확장 | giveback median/q90 | rolling 결과에 giveback 열 |
| 1.3 | CLI backtest/replay regime·giveback 로그 | EVAL 로그 | backtest 로그에 giveback 출력 |
| 1.4 | architecture/spec 문서 정렬 | 10개 파일 수술 | B0~B5, INV, G-4, roadmap 현행화 |

**이 단계에서 하지 않는 것**: `leadership.py`, vehicle, ML, competition DB

### Phase 2 — Harness 보강 (preflight 직후, 병렬 가능)

| # | 작업 | 목적 |
| --- | --- | --- |
| 2.1 | B5 재측정 (regime on) | §5.4 무효 비교 해소 |
| 2.2 | 2018~ ingest backfill | n_eff 17 → 목표 ~40+ |
| 2.3 | 2025 KRX gold replay 35일 로그 | case study (기각 아님) |
| 2.4 | 비용×참여율 grid 결과표 | G-3 robustness |

산출: `docs/results/07_preflight_harness.md` (신규 결과 문서, 선택)

### Phase 3 — Leadership (spec 07 확정 후)

**선행**: Phase 1 PASS + Phase 2.1  
**Command**: `/spec 07_leadership_engine` → `/implement`

| 구현 범위 | 게이트 |
| --- | --- |
| L1 `index_key` dedup, 대표 ETF 선정 | A-2 worst-5% |
| Theme panel = **대표 ETF 가격** (평균 아님) | B4 교훈 |
| SectorScore: **rs + accel + cluster_breadth** (ablation) | A-1: P(R>30%) **> 0.091** (B2) |
| 6-state hysteresis (OVERHEATED 등) | A-3 turnover |

**실패 시**: leadership 폐기 → **B1/B2로 대회**, Phase 4는 sizing만 축소 진행

### Phase 4 — Portfolio & Tournament (spec 08)

**선행**: Phase 3 A-1 결과 (또는 baseline 확정)

| 구현 범위 | 게이트 |
| --- | --- |
| Confidence sizing (B1↔B2 보간) | B-1 |
| Position state: HOLD/TRIM/EXIT/WATCH/RE_ENTER | B-2 path-dependent |
| ExposureSelector (leverage family) | ARCH-6, 유동성 재검사 |
| Aggression overlay (**기본 off**) | B-4 |
| `mt-etf decide` + 근거 리포트 | B-5 replay |

### Phase 5 — 선택 확장

- Cross-market adapter (P1)
- Competition history parquet
- ML ranker (G6~G8, **기각이 기본**)
- Event overlay (confidence feature만)

---

## 8. Spec · Architecture · Plan 관계

```text
docs/architecture/*       ← “계층·불변식·설계 원칙” (피드백 반영 완료)
docs/plans/* (본 문서)    ← “무엇을 언제 바꾸는가” (실행 순서)
docs/specs/*              ← “무엇을 코드에 박는가” (contract → implement)
docs/results/*            ← “실측 숫자” (spec 07/08 입력)
```

| 단계 | Plan | Spec | Architecture 갱신 |
| --- | --- | --- | --- |
| Preflight ✅ | 본 문서 §7 Phase 1 | `ADR_20260828_07_PREFLIGHT` | 01, 07, 08, 10, 11, 00 |
| Leadership | §7 Phase 3 | `07_leadership_engine` | 07 |
| Portfolio | §7 Phase 4 | `08_portfolio_tournament` | 07, 11 |

---

## 9. 의사결정 요약 (자주 묻는 것)

**Q. feedback을 전부 구현하나?**  
A. 아니다. **검증 원칙과 계층 순서**는 수용하고, 06에서 실패한 것(B4/B3)과 데이터가 부족한 것(flow, cross-market)은 빼거나 뒤로 미룬다.

**Q. 07 leadership은 확정인가?**  
A. 아니다. **가설**이다. A-1(P(R>30%) > B2) 통과 못 하면 baseline으로 나간다.

**Q. 기존 architecture 문서를 버리나?**  
A. 버리지 않는다. **드리프트 난 부분만** preflight/07 spec implement 시 맞춘다. 캐논 INV는 `architecture/10-invariants-and-gates.md`.

**Q. 지금 당장 뭘 실행하나?**  
A. `/implement docs/specs/09_vehicle_exposure_contract.json` (B-2·08 완료 후 Phase 4 잔여: vehicle/gross/aggression wiring)

---

## 10. 참고 링크

| 문서 | 용도 |
| --- | --- |
| [mt-data-report.md](../architecture/mt-data-report.md) | 제1·2회 사실 정리 |
| [06_research_harness_krx_baseline.md](../results/06_research_harness_krx_baseline.md) | B0~B5 실측 앵커 |
| [ADR_20260828_07_PREFLIGHT](../decisions/task_index.json) | preflight 완료 기록 |
| [07_leadership_engine.md](../specs/07_leadership_engine.md) | leadership blueprint (contract 유예) |
| [08_portfolio_tournament.md](../specs/08_portfolio_tournament.md) | portfolio blueprint (contract 유예) |
