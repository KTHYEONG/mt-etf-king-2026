# 06 Research Harness — KRX 실데이터 Baseline 분포 결과

**작성일**: 2026-08-28  
**목적**: spec 07 (`leadership_engine`) · spec 08 (`portfolio_tournament`) `/spec` 재실행을 위한 **실측 입력 데이터**  
**선행**: [06_research_harness](../specs/06_research_harness.md) W3 게이트 · harness 구현 완료  
**후속**: `/spec 07_leadership_engine`, `/spec 08_portfolio_tournament` (본 문서 기반 contract 확정)

---

## 1. Executive Summary

KRX Open API → bronze → silver → gold 파이프라인을 **2024-01-02 ~ 2026-08-27** 구간에서 실행한 뒤, 동일 프로토콜로 baseline B0~B5의 **36세션 롤링 수익률 분포**를 산출했다.

| 핵심 관측 | 내용 |
| --- | --- |
| **최고 median** | B0 (KOSPI200 B&H) **+4.39%** |
| **최고 P(R>30%)** | B1 (Top-1 mom) **12.1%** — 단, median 음수 |
| **집중도 (B1 vs B2)** | Top-3 분산(B2)이 median·CVaR 우위, Top-1(B1)이 우측 꼬리(q90, P>30%) 우위 |
| **추세 필터 (B1 vs B3)** | MA20 필터(B3)가 median·P>30% 모두 열위 → 본 구간에서 필터는 **손실** |
| **테마 (B2 vs B4)** | theme 모멘텀(B4)이 개별 Top-3 mom(B2) 대비 전 지표 열위 |
| **레짐 (B4 vs B5)** | **완전 동일** — engine이 `DecisionContext.regime` 미주입 |
| **n_effective** | 17 (목표 ~58은 2018 backfill 후) — 해석 시 bootstrap CI 폭 주의 |

**07/08 진입 판정**: harness 자체는 실데이터 E2E 통과. 다만 (1) 레짐 미연결, (2) 단일 비용·참여율 점, (3) 짧은 표본으로 **07 A-게이트·08 B-게이트 수치는 아직 확정 불가**. 본 표를 `/spec` 입력으로 사용하되, 아래 §7 갭을 contract에 반영할 것.

---

## 2. 데이터 파이프라인 & 품질

### 2.1 실행 이력

| 단계 | 명령 | 결과 |
| --- | --- | --- |
| Ingest | `mt-etf ingest --start 2024-01-02 --end 2026-08-27` | bronze **648** sessions |
| Normalize | `mt-etf normalize` | **630,708** rows, **646** sessions → `data/normalized/etf_daily.parquet` |
| Features | `mt-etf features --start 2024-01-02 --end 2026-08-27` | **632,990** rows, **1,286** tickers → `data/features/etf_features.parquet` |

Gold 패널 확인 (2026-08-28):

```
rows=632,990  tickers=1,286  date_range=2024-01-02 .. 2026-08-27
```

### 2.2 품질 경고

| 코드 | 내용 | 영향 |
| --- | --- | --- |
| **V5** | `net_assets` 회계 항등식 일치율 ~**27%** | AUM·flow 기반 feature(07 `flow` 성분) 신뢰도 제한 |
| **V6** | 누락 세션 2일: **2026-06-03**, **2026-07-17** | 해당일 rolling 창 경계 왜곡 가능 (미미) |
| **표본 길이** | ~2.7년 → `n_windows=613`, `n_effective=17` | 2018 backfill 전까지 tail 추정 불안정 |

---

## 3. 평가 프로토콜 (B0~B5 공통)

모든 baseline은 **동일 설정**으로 `TournamentSimulator.run_rolling(path_dependent=False)` 실행.

| 항목 | 값 |
| --- | --- |
| **기간** | 2024-01-02 .. 2026-08-27 |
| **Horizon** | **36** sessions (`2026-09-21` .. `2026-11-13` 대회 규칙과 동일) |
| **자본** | 10억 KRW |
| **Universe** | `deployment` mode (`configs/universe.yaml`) |
| **Warmup** | 80 sessions |
| **Participation φ** | **1%** (`max_order_to_adv=0.01`) |
| **비용** | commission **3 bps** + slippage **5 bps** (`CostModel.charge`, spread 0) |
| **체결** | `NextOpenExecution` — signal `close(t)` → fill `open(t+1)` (INV-10) |
| **유동성 cap** | 체결 시점 `ADV×φ` 상한 (`cap_target_weights_by_adv`) |
| **비용 시점** | 체결일 equity 기준 차감 (signal일 아님) |

### 3.1 Baseline 정의

| ID | 모델 | Sizing |
| --- | --- | --- |
| **B0** | KODEX 200 (`069500`) Buy & Hold | 단일 종목 100% |
| **B1** | Top-1 `mom_20` | Top1 |
| **B2** | Top-3 `mom_20` (동일 스코어) | Equal-K, k=3 |
| **B3** | `mom_20` + `close > MA20` | Top1 |
| **B4** | theme별 `mom_20` 평균 → 최고 theme 내 종목 스코어 | Top1 |
| **B5** | B4 + `STRONG_RISK_OFF` 시 현금 | Top1 |

### 3.2 분포 메트릭

- **Quantiles**: q05, q25, q50 (median), q75, q90, q95, q99  
- **Tail**: CVaR(5%), RTS (right-tail score, tail weights `{0.75:0.2, 0.90:0.3, 0.95:0.3, 0.99:0.2}`)  
- **Exceedance**: P(R>θ), θ ∈ {10%, 20%, 30%, 40%, 50%}  
- **n_effective**: `floor(n_windows / horizon)` = **17**  
- **MDD (rolling 중앙)**: 각 36일 창 내 max drawdown의 중앙값

---

## 4. 전체 결과표

### 4.1 요약 (primary)

| Model | n_windows | n_eff | median | q90 | CVaR(5%) | RTS | P(R>10%) | P(R>30%) | mdd_med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **B0** | 613 | 17 | **0.0439** | 0.3098 | -0.1909 | 0.3655 | **0.328** | 0.109 | -0.0654 |
| **B1** | 613 | 17 | -0.0119 | **0.3884** | -0.3862 | **0.6683** | 0.212 | **0.121** | -0.0576 |
| **B2** | 613 | 17 | 0.0130 | 0.2840 | -0.2528 | 0.3857 | 0.266 | 0.091 | -0.0661 |
| **B3** | 613 | 17 | -0.0150 | 0.2381 | -0.3862 | 0.4459 | 0.168 | 0.073 | -0.0576 |
| **B4** | 613 | 17 | -0.0050 | 0.1573 | -0.2593 | 0.5213 | 0.171 | 0.064 | -0.0609 |
| **B5** | 613 | 17 | -0.0050 | 0.1573 | -0.2593 | 0.5213 | 0.171 | 0.064 | -0.0609 |

### 4.2 분위수 전체

| Model | q05 | q25 | q50 | q75 | q90 | q95 | q99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | -0.1225 | -0.0155 | 0.0439 | 0.1637 | 0.3098 | 0.4021 | 0.5962 |
| B1 | -0.2909 | -0.0375 | -0.0119 | 0.0348 | 0.3884 | 0.8918 | 1.3864 |
| B2 | -0.2104 | -0.0357 | 0.0130 | 0.1098 | 0.2840 | 0.4207 | 0.7618 |
| B3 | -0.2909 | -0.0520 | -0.0150 | 0.0109 | 0.2381 | 0.5120 | 1.0934 |
| B4 | -0.2055 | -0.0408 | -0.0050 | 0.0594 | 0.1573 | 0.7927 | 1.1218 |
| B5 | -0.2055 | -0.0408 | -0.0050 | 0.0594 | 0.1573 | 0.7927 | 1.1218 |

### 4.3 Exceedance 곡선 P(R>θ)

| Model | P>10% | P>20% | P>30% | P>40% | P>50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.328 | 0.201 | 0.109 | 0.054 | 0.024 |
| B1 | 0.212 | 0.158 | 0.121 | 0.100 | 0.083 |
| B2 | 0.266 | 0.147 | 0.091 | 0.057 | 0.047 |
| B3 | 0.168 | 0.111 | 0.073 | 0.059 | 0.052 |
| B4 | 0.171 | 0.095 | 0.064 | 0.060 | 0.059 |
| B5 | 0.171 | 0.095 | 0.064 | 0.060 | 0.059 |

---

## 5. 연구 질문별 쌍 비교 (06 §1.3)

06 spec이 정의한 **순수 기여** 분리 쌍. 차이 = 후자 − 전자 (median·P>30% 기준).

### 5.1 B1 vs B2 — 집중도 (Top1 vs Top3 equal)

| 지표 | B1 (Top1) | B2 (Top3) | Δ (B2−B1) | 해석 |
| --- | ---: | ---: | ---: | --- |
| median | -0.0119 | **0.0130** | **+2.49pp** | 분산이 중앙 수익 개선 |
| P(R>30%) | **0.121** | 0.091 | -3.0pp | 집중이 tail lottery 유리 |
| q90 | **0.3884** | 0.2840 | -10.4pp | B1 극단 상승 꼬리 두꺼움 |
| CVaR(5%) | -0.3862 | **-0.2528** | +13.3pp | Top3가 하방 tail 완화 |
| RTS | **0.6683** | 0.3857 | -0.28 | tail score는 B1 우위 |

**→ 08 시사점**: 고정 Top1은 median 희생·tail 확률 증가 trade-off. `confidence_weights`는 **conf 낮을 때 B2 쪽(분산)**, conf 높을 때 B1 쪽으로 가야 본 데이터와 정합.

### 5.2 B1 vs B3 — 추세 필터 (MA20)

| 지표 | B1 | B3 | Δ (B3−B1) | 해석 |
| --- | ---: | ---: | ---: | --- |
| median | -0.0119 | -0.0150 | -0.31pp | 필터가 약간 악화 |
| P(R>30%) | **0.121** | 0.073 | **-4.8pp** | tail 기회 상당 부분 제거 |
| q90 | **0.3884** | 0.2381 | -15.0pp | 상승장 지속 구간에서 필터가 랭킹 희석 |

**→ 07 시사점**: 단순 MA20 게이트는 07 state machine의 `LEADING`/`BREAKDOWN` 전이 **대체재가 아님**. 상태 기계는 필터가 아니라 **히스테리시스·재진입 경로**에 초점.

### 5.3 B2 vs B4 — 섹터/테마 로테이션

| 지표 | B2 (개별 Top3) | B4 (theme mom) | Δ (B4−B2) | 해석 |
| --- | ---: | ---: | ---: | --- |
| median | **0.0130** | -0.0050 | -1.80pp | 현재 theme 집계가 열위 |
| P(R>30%) | **0.091** | 0.064 | -2.7pp | 테마 레벨 신호가 tail도 약화 |
| q90 | **0.2840** | 0.1573 | -12.7pp | theme 평균 mom이 노이즈·지연 가능 |

**→ 07 시사점**: B4의 단순 `theme` 평균 mom은 **실패 baseline**. 07은 (1) L1 `index_key` dedup, (2) 대표 ETF 실행 품질 선정, (3) multi-component `SectorScore` + ablation이 **필수 차별화**. A-1 게이트 기준선: **P(R>30%) ≥ 0.091 (B2)**.

### 5.4 B4 vs B5 — regime gate

| 지표 | B4 | B5 | Δ | 해석 |
| --- | ---: | ---: | ---: | --- |
| 전 지표 | 동일 | 동일 | 0 | **구현 갭**: engine이 regime 미주입 |

**→ 조치**: `BacktestEngine` 루프에서 `FeatureBuilder` regime snapshot을 `DecisionContext`에 연결 후 B5 재측정. 그 전까지 B4 vs B5 비교는 **무효**.

---

## 6. spec 07 / 08 `/spec` 입력 가이드

### 6.1 spec 07 — Leadership Engine

| 항목 | 본 결과 기반 권고 |
| --- | --- |
| **A-1 (vs B2)** | 신규 leadership 모델 P(R>30%) **> 0.091** 목표 |
| **A-2 (dedup)** | B1 q05 **-29.1%** vs B2 **-21.0%** — 중복 베팅 시 하방 악화 가능성 실측 근거 |
| **성분 ablation** | B3 실패 → `breakout`/MA 유사 단독 성분은 우선순위 낮춤; `rs`·`flow`·`breadth` 우선 검증 |
| **theme 정의** | B4 실패 → `ThemePanel`은 평균 수익률이 아닌 **대표 ETF 가격** (spec 07 §2.1) 고수 |
| **파라미터 유도** | q90~q99 스프레드(B1: 0.39~1.39)가 크므로 전이 임계는 고정 상수 금지, **분위수 기반** |
| **Contract 해제** | 본 문서 + regime 연결 후 B5 재측정 + (권장) 2018 backfill로 n_eff≥40 |

### 6.2 spec 08 — Portfolio & Tournament

| 항목 | 본 결과 기반 권고 |
| --- | --- |
| **B-1 (confidence sizing)** | 고정 Top1(B1) P>30%=0.121 vs Top3(B2)=0.091 — sizing은 **conf 연속 매핑**으로 두 극 사이 보간 |
| **B-1 worst-5%** | B2 CVaR -0.253 vs B1 -0.386 — 분산 정책이 하방 개선 시사 |
| **INV-08-2 (liquidity)** | φ=1% 고정 결과 — 08 stress는 harness `participation_grid` {1,2,5}% 재사용 |
| **B-2 (re-entry)** | path-dependent 평가 필수; 현재 rolling은 path-independent → 08 구현 시 `path_dependent=True` |
| **B-4 (aggression)** | overlay 기본 off 유지; B1 tail이 이미 높아 overlay 없이도 lottery 가능 |
| **2025 replay** | 단위 테스트 35세션 통과; **KRX gold replay 일별 로그**는 08 B-5 전 별도 실행 필요 |

### 6.3 수치 앵커 (contract 작성 시 인용)

```
# 07 acceptance reference (single-protocol, 2024-2026 KRX)
baseline_B2_P_R_gt_30pct: 0.091
baseline_B1_P_R_gt_30pct: 0.121
baseline_B0_median_36d:     0.0439
leadership_must_beat:       B2 on P(R>30%)  # gate A-1

# 08 sizing reference
top1_median:   -0.0119   top3_median: 0.0130
top1_cvar_05:  -0.3862   top3_cvar_05: -0.2528
```

---

## 7. 알려진 갭 & W3 게이트 체크리스트

### 7.1 구현/측정 갭

| # | 갭 | 심각도 | 07/08 블로커? |
| --- | --- | --- | --- |
| G1 | `DecisionContext.regime` 미주입 → B5=B4 | 높음 | 07 B5·08 regime gate 검증 전 **필수** |
| G2 | 평가 구간 2024~ (n_eff=17) | 중간 | 권장: 2018 ingest backfill |
| G3 | 본 보고서 단일 (comm, slip, φ) 점 | 중간 | harness 12×3 grid는 CLI 존재, 결과표 미첨부 |
| G4 | 슬리피지 `open×(1+slip)` 미적용 | 낮음 | contract상 `CostModel.charge` — 의도적 |
| G5 | KRX gold 2025 replay 35일 로그 | 중간 | 08 B-5 전 필요 |
| G6 | V5 net_assets 27% 일치 | 중간 | 07 `flow` 성분 신뢰도 |

### 7.2 W3 게이트 (06 §5)

| # | 기준 | 상태 |
| --- | --- | --- |
| 1 | B0~B5 동일 프로토콜 + `ReturnDistribution` | ✅ |
| 2 | P(R>θ) + n_effective 보고 | ✅ (본 문서) |
| 3 | 비용·유동성 그리드 축 | ⚠️ 구현됨, 본 보고서는 φ=1%·3+5bps 단일점 |
| 4 | 2025 replay 35세션 일별 로그 | ⚠️ synthetic 단위 테스트만, KRX replay 미실행 |

---

## 8. 재현 방법

```bash
# 데이터 (이미 존재 시 생략)
uv run mt-etf ingest --start 2024-01-02 --end 2026-08-27
uv run mt-etf normalize
uv run mt-etf features --start 2024-01-02 --end 2026-08-27

# 단일 baseline
uv run mt-etf backtest --model B1 --start 2024-01-02 --end 2026-08-27 --horizon-from-rules

# contract 검증
uv run pytest tests/unit/backtest tests/unit/tournament -q --tb=short
uv run python tools/agent_skills/lean_check.py --spec docs/specs/06_research_harness_contract.json
```

본 보고서 수치는 `TournamentSimulator` + `ReturnDistribution.summarise`로 산출 (프로토콜 §3과 동일).

---

## 9. 결론 — 07/08 `/spec` Go/No-Go

| 결정 | 조건 |
| --- | --- |
| **Go (conditional)** | harness + 실데이터 E2E 신뢰 가능 → **07/08 blueprint `/spec` 착수 가능** |
| **전제** | (1) regime 주입 후 B5 재측정, (2) acceptance 수치를 본 표 앵커로 명시, (3) n_eff 한계를 assumption에 기록 |
| **No-Go (폐기 시나리오)** | 07 구현 후 P(R>30%) < 0.091 지속 → spec 07 §3: baseline(B2/B1)으로 대회 참가 |

**다음 작업 순서 제안**:

1. Engine regime 주입 → B5 재실행 → §5.4·§7.1 G1 해소  
2. `/spec 07_leadership_engine` — A-1 기준선 `0.091`, B4 실패 근거 인용  
3. Leadership MVP + ablation → 분포 재측정  
4. `/spec 08_portfolio_tournament` — B-1·B-2는 B1/B2 쌍 실측 인용  
5. (병렬) 2018 backfill → n_eff 확대 후 acceptance 재확인
