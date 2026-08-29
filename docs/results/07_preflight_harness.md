# 07 Preflight Harness — B5 Regime-On Remeasurement

**작성일**: 2026-08-28  
**선행**: ADR_20260828_07_PREFLIGHT  
**목적**: regime 주입 후 B4 vs B5 유효 비교 + giveback 열 추가

## 1. Regime 시리즈

- index proxy: ETF `underlying_index_close` (코스피 200, 코스닥 150)
- sessions: 648, regime mapped: 648
- STRONG_RISK_OFF days: **146** (22.5%)

| Regime | Days |
| --- | ---: |
| NEUTRAL | 71 |
| RISK_OFF | 80 |
| RISK_ON | 87 |
| STRONG_RISK_OFF | 146 |
| STRONG_RISK_ON | 264 |

## 2. B4 vs B5 (regime ON)

| 지표 | B4 | B5 | Δ (B5−B4) |
| --- | ---: | ---: | ---: |
| median | -0.0041 | 0.0018 | +0.0059 |
| P(R>30%) | 0.064 | 0.064 | +0.000 |
| q90 | 0.1611 | 0.1714 | +0.0103 |
| giveback_median | 0.0358 | 0.0207 | -0.0151 |
| giveback_q90 | 0.1011 | 0.1410 | +0.0398 |

**판정**: B5 ≠ B4 → **PASS**

## 3. 전체 Baseline (regime ON, giveback 포함)

| Model | median | q90 | CVaR(5%) | RTS | P>30% | giveback_med | giveback_q90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.0439 | 0.3098 | -0.1909 | 0.3655 | 0.109 | 0.0257 | 0.1463 |
| B1 | -0.0090 | 0.3883 | -0.3704 | 0.6712 | 0.121 | 0.0320 | 0.1375 |
| B2 | 0.0166 | 0.2859 | -0.2416 | 0.3928 | 0.091 | 0.0311 | 0.1377 |
| B3 | -0.0122 | 0.2360 | -0.3704 | 0.4474 | 0.073 | 0.0332 | 0.3330 |
| B4 | -0.0041 | 0.1611 | -0.2476 | 0.5216 | 0.064 | 0.0358 | 0.1011 |
| B5 | 0.0018 | 0.1714 | -0.1609 | 0.5032 | 0.064 | 0.0207 | 0.1410 |

## 4. 07 Spec 시사점

- A-1 게이트 기준선: B2 P(R>30%) = **0.091**
- B4(theme 평균 mom) P(R>30%) = 0.064 — 07은 대표 ETF + rs/accel/breadth로 차별화 필요
- B1 vs B2: conf 낮을 때 분산(B2), 높을 때 집중(B1) — 08 sizing 근거
