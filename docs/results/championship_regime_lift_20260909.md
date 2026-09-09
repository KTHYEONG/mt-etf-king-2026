# Championship Regime Lift v2 — Sleeve Diagnostic (20260909, production 오라클로 재계산)

**이전 산출물 대체:** `championship_regime_lift_20260908.json`은 모델조건부 오라클 결함으로 폐기(아래 정정문 참조), 이 문서가 정본입니다.

## 정정 사유

20260908 산출물은 executable_oracle_p50/capture_50 계산에 P27 자신의 스코어 후보(모델조건부)를 오라클로 오인해 기록함(INACTIVE 오라클=0.0, LOTTERY_ON capture=1.0으로 나타난 원인). 본 v2는 production src.research.executable_oracle + src.tournament.championship_regime.sleeve_conditional_table 을 사용한 진짜 시장전체(배치가능 유니버스) 오라클로 재계산함.

**P27 run:** `20260908T112031Z_sticky.mom60_raw_20180102_20260827_0300_0500_0010`

## 슬리브별 진짜 시장오라클 vs P27 (production sleeve_conditional_table)

| Sleeve | n_windows | P27 P>50 | P27 P>30 | P27 Ruin<-25 | 시장오라클 P>50 | Capture@50 |
|---|---:|---:|---:|---:|---:|---:|
| CRASH_REBOUND | 17 | 0.00% | 0.00% | 0.00% | 29.41% | 0.00% |
| LOTTERY_ON | 544 | 19.12% | 25.74% | 5.33% | 34.19% | 50.54% |
| INACTIVE | 1465 | 0.00% | 2.87% | 1.77% | 8.26% | 0.00% |
| UNCERTAIN | 60 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

전체 P>50: **104 / 2086 = 4.99%**

## P>50 by year

| Year | Windows | P>50 hits | Rate |
|---|---:|---:|---:|
| 2018 | 244 | 0 | 0.00% |
| 2019 | 246 | 0 | 0.00% |
| 2020 | 248 | 4 | 1.61% |
| 2021 | 248 | 0 | 0.00% |
| 2022 | 246 | 0 | 0.00% |
| 2023 | 245 | 0 | 0.00% |
| 2024 | 244 | 0 | 0.00% |
| 2025 | 242 | 49 | 20.25% |
| 2026 | 123 | 51 | 41.46% |
