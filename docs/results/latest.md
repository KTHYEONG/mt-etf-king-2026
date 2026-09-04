# Latest — KRX Rolling-36D Championship Results

**as_of**: 2026-09-04  
**scope**: 현재 코드베이스(`main`) — sell-first execution (TASK_41) + tail forensics (TASK_42) + P31 convex impulse (TASK_48) 반영  
**primary_run**: `20260903T034356Z_P27_20180102_20260827_0300_0500_0010`  
**research_run**: `20260904T033654Z_P31_20180102_20260827_0300_0500_0010`  
**champion**: `P27` (`sticky.mom60_raw`) · **anchor**: `P21` (`sticky.impulse_crash`)

---

## 1. Executive Summary

| 판정 | 내용 |
| --- | --- |
| **운영 채택 (decide/backtest)** | **P27** — `CHAMPION_STRATEGY`, sell-first rebaseline, **championship/adoption gate PASS** |
| **실행 무결성** | `gross_violation_count=0`, `effective_gross_max=1.91` (prior run 1,958 / 3.55) |
| **Tail forensics (P2)** | `primary_gap=timing` — entry+exit timing 손실이 selection보다 큼; per-window dominant는 selection 53.6% |
| **우측 꼬리 연구 1위 (미채택)** | **P26** — raw `P>50%=5.31%`, `championship_score=0.064` (gate FAIL) |
| **구조 연구 후보 (미채택)** | **P31** — beta 2x 제거·`q99=106.5%`·2025 oneshot +47.4% but `gross_viol=70`, gate FAIL |
| **역사 벤치마크** | 제2회 우승 +47.82% · P27 2025 oneshot +41.4% · P31 2025 oneshot +47.4% · oracle ceiling `P>50%≈20.1%` |

**핵심**: sell-first 실행으로 gross invariant가 복구되어 P27이 **채택 gate를 통과**했다. **P31**은 beta 2x를 제거하고 2025 oneshot +47.4%·q99 106.5%를 달성했으나 `gross_viol=70`·P>30% 열위로 **연구용**에 머문다. Forensics는 **집계 기준 timing gap**, **개별 윈도우 기준 selection gap**을 동시에 보여주며, P3는 timing/giveback controller + selection ensemble 병행이 타당하다.

---

## 2. 평가 프로토콜 (공통)

모든 수치는 아래 설정으로 `TournamentSimulator.run_rolling(path_dependent=True, fast)` 산출.

| 항목 | 값 |
| --- | --- |
| 기간 | 2018-01-02 .. 2026-08-27 |
| Horizon | 36 sessions |
| n_windows | 2,090 |
| n_effective | 58 |
| 자본 | 10억 KRW |
| Universe | `deployment` |
| Participation φ | 1% (`max_order_to_adv=0.01`) |
| 비용 | commission 3 bps + slippage 5 bps |
| 체결 | `NextOpenExecution` (causal open fill, TASK_37) |
| Exposure (P21–P31) | max_single=0.95, max_gross=1.90, min_cash=0.05 |
| 실행 순서 | sell-first (`constrain_target_weights_sell_first`, TASK_41) |

`championship_score` = `configs/gates.yaml` 의 `championship` 시나리오 가중 exceedance  
(가중치: P>30%×0.10 + P>40%×0.25 + P>50%×0.45 + P>60%×0.20).

---

## 3. Champion — P27 (sell-first rebaseline)

**run_id**: `20260903T034356Z_P27_20180102_20260827_0300_0500_0010`  
**path**: `data/results/20260903T034356Z_P27_20180102_20260827_0300_0500_0010/`  
**artifacts**: `summary.json`, `meta.json`, `windows.parquet` (2,090 rows), `daily.parquet` (2,125 rows), `trades.parquet` (1,332 rows), `tail_attribution_report.json`

### 3.1 Tail & 분포 지표

| 지표 | 값 |
| --- | ---: |
| P>30% | 8.71% |
| P>40% | 6.22% |
| P>50% | **4.26%** |
| P>60% | **3.92%** |
| championship_score | **0.0534** |
| median (q50) | +0.00% |
| q90 | +27.0% |
| q95 | +42.9% |
| q99 | +80.8% |
| CVaR(5%) | −30.1% |
| giveback_median | 3.40% |
| giveback_q90 | 17.6% |
| right_tail_score | 0.387 |
| objective_ruin (P<-25%) | **2.54%** |

### 3.2 Exceedance (raw)

| threshold | exceedance |
| --- | ---: |
| P>10% | 22.97% |
| P>20% | 13.78% |
| P>30% | 8.71% |
| P>40% | 6.22% |
| P>50% | 4.26% |

### 3.3 Quantiles

| quantile | terminal_return |
| --- | ---: |
| q05 | −16.65% |
| q25 | −4.49% |
| q50 | 0.00% |
| q75 | +7.69% |
| q90 | +27.0% |
| q95 | +42.9% |
| q99 | +80.8% |

### 3.4 실행·노출 진단

| 지표 | 값 | 비고 |
| --- | ---: | --- |
| gross_violation_count | **0** | prior `20260902T091905Z`: 1,958 |
| effective_gross_max | **1.91** | prior: 3.55 |
| carry_gross_drift_count | 19,430 | rolling session diagnostics 합산 (calendar-unique 아님) |
| delever_required_count | 19,430 | 동일 집계 방식 |

### 3.5 Field-relative (vs P21 incumbent, 동일 exposure)

| 지표 | 값 | prior run |
| --- | ---: | ---: |
| win_rate | **59.0%** | 56.8% |
| top2_rate | 100% | 100% |
| median_rank_percentile | 50% | 50% |

### 3.6 Annual oneshot (36D, 대회 시작일 anchor)

| year | start | return |
| ---: | --- | ---: |
| 2018 | 2018-09-21 | −11.3% |
| 2019 | 2019-09-23 | −4.1% |
| 2020 | 2020-09-21 | +4.0% |
| 2021 | 2021-09-23 | 0.0% |
| 2022 | 2022-09-21 | 0.0% |
| 2023 | 2023-09-21 | −2.1% |
| 2024 | 2024-09-23 | −0.4% |
| **2025** | **2025-09-22** | **+41.4%** |

제2회 우승 +47.82% 대비 −6.4pp (2025 anchor 동일).

### 3.7 Gate 상태

| gate | status | failures |
| --- | --- | --- |
| objective | **PASS** | — |
| championship | **PASS** | — |
| adoption | **PASS** | — |

### 3.8 Prior run 대비 (sell-first 전후)

| metric | prior `20260902T091905Z` | **current** | Δ |
| --- | ---: | ---: | ---: |
| gross_violation_count | 1,958 | **0** | −1,958 |
| effective_gross_max | 3.55 | **1.91** | −1.64 |
| championship_gate | FAIL | **PASS** | — |
| adoption_gate | FAIL | **PASS** | — |
| P>50% | 4.69% | 4.26% | −0.43pp |
| P>60% | (미저장) | 3.92% | — |
| q99 | 86.9% | 80.8% | −6.1pp |
| ruin | 4.21% | 2.54% | −1.67pp |
| field_win_rate | 56.8% | 59.0% | +2.2pp |

Tail 수치 하락은 gross inflation artifact 제거에 따른 **보수적 재측정**으로 해석.

---

## 4. Research — P31 (Convex Lottery Impulse, TASK_48)

**run_id**: `20260904T033654Z_P31_20180102_20260827_0300_0500_0010`  
**semantic_id**: `convex.lottery_impulse`  
**path**: `data/results/20260904T033654Z_P31_20180102_20260827_0300_0500_0010/`  
**artifacts**: `summary.json`, `meta.json`, `loyo_report.json`, `windows.parquet` (2,090 rows), `daily.parquet` (2,125 rows), `trades.parquet` (126 rows)  
**incumbent**: P27 (`20260903T034356Z_…`) · **판정**: 연구용 — **채택 불가** (`gross_viol=70`, adoption/championship FAIL)

**가설**: P27의 beta 2x(122630/233740) 집중이 tail ceiling을 제한; sector convex 2x + impulse gate + CASH default로 우측 꼬리를 회복할 수 있다.

### 4.1 Tail & 분포 지표 (vs P27)

| 지표 | P31 | P27 | Δ |
| --- | ---: | ---: | ---: |
| P>30% | 5.55% | 8.71% | −3.16pp |
| P>40% | 5.22% | 6.22% | −1.00pp |
| P>50% | **4.16%** | **4.26%** | −0.10pp |
| P>60% | 3.35% | 3.92% | −0.57pp |
| championship_score | 0.0440 | 0.0534 | −0.0094 |
| median (q50) | +0.00% | +0.00% | — |
| q90 | +0.24% | +27.0% | −26.8pp |
| q95 | +44.2% | +42.9% | +1.3pp |
| q99 | **+106.5%** | +80.8% | **+25.7pp** |
| CVaR(5%) | −13.4% | −30.1% | +16.7pp |
| giveback_median | 0.00% | 3.40% | — |
| giveback_q90 | 4.28% | 17.6% | −13.3pp |
| right_tail_score | 0.346 | 0.387 | −0.041 |
| objective_ruin (P<-25%) | **0.86%** | 2.54% | −1.68pp |

### 4.2 실행·노출 진단

| 지표 | P31 | P27 | 비고 |
| --- | ---: | ---: | --- |
| gross_violation_count | **70** | **0** | CASH↔TARGET 전환 — **채택 blocker** |
| effective_gross_max | 1.90 | 1.91 | limit 내 |
| invested_weight_mean | **3.4%** | — | 과도한 cashification |
| trade_rows | **126** | 1,332 | beta 2x BUY **0건** |
| field win_rate | 33.6% | 59.0% | vs P21 incumbent |

**체결 ticker** (전량 sector convex 2x): `488080`(53), `494310`(50), `243880`(14), `462330`(9). `122630`/`233740` 없음.

### 4.3 Annual oneshot (36D)

| year | start | P31 | P27 |
| ---: | --- | ---: | ---: |
| 2018 | 2018-09-21 | 0.0% | −11.3% |
| 2019 | 2019-09-23 | 0.0% | −4.1% |
| 2020 | 2020-09-21 | 0.0% | +4.0% |
| 2021 | 2021-09-23 | 0.0% | 0.0% |
| 2022 | 2022-09-21 | 0.0% | 0.0% |
| 2023 | 2023-09-21 | −1.7% | −2.1% |
| 2024 | 2024-09-23 | −1.7% | −0.4% |
| **2025** | **2025-09-22** | **+47.4%** | **+41.4%** |

제2회 우승 +47.82% 대비 P31 −0.4pp · P27 −6.4pp (2025 anchor 동일).

### 4.4 Gate & LOYO

| gate | status | failures |
| --- | --- | --- |
| objective | **FAIL** | `G1_TAIL` (P>30% 5.55% < 8%) |
| championship | **FAIL** | `PRIMARY_CI_VS_INCUMBENT`, scenario vs incumbent |
| adoption | **FAIL** | 동일 |
| LOYO | **FAIL** | `P50_NOT_IMPROVED`, `CONCENTRATION=1.0` (P>50% 전부 2025–26) |

LOYO non-inferior: 6/9 years. 2018–22 impulse gate 미충족 → 거의 전기간 flat.

### 4.5 해석

| signal | implication |
| --- | --- |
| beta 제거 성공 | sector convex만 체결; 구조적 가설 부분 검증 |
| q99↑ ruin↓ | cash-default + crash-cash가 tail shape 개선 |
| P>30/P>40↓ | gate 과엄격 → 2018–24 투자일수 극소 |
| gross_viol=70 | sell-first CASH 전환 미준수 — 수정 전 채택 불가 |
| P>50 미개선 | concentration 100% 2025–26 — LOYO FAIL 지속 |

**다음**: gross invariant 수정 → gate calibration (impulse threshold) → LOYO 재평가.

---

## 5. Tail Forensics Attribution (TASK_42)

**command**: `uv run mt-etf forensics --run-id 20260903T034356Z_P27_20180102_20260827_0300_0500_0010`  
**report**: `data/results/…/tail_attribution_report.json`  
**runtime**: ~3–4 min

### 5.1 Window set

| 항목 | 값 |
| --- | ---: |
| n_windows_total | 2,090 |
| n_analyzed | 265 |
| selection rule | top_q=0.95 (upper 5%) ∪ near-miss [20%, 50%) |
| oracle | ex-post +2x close-to-close BH per family (research proxy) |

### 5.2 Aggregate losses

| loss bucket | mean | share of total* |
| --- | ---: | ---: |
| selection | 0.196 | 27.0% |
| entry_timing | **0.216** | 29.8% |
| exit_timing | 0.120 | 16.5% |
| giveback | 0.124 | 17.1% |
| timing (entry+exit) | **0.336** | 46.3% |

\* mean losses 합 대비 비율 (additive, clipped ≥0).

| summary field | value |
| --- | --- |
| selection_dominates_timing | **false** |
| **primary_gap** | **timing** |

### 5.3 Per-window dominant bucket

| bucket | count | share |
| --- | ---: | ---: |
| selection | 142 | 53.6% |
| entry_timing | 65 | 24.5% |
| exit_timing | 22 | 8.3% |
| giveback | 26 | 9.8% |
| NONE | 10 | 3.8% |

### 5.4 Worst-case windows (by loss type)

**selection** (top 3)

| window | selection_loss | realized | best_family |
| --- | ---: | ---: | --- |
| 2026-03-31 ~ 2026-05-21 | 1.123 | 1.795 | 코스피 200 정보기술 |
| 2025-08-20 ~ 2025-10-15 | 1.118 | 0.237 | krx 반도체 |
| 2025-08-21 ~ 2025-10-16 | 1.112 | 0.263 | krx 반도체 |

**entry_timing** (top 3)

| window | entry_timing_loss | realized | best_family |
| --- | ---: | ---: | --- |
| 2026-04-07 ~ 2026-05-29 | 3.159 | 2.406 | 코스피 200 정보기술 |
| 2026-04-02 ~ 2026-05-26 | 2.869 | 1.567 | 코스피 200 정보기술 |
| 2026-04-03 ~ 2026-05-27 | 2.848 | 2.148 | 코스피 200 정보기술 |

**exit_timing** (top 3)

| window | exit_timing_loss | realized | best_family |
| --- | ---: | ---: | --- |
| 2026-05-19 ~ 2026-07-08 | 1.180 | 0.749 | euro stoxx 50 index |
| 2026-05-20 ~ 2026-07-09 | 1.114 | 0.949 | phlx semiconductor sector index |
| 2026-05-18 ~ 2026-07-07 | 1.020 | 0.978 | euro stoxx 50 index |

**giveback** (top 3)

| window | giveback_loss | realized | best_family |
| --- | ---: | ---: | --- |
| 2026-05-19 ~ 2026-07-08 | 1.422 | 0.749 | euro stoxx 50 index |
| 2026-05-20 ~ 2026-07-09 | 1.408 | 0.949 | phlx semiconductor sector index |
| 2026-05-13 ~ 2026-07-02 | 1.340 | 1.127 | phlx semiconductor sector index |

windows with `giveback_loss > 0.30`: **35**

### 5.5 P3 방향 시사

| signal | implication |
| --- | --- |
| `primary_gap=timing` | entry/exit controller, regime vehicle가 1차 과제 |
| dominant selection 53.6% | family mom ensemble이 보조 레버 |
| 2026 Q2 cluster | entry_timing + giveback 동시 악화 — exit/giveback lock 검토 |

---

## 6. Sticky Family 비교 (reference, pre–sell-first runs)

아래는 sell-first 이전 artifact-complete run 기준 참고치. **채택 판정은 §3 rebaseline만 사용.**

| model | semantic_id | run_id | P>50% | q99 | ruin | gross_viol | champ_gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **P27** (prior) | sticky.mom60_raw | `20260902T091905Z_…` | 4.69% | 86.9% | 4.21% | 1,958 | FAIL |
| P28A | sticky.mom60_hold | `20260902T093604Z_…` | 4.78% | 90.2% | 4.21% | 33,397 | FAIL |
| P28B | sticky.mom60_abs_cash | `20260902T114008Z_…` | 4.11% | 76.7% | 2.58% | 87 | FAIL |
| P26 | sticky.mom60_concentrated | `20260901T034103Z_…` | **5.31%** | 89.9% | 4.55% | 0 | FAIL |
| P21 | sticky.impulse_crash | `20260902T034957Z_…` | 2.73% | 71.5% | 2.01% | — | — |
| **P31** | convex.lottery_impulse | `20260904T033654Z_…` | 4.16% | **106.5%** | **0.86%** | 70 | FAIL |

---

## 7. 벤치마크·천장

| 기준 | P>50% | 비고 |
| --- | ---: | --- |
| P21 anchor | 2.73% | impulse+crash+lock@40% |
| **P27 champion (sell-first)** | **4.26%** | 운영 채택 |
| P31 convex research | 4.16% | beta 제거·q99↑, gross_viol=70 |
| P27 prior (gross inflated) | 4.69% | 신뢰 불가 tail |
| P26 tail research | 5.31% | gross_viol=0, gate FAIL |
| Oracle (selection+timing) | ~20.1% | TASK_39, 2090 windows |
| Inverse oracle | ~0.4% | tail은 선택·타이밍 지배 |

---

## 8. 채택 판정 로직 (현재 코드)

```
CHAMPION_STRATEGY = sticky.mom60_raw   # legacy P27
ANCHOR_STRATEGY   = sticky.impulse_crash  # legacy P21
```

| 후보 | tail | gross hygiene | field | gate | 운영 |
| --- | --- | --- | --- | --- | --- |
| **P27 (sell-first)** | ★★★★ | ★★★★★ | ★★★★★ | **PASS** | **채택** |
| P31 | ★★★★☆ (q99) | ✗ (70) | ★★ | FAIL | 연구용 |
| P26 | ★★★★★ | ★★★★★ | — | FAIL | 연구용 |
| P28A | ★★★★☆ | ✗ (33397) | △ | FAIL | 기각 |
| P28B | ★★★ | ★★★★ | △ | FAIL | 기각 |
| P21 | ★★ | — | — | — | anchor |

---

## 9. 재현 명령

```bash
# Champion full-period adoption backtest (~10 min)
uv run mt-etf backtest --model P27 \
  --start 2018-01-02 --end 2026-08-27 --eval-mode adoption

# P31 convex impulse research backtest (~27 min)
uv run mt-etf backtest --model P31 \
  --start 2018-01-02 --end 2026-08-27 --eval-mode adoption

# P31 LOYO vs P27 incumbent
uv run mt-etf loyo \
  --run-id 20260904T033654Z_P31_20180102_20260827_0300_0500_0010 \
  --incumbent-run-id 20260903T034356Z_P27_20180102_20260827_0300_0500_0010

# Tail forensics attribution (~3–4 min; requires windows+trades parquet)
uv run mt-etf forensics \
  --run-id 20260903T034356Z_P27_20180102_20260827_0300_0500_0010

# Anchor 비교
uv run mt-etf backtest --model P21 \
  --start 2018-01-02 --end 2026-08-27

# 일일 의사결정
uv run mt-etf decide --model P27
```

결과 경로: `data/results/<run_id>/summary.json`

---

## 10. TSV — pandas 직접 로드용

```tsv
model	semantic_id	run_id	p_gt_30	p_gt_40	p_gt_50	p_gt_60	q50	q90	q95	q99	cvar_05	ruin	gross_viol	effective_gross_max	rts	champ_gate	adopt_gate	field_win_rate	primary_gap
P27	sticky.mom60_raw	20260903T034356Z_P27_20180102_20260827_0300_0500_0010	0.087081	0.062201	0.047368	0.039234	0.000000	0.269686	0.428840	0.808328	-0.301259	0.025359	0	1.913200	0.386601	PASS	PASS	0.590431	timing
P31	convex.lottery_impulse	20260904T033654Z_P31_20180102_20260827_0300_0500_0010	0.055502	0.052153	0.041627	0.033493	0.000000	0.002432	0.441991	1.064522	-0.134439	0.008612	70	1.900000	0.346231	FAIL	FAIL	0.335885	—
```

```python
import io, pandas as pd
tsv = open("docs/results/latest.md").read().split("```tsv\n")[1].split("\n```")[0]
df = pd.read_csv(io.StringIO(tsv), sep="\t")
```

---

## 11. 신뢰도·한계

1. **Sell-first rebaseline** (`20260903T034356Z`): TASK_41 적용; gross_violation 0, gate PASS. Prior `20260902T091905Z` tail 수치는 gross inflation으로 **비교 전용**.
2. **P31 research** (`20260904T033654Z`): beta 2x 제거·sector convex만 체결; `gross_viol=70`으로 tail 수치 신뢰 제한 — gross fix 후 재평가 필요.
3. **carry_gross_drift_count=19,430**: rolling window×session 진단 합산; unique calendar day 수가 아님.
4. **Forensics oracle**: ex-post +2x family BH — 연구용 상한 proxy; tradable claim 아님.
5. **trades.parquet vs windows.parquet**: artifact completion 시 full-span `engine.run` trades + path_dependent rolling windows — attribution에 minor path mismatch 가능.
6. **우승확률 아님**: `P>50%≈4%` 수준은 역사 우승(+48%)·oracle(20%) 대비 gap 큼.

---

## 12. Artifact Index

| model | run_id | path | notes |
| --- | --- | --- | --- |
| **P27** ★ | `20260903T034356Z_P27_20180102_20260827_0300_0500_0010` | `data/results/…/` | sell-first, gate PASS, forensics |
| **P31** | `20260904T033654Z_P31_20180102_20260827_0300_0500_0010` | `data/results/…/` | convex impulse research, loyo_report |
| P27 (superseded) | `20260902T091905Z_P27_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` | gross_viol=1958, gate FAIL |
| P28A | `20260902T093604Z_P28A_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` | HOLD drift, 기각 |
| P28B | `20260902T114008Z_P28B_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` | ruin 최저, tail 열위 |
| P26 | `20260901T034103Z_P26_20180102_20260827_0300_0500_0010` | `data/results/…/` | tail research |
| P21 | `20260902T034957Z_P21_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` | anchor |

**다음 갱신 트리거**: P31 gross fix · gate calibration · P3 family/regime alpha · silver end-date 연장.
