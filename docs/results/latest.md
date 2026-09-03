# Latest — KRX Rolling-36D Championship Results

**as_of**: 2026-09-03  
**scope**: 현재 코드베이스(`main`)에서 **신뢰 가능·재현 가능**한 full-period 백테스트만 포함  
**primary_run**: `20260902T091905Z_P27_20180102_20260827_0300_0500_0010`  
**champion**: `P27` (`sticky.mom60_raw`) · **anchor**: `P21` (`sticky.impulse_crash`)

---

## 1. Executive Summary

| 판정 | 내용 |
| --- | --- |
| **운영 채택 (decide/backtest)** | **P27** — `CHAMPION_STRATEGY`, execution-correct rebaseline 완료, field-relative 진단 포함 |
| **우측 꼬리 연구 1위** | **P26** — `P>50%=5.31%`, `championship_score=0.064`, `gross_violation=0` (단, championship gate FAIL) |
| **구조 실험 (비채택)** | P28A tail micro-lift (`q99=90.2%`) but `gross_violation=33,397` (HOLD drift; 신뢰 불가) |
| **리스크 완화 실험** | P28B — `ruin=2.58%` 최저, tail 열위 (`P>50%=4.11%`) |
| **역사 벤치마크** | 제2회 우승 +47.82% · P27 2025 oneshot +41.4% · oracle ceiling `P>50%≈20.1%` |

**핵심**: 현 로직에서 **우승권(고수익 tail)에 가장 근접하면서 운영 신뢰도가 높은 결과는 P27**이다. P26이 raw tail 지표는 더 높으나 overlay/CI gate 미통과로 채택하지 않는다. 어떤 후보도 oracle 대비 `P>50%` gap이 크며, **실제 우승확률 수준은 아님** (roadmap 명시).

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
| Exposure (P21–P28) | max_single=0.95, max_gross=1.90, min_cash=0.05 |

`championship_score` = `configs/gates.yaml` 의 `championship` 시나리오 가중 exceedance  
(가중치: P>30%×0.10 + P>40%×0.25 + P>50%×0.45 + P>60%×0.20).

---

## 3. Sticky Family 비교 (full-period, 최신 run)

### 3.1 요약 표

| model | semantic_id | run_id | P>30% | P>40% | P>50% | P>60% | q50 | q99 | ruin | gross_viol | champ_gate | field_win% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **P27** ★ | sticky.mom60_raw | `20260902T091905Z_…` | 11.10% | 6.99% | **4.69%** | — | +0.12% | 86.9% | 4.21% | 1,958 | FAIL | **56.8%** |
| P28A | sticky.mom60_hold | `20260902T093604Z_…` | 11.05% | 7.03% | 4.78% | — | +0.16% | **90.2%** | 4.21% | 33,397 | FAIL | 22.8% |
| P28B | sticky.mom60_abs_cash | `20260902T114008Z_…` | 8.47% | 5.45% | 4.11% | — | 0.00% | 76.7% | **2.58%** | 87 | FAIL | 20.5% |
| P26 | sticky.mom60_concentrated | `20260901T034103Z_…` | **11.24%** | **8.04%** | **5.31%** | **4.55%** | 0.00% | 89.9% | 4.55% | **0** | FAIL | — |
| P21 | sticky.impulse_crash | `20260902T034957Z_…` | 7.56% | 3.88% | 2.73% | 2.15%† | −0.79% | 71.5% | **2.01%** | — | — | — |
| B1 | baseline.mom20_top1 | `20260831T044747Z_…` | 4.07% | 2.82% | 2.25% | 1.44%† | −5.65% | 66.3% | 6.79% | — | — | — |

★ = 운영 champion.  
† = `windows.parquet` 보유 run 기준 (`P21`/`B1`은 구버전 run; sticky 최신 run은 parquet 미저장).  
`P>60%`·`championship_score` 미기재 = adoption run에 `windows.parquet` 없음 (P27/P28A/P28B).

### 3.2 Championship score (windows.parquet 보유 시)

| model | P>30% | P>40% | P>50% | P>60% | championship_score | hot_field_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P26 | 11.24% | 8.04% | 5.31% | 4.55% | **0.0643** | 0.0516 |
| P27 | 11.10% | 6.99% | 4.69% | (미저장) | ~0.050‡ | — |
| P28A | 11.05% | 7.03% | 4.78% | (미저장) | ~0.050‡ | — |
| P21 | 7.56% | 3.88% | 2.73% | 2.15% | 0.0295 | 0.0134 |
| B1 | 4.07% | 2.82% | 2.25% | 1.44% | 0.0348 | 0.0481 |

‡ P>60% 미확인 시 championship_score는 하한 추정치; P26이 sticky 계열 확정 최고.

### 3.3 TSV — pandas 직접 로드용

```tsv
model	semantic_id	run_id	p_gt_30	p_gt_40	p_gt_50	q50	q90	q95	q99	cvar_05	ruin	gross_viol	rts	champ_gate	adopt_gate	field_win_rate
P27	sticky.mom60_raw	20260902T091905Z_P27_20180102_20260827_0300_0500_0010	0.111005	0.069856	0.046890	0.001191	0.320603	0.486446	0.869285	-0.372048	0.042105	1958	0.442674	FAIL	FAIL	0.567943
P28A	sticky.mom60_hold	20260902T093604Z_P28A_20180102_20260827_0300_0500_0010	0.110526	0.070335	0.047847	0.001632	0.321312	0.489228	0.901529	-0.492793	0.042105	33397	0.450220	FAIL	FAIL	0.227751
P28B	sticky.mom60_abs_cash	20260902T114008Z_P28B_20180102_20260827_0300_0500_0010	0.084689	0.054545	0.041148	0.000000	0.268658	0.418207	0.766631	-0.441767	0.025837	87	0.374611	FAIL	FAIL	0.205263
P26	sticky.mom60_concentrated	20260901T034103Z_P26_20180102_20260827_0300_0500_0010	0.112440	0.080383	0.053110	0.000000	0.328374	0.515886	0.899239	-0.355624	0.045455	0	0.456940	FAIL	FAIL	
P21	sticky.impulse_crash	20260902T034957Z_P21_20180102_20260827_0300_0500_0010	0.075598	0.038756	0.027273	-0.007892	0.246508	0.361331	0.715148	-0.245954	0.020096		0.342961			
B1	baseline.mom20_top1	20260831T044747Z_B1_20180102_20260827_0300_0500_0010	0.040670	0.028230	0.022488	-0.056543	0.130475	0.258596	0.662626	-0.313983	0.067943		0.256330			
```

```python
import io, pandas as pd
tsv = open("docs/results/latest.md").read().split("```tsv\n")[1].split("\n```")[0]
df = pd.read_csv(io.StringIO(tsv), sep="\t")
```

---

## 4. Champion 상세 — P27

### 4.1 분포 지표

| 지표 | 값 |
| --- | ---: |
| median (q50) | +0.12% |
| q90 | +32.1% |
| q95 | +48.6% |
| q99 | +86.9% |
| CVaR(5%) | −37.2% |
| giveback_median | 7.75% |
| giveback_q90 | 22.1% |
| right_tail_score | 0.443 |
| objective_ruin (P<-25%) | 4.21% |
| effective_gross_max | 3.55 |
| gross_violation_count | 1,958 |

### 4.2 Field-relative (vs P21 incumbent, 동일 exposure)

| 지표 | 값 |
| --- | ---: |
| win_rate | 56.8% |
| top2_rate | 100% |
| median_rank_percentile | 50% |

### 4.3 Annual oneshot (36D, 대회 시작일 anchor)

| year | start | return |
| ---: | --- | ---: |
| 2018 | 2018-09-21 | −31.6% |
| 2019 | 2019-09-23 | −3.4% |
| 2020 | 2020-09-21 | +4.0% |
| 2021 | 2021-09-23 | −1.2% |
| 2022 | 2022-09-21 | +7.6% |
| 2023 | 2023-09-21 | −6.1% |
| 2024 | 2024-09-23 | −12.3% |
| **2025** | **2025-09-22** | **+41.4%** |

제2회 우승 +47.82% 대비 −6.4pp. sticky 계열 중 역사 anchor에 가장 근접.

### 4.4 Gate 상태

| gate | status | failures |
| --- | --- | --- |
| objective | PASS | — |
| championship | FAIL | `GROSS_EXPOSURE` |
| adoption | FAIL | `GROSS_EXPOSURE` |

---

## 5. 벤치마크·천장

| 기준 | P>50% | 비고 |
| --- | ---: | --- |
| B1 (mom20 Top1) | 2.25% | baseline 하한 |
| P21 anchor | 2.73% | impulse+crash+lock@40% |
| **P27 champion** | **4.69%** | 운영 채택 |
| P26 tail research | 5.31% | gross_viol=0, gate FAIL |
| Oracle (selection+timing) | ~20.1% | TASK_39, 2090 windows |
| Inverse oracle | ~0.4% | tail은 선택·타이밍 지배 |

---

## 6. 채택 판정 로직 (현재 코드)

```
CHAMPION_STRATEGY = sticky.mom60_raw   # legacy P27
ANCHOR_STRATEGY   = sticky.impulse_crash  # legacy P21
```

| 후보 | tail | gross hygiene | field | 운영 |
| --- | --- | --- | --- | --- |
| P27 | ★★★★ | △ (1958 viol) | ★★★★★ | **채택** |
| P26 | ★★★★★ | ★★★★★ | — | 연구용 |
| P28A | ★★★★☆ | ✗ (33397) | △ | 기각 |
| P28B | ★★★ | ★★★★ | △ | 기각 |
| P21 | ★★ | — | — | anchor 비교 |

---

## 7. 재현 명령

```bash
# Champion full-period (약 7분)
uv run mt-etf backtest --model P27 \
  --start 2018-01-02 --end 2026-08-27

# Anchor 비교
uv run mt-etf backtest --model P21 \
  --start 2018-01-02 --end 2026-08-27

# 일일 의사결정
uv run mt-etf decide --model P27
```

결과 경로: `data/results/<run_id>/summary.json`

---

## 8. 신뢰도·한계

1. **P27 rebaseline** (`20260902T091905Z`): TASK_37 execution-correctness 이후; `gross_violation_count` 수집됨 (이전 run `GROSS_METRIC_UNAVAILABLE`).
2. **P28A gross_viol=33,397**: HOLD 가격 drift artifact — tail 수치 참고만, 채택 금지.
3. **P27 gross_viol=1,958**: `effective_gross_max=3.55` — exposure invariant 미충족; championship/adoption gate FAIL 원인.
4. **windows.parquet**: P27 adoption run은 summary만 저장 — `P>60%`·일별/거래 forensics 필요 시 artifacts 옵션으로 재실행.
5. **우승확률 아님**: `P>50%≈5%` 수준은 역사 우승(+48%)·oracle(20%) 대비 gap 큼.

---

## 9. Artifact Index

| model | run_id | path |
| --- | --- | --- |
| P27 ★ | `20260902T091905Z_P27_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` |
| P28A | `20260902T093604Z_P28A_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` |
| P28B | `20260902T114008Z_P28B_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` |
| P26 | `20260901T034103Z_P26_20180102_20260827_0300_0500_0010` | `data/results/…/` (+ daily/trades/windows) |
| P21 | `20260902T034957Z_P21_20180102_20260827_0300_0500_0010` | `data/results/…/summary.json` |
| B1 | `20260831T044747Z_B1_20180102_20260827_0300_0500_0010` | `data/results/…/` (+ windows) |

**다음 갱신 트리거**: P27 gross gate PASS rebaseline · P29+ 채택 · silver end-date 연장.
