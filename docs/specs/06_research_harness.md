# 06. Research Harness — next-open backtest · rolling-36D distribution · baselines · 2025 replay

**선행**: [05_feature_engine](05_feature_engine.md)
**상위**: [00_architecture.md](00_architecture.md)

---

## 1. Diagnosis & Invariants

### 1.1 이 spec 이 프로젝트의 실질적 결승선이다

W3 종료 시점(2026-09-16, 참가 접수 마감일)에 이것까지 완성되면 **검증된 baseline 전략으로 대회에 나갈 수 있다**. 07·08 은 개선이지 필수가 아니다.

산출해야 하는 것은 단일 수익률이 아니라 **분포**다(INV-OBJ).

### 1.2 체결 규칙 — look-ahead 의 마지막 관문

$$
\text{signal} = f\big(\text{data}(\le t)\big) \;\longrightarrow\; \text{fill at } \texttt{open}(t{+}1)
$$

`close(t)` 를 보고 `close(t)` 에 체결하는 것은 look-ahead 다. 그런데 이 실수는 코드에서 매우 자연스럽게 발생한다 — 신호 프레임과 가격 프레임을 같은 날짜로 조인하면 끝이다.

**INV-06-1 (구조적 분리)**: 엔진은 `decision_date` 와 `execution_date` 를 **서로 다른 필드로** 들고 다니며, `execution_date = calendar.next_session(decision_date)` 로만 생성된다. 체결가 조회는 `execution_date` 의 `open` 컬럼에서만 이루어진다. 같은 날짜로 조인하는 코드 경로가 존재하지 않도록 API를 설계한다.

`decision_date` 가 구간의 마지막 세션이면 체결할 다음 세션이 없다 → 그 신호는 **버린다**(마지막 날 신호로 체결한 척하지 않는다).

### 1.3 Rolling-36D — 겹치는 표본을 독립으로 착각하지 않기

가능한 모든 시작일에서 36세션 창을 굴리면 2018~2026 구간에서 약 2,090개의 창이 나온다. 그러나 인접한 두 창은 35 세션을 공유한다. **이 2,090개를 독립 표본으로 보고 신뢰구간을 계산하면 폭이 실제의 1/6로 축소**되어, 존재하지 않는 유의성을 만들어낸다.

**INV-06-2 (유효 표본수)**: 모든 분포 통계는 다음을 함께 보고한다.

$$
n_{\text{eff}} = \left\lfloor \frac{n_{\text{windows}}}{h} \right\rfloor, \qquad h = 36
$$

$n=2{,}090$, $h=36$ → $n_{\text{eff}} \approx 58$. **58개**가 우리가 실제로 가진 정보량이다.

신뢰구간은 i.i.d. 부트스트랩이 아니라 **stationary bootstrap** (Politis–Romano, 기대 블록길이 $h$)으로 계산한다. 시계열 의존성을 블록으로 보존해야 CI 폭이 정직해진다.

### 1.4 평가 지표 — CAGR 을 쓰지 않는 이유

36세션 단일 구간 문제에서 CAGR은 의미가 없다(연율화가 정보를 더하지 않고 해석만 왜곡한다). 산출 지표:

```
mean, median
q05 q25 q50 q75 q90 q95 q99
P(R > 0.10)  P(R > 0.20)  P(R > 0.30)  P(R > 0.40)  P(R > 0.50)
CVaR(5%)  =  worst 5% 평균
MDD 분포 (창별 최대낙폭의 분포)
n_windows, n_effective
right_tail_score
```

**Right tail score** (연구용 스칼라, 결정용 아님):

$$
\text{RTS} = 0.2\,q_{75} + 0.3\,q_{90} + 0.3\,q_{95} + 0.2\,q_{99}
$$

가중치는 `configs/strategies.yaml` 파라미터다(INV-11). **RTS 단독으로 전략을 채택하지 않는다** — INV-TAIL 에 따라 하위 꼬리와 시장 조건부 성능을 함께 본다.

### 1.5 Baseline 사다리 (INV-10)

새 전략은 반드시 아래와 **동일 프로토콜**(같은 universe, 같은 비용, 같은 창 집합)로 비교된다.

| ID | 전략 | 검증 목적 |
| --- | --- | --- |
| B0 | KOSPI 200 ETF Buy & Hold | 시장 대비 초과가 있는가 |
| B1 | Top-1 `mom_20` | 가장 단순한 집중 모멘텀 |
| B2 | Top-3 `mom_20` 동일가중 | 집중도의 효과 |
| B3 | `mom_20` + trend filter (`close > MA20`) | 추세 필터의 효과 |
| B4 | theme 모멘텀 → 테마 내 최우량 ETF | 섹터 로테이션의 효과 |
| B5 | B4 + regime gate (risk-off 시 현금) | regime 의 효과 |

**연구 질문과의 대응** (next.md §67): B1 vs B2 = 집중도, B1 vs B3 = 추세, B2 vs B4 = 섹터, B4 vs B5 = regime. 각 쌍의 차이가 그 요소의 순수 기여다.

### 1.6 2025 대회 Replay — 가장 가치 있는 단일 검증

2025-09-22 ~ 2025-11-14 (**35 세션**). 데이터 가용성 실측 확인: 2025-09-22 → 1,019 ETF, 2025-11-13 → 1,044 ETF.

replay 는 하루씩만 데이터를 공개하면서 **live 와 완전히 동일한 코드 경로**를 실행한다. 매일 기록하는 것:

```
date · regime state/score · universe size + drop counts
top-5 ranking (ticker, name, theme, score)
portfolio weights · turnover
realized daily return · cumulative return
```

**수용 게이트 G-4**: 대회 시작 전 데이터만으로 당시 주도 섹터를 포착했는가. 이 판정은 자동화하지 않는다 — 리포트가 실제 선택 종목명을 남기고 사람이 판정한다. (2025 우승자 최종 +47.82%, 5주차 선두 +72.28% 가 참조점이다.)

replay 가 단순 백테스트보다 나은 점은 **디버깅 가능성**이다. "왜 10월 7일에 이 종목을 샀는가"에 그날의 regime·universe·랭킹으로 답할 수 있다.

### 1.7 비용 — 모르는 것을 하나로 고정하지 않기

R-3 에 따라 수수료·슬리피지가 미확정이다.

**INV-06-3**: `CostConfig` 의 필드가 `None`(Unknown) 이면 harness 는 단일 실행이 아니라 **그리드 실행**을 한다. 기본 그리드 `commission_bps ∈ {0, 1.5, 3, 15}`, `slippage_bps ∈ {0, 5, 20}`. 결과 표는 항상 비용 축을 포함한다.

턴오버가 높은 전략은 비용에 민감하므로, 비용 축을 지우면 전략 순위가 뒤집힐 수 있다.

### 1.8 포트폴리오 최소 계층

06 에서는 baseline 실행에 필요한 **최소 sizing** 만 만든다. cluster 중복제거·confidence 집중도·state machine 은 08 소관이다.

| scheme | 가중치 |
| --- | --- |
| `TOP1` | 100% |
| `TOP2_70_30` | 70 / 30 |
| `TOP3_50_30_20` | 50 / 30 / 20 |
| `EQUAL_K` | 1/k 균등 |

**INV-7 강제**: `normalize_weights` 가 $\sum w_i + w_{\text{cash}} = 1 \pm 10^{-6}$ 를 검증하고 위반 시 `WeightViolationError`.

### 1.9 복잡도 예산

- 백테스트 1회: $T$ 세션 × $O(|\mathcal{U}|)$. 사전 계산된 gold feature 패널을 날짜 슬라이스만 하므로 세션당 sub-ms 목표.
- rolling-36D 는 **백테스트를 창마다 재실행하지 않는다.** 전 구간 일별 수익률 시계열을 1회 산출한 뒤, 창 수익률은 누적 로그수익률의 차분으로 $O(T)$ 에 계산한다.

$$
R_{[a,b]} = \exp\!\Big(\textstyle\sum_{t=a+1}^{b} \log(1+r_t)\Big) - 1
$$

$O(T \cdot h)$ 재실행(수천 초)을 $O(T)$(밀리초)로 낮추는 핵심 최적화다.
단, **경로 의존 전략**(stop-loss, state machine)은 이 단축이 성립하지 않는다 → 08 도입 시 창별 재실행 모드를 별도 플래그로 제공하고, 그때 비용을 감수한다.

---

## 2. Architecture & Mitigation

```
 gold/etf_features.parquet + UniverseSnapshot
                  │
                  ▼
        alpha/base.py  AlphaModel.score(snapshot, ctx) -> {ticker: score}
                  │            ▲ baselines.py  B0~B5
                  ▼
        portfolio/sizing.py  weights_from_scores
        portfolio/constraints.py  normalize_weights (INV-7)
                  │
                  ▼
        backtest/engine.py  BacktestEngine.run
          decision_date(t) ──► execution_date = next_session(t)
                  │              fill = open(execution_date)
                  ├─ costs.py  CostModel
                  └─ metrics.py  equity · drawdown
                  │
                  ▼
        tournament/simulator.py  run_rolling(h=36)
                  │
                  ▼
        tournament/distribution.py  ReturnDistribution
          quantiles · exceedance · RTS · n_effective · stationary bootstrap
                  │
                  ▼
        tournament/replay.py  2025 day-by-day report
```

### 2.1 `AlphaModel` 계약

```python
class AlphaModel(Protocol):
    name: str
    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]: ...
```

`snapshot` 은 `FeatureBuilder.snapshot` 산출물 — **이미 universe 로 필터되고 decision_date 한 날짜만 담긴** 프레임이다. 모델은 날짜 필터링 책임을 지지 않으며, 애초에 미래 행에 접근할 수 없다.

`DecisionContext` 는 `decision_date`, `regime`, `rules`, `capital`, `held_positions` 를 담는다.

### 2.2 엔진 루프

```
for t in sessions[start..end]:
    e = calendar.next_session(t)          # 없으면 루프 종료
    snap  = features.snapshot(panel, universe.get(t, filters))
    scores = model.score(snap, ctx(t))
    target = normalize_weights(weights_from_scores(scores, k, scheme))
    fills  = execution.fill(target, panel, execution_date=e)   # open(e)
    costs  = cost_model.apply(fills)
    portfolio.apply(fills, costs)
```

체결가는 `open(e)` 다. `open` 이 `None` 이거나 그 종목이 `is_tradable=False` 면 **주문이 체결되지 않는다** — 기존 포지션을 유지하고 미체결을 기록한다. 조용히 종가로 대체하지 않는다.

### 2.3 결과 객체

```python
BacktestResult(
    name, daily_returns,      # pl.DataFrame(date, ret, equity)
    positions, trades,
    unfilled,                 # 미체결 기록 — 유동성 가정 검증용
    config,
)
```

`unfilled` 이 많다면 유동성 필터가 헐겁다는 신호다.

---

## 3. Assumptions

- **A-1**: 초기자본 10억, 현금 잔액은 무이자.
- **A-2**: 분배금은 기본 미반영(PR 기준). R-4 확정 시 TR 모드 추가.
- **A-3**: 부분체결 모델링은 하지 않는다. 유동성 필터가 이미 주문/ADV 를 제약하므로 이중 모델링이다. 대신 `unfilled` 로 가정 위반을 관측한다.
- **A-4**: rolling 창 시작일은 모든 세션. 대회 시작일이 특정 요일이라는 제약은 두지 않는다(2026-09-21 은 월요일이지만 이를 조건으로 표본을 줄이면 $n_{\text{eff}}$ 가 붕괴한다).

---

## 4. Execution Target

```bash
uv run pytest tests/unit/backtest tests/unit/tournament -q --tb=short
uv run python tools/agent_skills/lean_check.py --spec docs/specs/06_research_harness_contract.json

uv run mt-etf backtest --model B1 --start 2018-01-01 --end 2026-08-27 --horizon-from-rules
uv run mt-etf replay --year 2025 --model B4
```

## 5. 완료 판정 (W3 게이트)

1. B0~B5 전부가 동일 프로토콜로 실행되고 `ReturnDistribution` 이 산출된다.
2. `P(R>θ)` 곡선과 $n_{\text{eff}}$ 가 함께 보고된다.
3. 비용 그리드와 유동성 그리드(`φ ∈ {1,2,5,10}%`)가 결과표의 축으로 존재한다.
4. 2025 replay 리포트에 35개 세션의 일별 의사결정 로그가 남는다.
