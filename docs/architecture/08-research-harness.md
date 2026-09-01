# 08. Research Harness — L6~L7 계층

백테스트·rolling-36D 분포·2025 replay 로 전략을 검증하는 연구 인프라입니다.

---

## 1. 두 종류의 백테스트 (재확인)

| | Deployment | Structural |
| --- | --- | --- |
| Universe | **후원 운용사 ETF** + 유동성·규칙 | 각 시점 실제 존재 ETF (후원사 필터 없음) |
| 기간 | 2024-01 ~ | 2018-01 ~ |
| 목적 | 이번 대회 전략·ML·채택 판정 | 아이디어 구조적 타당성 |
| 리포트 라벨 | `[DEPLOY]` | `[STRUCT]` |
| 후원사 필터 | **ON** (`universe.yaml` → `modes.deployment`) | OFF |

**혼합 금지.** 하나의 숫자로 합치지 않습니다.

---

## 2. BacktestEngine

```python
engine = BacktestEngine(
    execution=NextOpenExecution(),
    costs=CostModel(CostConfig.default()),
    calendar=TradingCalendar(),
)
result = engine.run(
    signals=strategy_signals,
    prices=etf_daily,
    universe=universe_provider,
    start=date(2024, 1, 1),
    end=date(2026, 8, 27),
)
```

### 2.1 일별 루프

```
for session t in calendar.sessions:
    1. universe(t)           — PIT
    2. features(t)           — PIT
    3. alpha.score(t)        — signal
    4. portfolio.weights(t)  — target
    5. execution.fill(t+1)   — next-open
    6. costs.apply()
    7. record NAV
```

### 2.2 NextOpenExecution

- `decision_date = t` (close 관측 후)
- `execution_date = next_session(t)` (다음 시가 체결)
- same-bar 금지 (INV-3)

---

## 3. Rolling 36-Day Tournament Simulator

대회와 동일한 **36 session window** 를 역사적으로 굴립니다.

```python
sim = TournamentSimulator(window=36, calendar=calendar)
distribution = sim.run(
    strategy=my_strategy,
    start=date(2018, 1, 1),
    end=date(2026, 8, 27),
)
```

### 3.1 Window 생성

```
sessions = calendar.sessions_in_range(2018-01-01, 2026-08-27)
for i in range(len(sessions) - 35):
    window = sessions[i : i+36]
    R_i = strategy_return(window)
    distributions.append(R_i)
```

2026 대회: 36 sessions. 2025 제2회: 35 sessions → replay 시 별도 처리.

### 3.2 산출물: ReturnDistribution

| 지표 | 설명 |
| --- | --- |
| `P(R > θ)` | θ ∈ {10, 20, 30, 40, 50}% |
| `quantiles` | q05, q25, q50, q75, q95, q99 |
| `CVaR_05` | worst 5% 평균 |
| `MDD_dist` | window별 MDD 분포 |
| `giveback_median/q90` | peak_to_final_giveback median·q90 (INV-25) |
| `n_effective` | overlap 보정 유효 샘플 수 |

### 3.3 Overlap 보정

연속 window 는 35일 겹침 → 독립 표본 아님.

```
n_effective ≈ n_windows × (1 - overlap_fraction)
             ≈ n_windows / 36
```

bootstrap·신뢰구간 계산 시 사용.

---

## 4. 2025 Tournament Replay

제2회 대회 기간을 **day-by-day** 재현합니다.

```python
replay = TournamentReplay(
    tournament_start=date(2025, 9, XX),  # 실제 일정
    tournament_end=date(2025, 11, XX),
    sessions=35,
)
result = replay.run(strategy, prices, features, universe)
```

### 4.1 목적

- 2025 데이터로 전략이 어떤 순위권 수익률을 냈을지 **참조**
- 분포 추정치가 아닌 **단일 경로 sanity check**
- parameter tuning 용도 금지 (overfitting)

### 4.2 출력

```
Day 1:  entry KODEX 반도체, weight=0.6
Day 5:  rebalance → TIGER 2차전지, weight=0.7
...
Final:  R = +42.3%  (hypothetical rank ≈ top 5%)
```

---

## 5. Robustness Grid

전략 채택 전 필수 sweep:

| 축 | 값 |
| --- | --- |
| Cost (commission + slippage) | low / mid / high |
| Liquidity (participation rate) | 1% / 2% / 5% |
| Leverage scenario | **allow (primary)** / deny (fallback) |
| Universe | deployment / structural |

모든 조합에서 $P(R > 30\%)$ 와 $P(R < -25\%)$ 를 확인합니다. **하나라도 G1·G2a 위반 시 기각.**

참여율 축이 중요한 이유는 유니버스 크기가 참여율에 급격히 반응하기 때문입니다 (1% → 26종목, 5% → 65종목, [05 §4.3](05-universe-and-instruments.md)). 하나의 참여율로만 검증한 결과는 신뢰할 수 없습니다.

레버리지 축은 **두 시나리오 모두 게이트를 통과해야 합니다.** allow 가 primary 이지만, 규칙이 deny 로 확정될 경우 대회 첫날 쓸 전략이 준비되어 있어야 합니다.

---

## 6. Stationary Bootstrap

window 수익률의 autocorrelation 을 보존하면서 신뢰구간 추정:

```python
ci = stationary_bootstrap(
    returns=distribution.window_returns,
    n_bootstrap=1000,
    block_length=auto,
    stat=lambda x: percentile(x, 95),
)
```

---

## 7. Competitor Field

참가자 CDF 는 미관측입니다. `CompetitorField.rank_interval` 은 **stress 구간**만 반환합니다. 스칼라 `win_probability` 를 의사결정 입력으로 쓰지 않습니다 (INV-08-7).

식별 가능한 진단은 **같은 36일 window 의 구현된 rival 수익률**입니다.

```
field_relative_report(candidate, rivals) → win_rate, top2_rate, median_rank_percentile
win = I(R_candidate > max_j R_rival_j)   # 동점은 패배
```

채택 게이트를 이 값으로 교체하지 않습니다. $n_{\text{effective}} \approx n/36$ 이라 1-0 win 지표의 CI 가 더 넓습니다.

### 7.1 Annual one-shot (진단)

Rolling 과 별도로, 매년 `date(Y,9,21)` 이후 **첫 세션에서 36일 1회**만 자릅니다 (2025-09-22 · 2026-09-21 정렬). 표본 ~8개라 게이트가 아닙니다.

---

## 8. Accept / Reject 게이트

### 8.1 CVaR 게이트를 폐기하는 이유

초기 설계의 `G2: CVaR(5%) 악화 ≤ 3%p vs B0` 는 **레버리지 허용과 논리적으로 충돌**합니다. `+2x` 포지션은 정의상 CVaR 을 크게 악화시키므로, 이 게이트를 유지하면 레버리지 전략이 실제 성능과 무관하게 자동 기각됩니다.

근본 원인은 게이트가 **잘못된 효용함수**를 쓰고 있다는 것입니다. 대회 보상은 순위의 계단 함수입니다.

```
rank 1   → 1,000만
rank 2   →   500만
rank 3   →     0
rank 400 →     0        ← rank 3 과 동일한 보상
```

3위와 400위의 보상이 같으므로, 좌측 꼬리를 대칭적으로 벌하는 CVaR 은 목적함수와 정합하지 않습니다.

다만 좌측 꼬리가 **완전히 무해한 것도 아닙니다**. 두 가지 실질 제약이 있습니다.

1. **회복 불가능성** — 36세션 중 `-40%` 를 맞으면 남은 기간에 `+67%` 가 필요합니다. 사실상 대회 종료입니다.
2. **부문별 우수상(100만)** — 대상보다 달성 가능성이 높고, 중간 수준 수익률로도 노릴 수 있습니다.

### 8.2 재정의된 게이트

| # | 조건 | 성격 |
| --- | --- | --- |
| G1 | B0 대비 $P(R>30\%)$ ≥ +2%p | 우측 꼬리 |
| **G2a** | $P(R_{36d} < -25\%) \le 5\%$ | **파산 제약** (B0 상대가 아닌 절대 기준) |
| **G2b** | $\mathbb{E}[\text{Prize}]$ ≥ B0 수준 | 보상 정합 |
| G3 | robustness grid 전 조합 PASS | 강건성 |
| G4 | 2025 replay R > median anchor | **경고** (INV-23, 단독 기각 아님) |
| G5 | structural + deployment 모두 양(+)의 median | 구조적 타당성 |
| overlay vs raw | primary scenario CI lower bound $\ge 0$ vs raw | overlay 채택 불변식. 위반 시 identity/raw 가 live |
| field win_rate | 구현 rival 대비 | 진단 (기각 사유 아님) |

CVaR(5%) 와 MDD 분포는 **하드 게이트가 아니라 리포트 필수 진단 지표**로 유지합니다. 기각 사유는 못 되지만 항상 함께 보여줍니다.

`-25%` 와 `5%` 는 `configs/gates.yaml` 값입니다. 코드에 박지 않습니다.

### 8.3 ML 전용 추가 게이트

ML 모델은 G1~G5 에 더해 아래를 통과해야 합니다. 상세는 [12-ml-layer.md](12-ml-layer.md) 를 참조하십시오.

| # | 조건 |
| --- | --- |
| G6 | purged walk-forward 전 fold 에서 IC > 0, 부호 일관 |
| G7 | best rule baseline 대비 `P(R>30%)` 개선 |
| G8 | fold 간 성능 편차가 baseline 대비 크게 확대되지 않음 |

하나라도 FAIL → **기각**. 수정 후 재검증.

---

## 9. CLI

| Command | 동작 |
| --- | --- |
| `mt-etf backtest --strategy B1 --mode deploy` | deployment 백테스트 |
| `mt-etf backtest --strategy B1 --mode struct` | structural 백테스트 |
| `mt-etf replay --strategy leadership --year 2025` | 2025 replay |
| `mt-etf simulate --window 36` | rolling-36D 분포 |
