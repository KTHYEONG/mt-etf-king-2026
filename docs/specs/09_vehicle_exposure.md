# 09. Vehicle Exposure Wiring — ExposureSelector · gross exposure · aggression(default off)

**선행**: ADR_20260829_08_PORTFOLIO_TOURNAMENT, ADR_20260829_B2_STATE_BACKFILL_PERF  
**근거**: [plans/feedback-architecture-adoption.md](../plans/feedback-architecture-adoption.md) §7 Phase 4 잔여, [architecture/07](../architecture/07-alpha-and-portfolio.md) §5.2–5.3, INV-14/17/18/19/24, ARCH-5/6  
**상태**: Blueprint + contract (본 문서 + `09_vehicle_exposure_contract.json`)

---

## 1. Diagnosis & Invariants

### 1.1 병목

Phase 4에서 sizing(B-1)·state(B-2)·decide WHY는 동작한다. 그러나:

| 모듈 | 상태 | 영향 |
| --- | --- | --- |
| `ExposureSelector` | 단위 테스트만, `allocate` 미호출 | INV-24 / ARCH-6 위반 — alpha ticker가 곧 vehicle |
| `gross_exposure` | `portfolio.yaml`에만 존재 | INV-18 미강제 — +2x 80%면 gross=1.6인데 비중만 검사 |
| `AggressionPolicy` | 기본 off 테스트만, 파이프 미연결 | B-4 게이트 측정 불가 |
| `P08` / `decide` | `master=None` | family·confidence 기반 배수 선택 불가 |

제2회 실측(mt-data-report)의 공통 패턴은 **주도섹터 + 레버리지 vehicle**이다. vehicle 미연결이면 P08은 1x 종목에 갇혀 연구 질문 #11에 답할 수 없다.

### 1.2 Core Invariants

$$
\texttt{vehicle}(f, r, L, c) =
\begin{cases}
+1 & L=\texttt{UNKNOWN}\ \lor\ c=\texttt{LOW}\ \lor\ r\notin\{\texttt{RISK\_ON},\texttt{STRONG\_RISK\_ON}\} \\
m^\star & L=\texttt{True}\ \land\ c\neq\texttt{LOW}\ \land\ \text{ADV ok}
\end{cases}
$$

$$
G = \sum_i |w_i \cdot m_i| \le G_{\max}\quad(G_{\max}=1.60\ \text{config})
$$

- **INV-09-1 (alpha ≠ vehicle)**: `score` 키는 theme/`index_key` 후보; 최종 `weights` 키는 `ExposureSelector`가 고른 `isu_cd`.
- **INV-09-2 (fail-closed UNKNOWN)**: `leverage_allowed is None` → +1x만 (ARCH-5 라이브 경로). harness는 aggressive/conservative **양쪽 분포**를 별도 실행.
- **INV-09-3 (LOW → +1)**: `InstrumentAttributes.confidence == LOW` 이면 배수 강제 +1 (INV-19).
- **INV-09-4 (실효 노출)**: `apply_gross_exposure_cap` 후 $G \le G_{\max}$ (INV-18).
- **INV-09-5 (유동성 재검사)**: vehicle 교체 후 교체된 ticker의 ADV로 `apply_liquidity_cap` 재적용. 미달 시 +1x로 강등.
- **INV-09-6 (실제 가격)**: 배수는 합성 금지 — family 내 **실제 ETF ticker**만 (INV-14).
- **INV-09-7 (aggression default off)**: `AggressionPolicy(enabled=False)` 기본. on일 때만 `risk_multiplier`로 비중 스케일(합≤1 유지).

### 1.3 Complexity

- `ExposureSelector.select`: family members $O(M_f)$, $M_f \le 4$ → 상수.
- `allocate` vehicle pass: $O(K)$ selected positions ($K\le 3$).
- harness dual scenario: 기존 path_dependent 비용 ×2 (conservative/aggressive). 허용.

---

## 2. Architecture & Mitigation

### 2.1 Pipeline order (변경 후)

```
scores (alpha tickers)
  → select_positions
  → confidence_weights
  → ExposureSelector.pick_vehicle / select   ← NEW (regime · L · confidence)
  → remap weights keys to vehicles
  → state multipliers (B-2)
  → apply_liquidity_cap (vehicle ADV)
  → apply_gross_exposure_cap                ← NEW
  → AggressionPolicy.apply (default identity)
  → rebalance_band
  → PortfolioDecision(weights, rationale, vehicles, gross)
```

### 2.2 Regime → target multiple (MVP)

| Regime | `leverage_allowed` | target multiple |
| --- | --- | --- |
| STRONG_RISK_ON / RISK_ON | True | max available in family among `{+1,+2}` after ADV |
| NEUTRAL | True | +1 |
| RISK_OFF / STRONG_RISK_OFF | True | +1 (inverse는 `inverse_allowed is True`일 때만; UNKNOWN이면 금지) |
| any | None / False | +1 |

MVP는 **롱 레버리지(+1/+2)** 에 집중. 인버스는 `inverse_allowed is True`일 때만 `-1` 후보 (기본 대회 config는 UNKNOWN → 금지).

### 2.3 Caller wiring

| Caller | Change |
| --- | --- |
| `PortfolioPolicy.allocate` | kwargs: `regime`, `leverage_allowed`, `inverse_allowed`, `aggression_input`, `max_gross_exposure`; call ExposureSelector + gross cap |
| `BacktestEngine.run` | `allocate(scores, regime=..., leverage_allowed=rules.leverage_allowed, ...)` |
| `TournamentSimulator` / replay | 동일 kwargs 전달 |
| `_make_p08` | `InstrumentMaster` 주입 (panel 또는 factory); vehicle on |
| `cmd_decide` | master 로드, regime snapshot, rules → allocate; WHY에 `vehicle=` · `mult=` · `gross=` |

### 2.4 Fail-closed

- `master is None` → vehicle pass skip (identity), rationale에 `vehicle=identity`.
- family에 목표 배수 없음 → +1x 유지.
- ADV 재검사 실패 → +1x 강등; 그래도 실패 → weight 0 + 현금.
- aggression 예외 → multiplier 1.0.

---

## 3. Assumptions

- **A-1**: 점수 입력 ticker는 deployment 유니버스의 대표(대개 +1x). selector가 family 내 교체.
- **A-2**: 라이브/decide는 `tournament.yaml`의 UNKNOWN을 **+1x only**로 해석. 연구 harness만 `leverage_allowed=True|False` 시나리오를 명시 주입.
- **A-3**: Aggression은 대회 중 수동 rank/Δ 입력 전까지 off. 본 spec은 wiring + unit property만; B-4 E[Prize] 실측은 후속 results.
- **A-4**: M07 A-1 FAIL 유지 → alpha는 B1/P08. vehicle 게이트는 P08 대비 **동일 alpha + vehicle on/off**.
- **A-5**: `max_gross_exposure` 기본 1.60 (`configs/portfolio.yaml`).

---

## 4. Acceptance Gates

| Gate | Predicate |
| --- | --- |
| V-1 | `leverage_allowed=None` ⇒ selected multiples ⊆ `{1}` (unit) |
| V-2 | `confidence=LOW` ⇒ multiple==1 even if leverage_allowed=True (unit) |
| V-3 | after allocate, `gross_exposure(weights, multiples) ≤ max_gross` (unit) |
| V-4 | vehicle remap 후 liquidity cap이 **새 ticker** ADV를 사용 (unit) |
| V-5 | `AggressionPolicy(enabled=False).apply(*)==1.0`; enabled 시 INV-08-5 부호 유지 (unit) |
| V-6 | decide/replay WHY에 `vehicle=` 및 `mult=` 문자열 포함 (CLI) |
| V-7 | KRX gold path_dependent: P08+vehicle(RISK_ON, L=True) vs P08+1x — **CVaR(5%) ≥ P08_1x_CVaR − 0.03** 이면서, RISK_ON 창 subset에서 median 또는 P(R>30%) 비열위. 실패 시 대회는 +1x only 채택 |

---

## 5. Execution Target

```bash
uv run python tools/agent_skills/lean_check.py --spec docs/specs/09_vehicle_exposure_contract.json --pre-impl
# implement 후
uv run pytest tests/unit/portfolio/test_exposure.py tests/unit/portfolio/test_constraints.py tests/unit/portfolio/test_policy.py tests/unit/tournament/test_policy.py tests/unit/cli/test_decide.py -k "SCENARIO-09 or SCENARIO_09" -q
```
