# Contest Closed-Loop Re-analysis (2026-09-24)

Purpose: re-audit of `feedback.md`, `contest-strategy-probe-2026-09-24.md`, `contest-pure-strategy-audit-2026-09-24.md` with a corrected closed-loop harness. Machine-readable; P1 = P(rank 1) under the stated scenario, never a calibrated live probability.

## 1. Verdict

| id | finding | status |
|---|---|---|
| V1 | Objective P(rank1) (+P2 tie-break) is correct for the contest; no objective change needed | keep |
| V2 | Evaluation harness has defects that bias static rankings (see §2); live `decide_week` inherits D1, D3, D4 | fix required |
| V3 | Largest lever is the POLICY CLASS: daily state-feedback positional policy > weekly hold-to-end > static hold | adopt candidate |
| V4 | Static HY2 hold is best only if (rank-5 is NOT a persistent HY2 holder) AND (recent bullish semis drift continues) | conditional |
| V5 | Evaluation reform and strategy reform are inseparable: open-loop hold-to-end scoring cannot see state-feedback policies | do together |

## 2. Harness Defects (verified in code)

| id | defect | location | effect |
|---|---|---|---|
| D1 | drift neutralization demeans LOG returns per vehicle → removes volatility drag, incoherent for long/inverse pairs ((1+2r)(1-2r)<1) | `src/contest/panel.py:neutralize_drift`, scratch `Panel.neutralized` | inflates high-vol vehicles; HY2 P1 recent 9.1% (log0) vs 7.2% (arith0); winner bar median 0.73 vs 0.56 |
| D2 | missing legs zero-filled in history (K2/Q2/Q2I 2026-08-28,31; HY2/SS2 families 6 rows) | `build_vehicle_panel` (pre-contest rows), scratch `Panel` | minor; phantom flat days in bootstrap |
| D3 | our own leaderboard entry is assigned to a model agent (phantom self competitor) | `build_explicit_leaders` top_values includes nickname → `pin_to_leaderboard` | minor downward bias on all candidates |
| D4 | explicit leaders always churn (q=0.10); persistent-rival scenario absent | `append_explicit_leaders` + `leader_churn` config | if rank-5 holds HY2 persistently, HY2 hold P1 → 0.2-0.6% (hidden) |
| D5 | era mixture (2 raw + 2 neutral) implicitly assigns ~50% weight to raw bullish drift | `decision.eras` | directional view embedded, not declared |
| D6 | open-loop "hold candidate to end" scoring, weekly cadence, 5pp switch threshold | `decide_week` | cannot represent path-dependent positional play (§4) |
| D7 | holder inference uses KODEX HY2 only; TIGER 0195S0 differs 0.42pp median/day | `vehicles` config | misses HY2 holders on other wrappers |
| D8 | scratch forward probes started our equity at 1.0671 vs actual 1.0371 | scratch only | prior scratch P1 levels overstated; production rescales correctly |

Fix for D1 (coherent): demean ARITHMETIC close-to-close returns per vehicle (martingale). Linear → preserves long/inverse coherence. Directional scenarios: `cc' = cc - mean(cc) + beta_v * mu_HY2` (beta to HY2).

## 3. Corrected Harness (v2)

| item | value |
|---|---|
| code | `scratch/probe_v2.py` (lib + CLI), `probe_v2_grid.py`, `probe_v2_robust.py`, `probe_v2_tilt.py`, `probe_v2_summary.py` |
| crowd | 1,268 agents, f_auto 0.39, base/hot profiles; explicit top-11 leaders from 9/23 inference |
| scenario axes | era {recent 2024-10-23, long 2018-05-02} × drift {raw, arith0, tilt mu} × rank-5 {HY2, K2} × rival churn q {0, 0.1} × crowd {base, hot} × block {10, 20, 40} × worlds {bootstrap, real consecutive historical paths} |
| cost | 0.3% of equity per additional switch (P1_net30) |
| size | W=3000 per scenario (SE ≈ 0.7pp); 34-scenario grid + 10-point tilt scan; RSS ≤ 0.52GB |

## 4. Policies

| policy | rule (decide at open d using closes ≤ d-1) | observability |
|---|---|---|
| HY2 / HY2I / SS2I / BAT2 | static hold from 9/28 | - |
| MIRROR_VIS | best visible (top-50, weight ≥0.9) HY2 holder vs best HY2I holder (unseen side = 50th value); hold the OPPOSITE side of the higher one | realistic |
| TMC_e{n}_g{g} | every n sessions: P(win\|c) = mean_k 1[ours_c > max threats] over K=2000 martingale bootstrap paths; threats = best visible concentrated holder per vehicle + per-vehicle shadow at 50th value + best unknown entry as K2@0.5; switch if gain ≥ g | realistic |
| TMC_e33 | same, decided once (≈ open-loop static choice) | realistic |
| TMC_e5 | weekly re-decision (≈ current weekly engine cadence) | realistic |
| MIRROR_HY / AVOID5 | oracle variants (see all agents / rank-5 identity) | not tradable; upper bounds |

## 5. Results

### 5.1 Robust grid (34 scenarios, P1 net of 0.3%/switch)

| policy | mean | min | q25 | switches | P2 |
|---|---|---|---|---|---|
| MIRROR_VIS | 0.155 | 0.131 | 0.145 | 2.9 | 0.198 |
| TMC_e1_g0.02 (daily) | 0.152 | 0.081 | 0.117 | 2.1 | 0.198 |
| TMC_e5 (weekly) | 0.140 | 0.083 | 0.118 | 2.2 | 0.178 |
| HY2 static | 0.108 | 0.002 | 0.067 | 0 | 0.194 |
| TMC_e33 (one-shot) | 0.089 | 0.004 | 0.051 | 0 | 0.138 |
| HY2I static | 0.052 | 0.012 | 0.015 | 0 | 0.093 |

### 5.2 By rival identity × regime (mean P1 net)

| rank-5 | regime | HY2 static | MIRROR_VIS | TMC daily | TMC weekly |
|---|---|---|---|---|---|
| HY2 | bull (recent raw) | 0.124 | **0.140** | 0.089 | 0.095 |
| HY2 | other | 0.054 | 0.161 | **0.168** | 0.141 |
| K2 | bull (recent raw) | **0.204** | 0.158 | 0.122 | 0.133 |
| K2 | other | 0.100 | 0.156 | **0.187** | 0.167 |

### 5.3 Directional tilt (recent era, arith0 + beta·mu, rank-5 HY2, q=0.1)

| mu_HY2/day | P(HY2 fwd>0) | HY2 | HY2I | MIRROR_VIS | TMC daily |
|---|---|---|---|---|---|
| -0.008 | 0.22 | 0.033 | 0.187 | 0.143 | 0.231 |
| -0.004 | 0.31 | 0.050 | 0.139 | 0.163 | 0.217 |
| 0.000 | 0.40 | 0.067 | 0.096 | 0.173 | 0.189 |
| +0.004 | 0.50 | 0.088 | 0.064 | 0.177 | 0.159 |
| +0.008 | 0.60 | 0.115 | 0.034 | 0.170 | 0.116 |

### 5.4 Findings

- F1: Closed-loop daily positional policies dominate static holds except under (rank-5 not HY2) × (bullish raw drift).
- F2: MIRROR_VIS is the most robust (min 13.1%); TMC daily has the best mean outside the bull regime; neither needs a directional forecast.
- F3: Mechanism: a trailing player gains by holding the side anti-correlated to the nearest same-family leader, and re-choosing as leaders move. Static holds can't close a gap to a leader in the same vehicle.
- F4: Decision frequency matters: one-shot 8.9% → weekly 14.0% → daily 15.2% (TMC mean).
- F5: Visible-only information (top-50 + return-matched concentrated holders) retains most of the oracle value (MIRROR_VIS 17.9% vs oracle 20.9% in a base scenario).
- F6: Evidence on rank-5: 9/23 daily +2.714% vs KODEX HY2 +2.727% (99.5% weight) / TIGER HY2 +2.696%; ranks 7/8/17 = HY2I exact (-2.509%). Rank-5 as HY2 holder is likely, not proven.
- F7: MIRROR_VIS 9/28-open action from the 9/23 state: HY2 side best 1.1010 (rank 5) > HY2I side best 1.0780 (rank 7) → HY2I. If rank-5 is not HY2 → HY2.
- F8: Same-day ETF opens/closes (incl. 0193T0, 0195S0, 0197X0) are available from the Yahoo chart API used by `src/contest/reference.py` → daily decision can be produced the evening before (no KRX T+1 dependency).

## 6. Limits

| id | limit |
|---|---|
| L1 | Crowd/rival behavior is modeled; real rivals may switch intraday or hold mixes (inference misses them). |
| L2 | Policy variants were chosen after exploration (~10 variants); the gap to static (+4-10pp) is large vs SE 0.7pp but still model-conditional. |
| L3 | MIRROR ignores our own lead (no lock when leading); TMC includes lock implicitly. A lock-when-leading hybrid is untested. |
| L4 | Switch cost 0.3% assumed; mock-trading fees unknown. |
| L5 | SS2I fails the 10bn ADV gate; BAT2/other differentiated vehicles pass. |

## 7. Roadmap

| step | action | blocking for |
|---|---|---|
| R1 | fix D1 (arith demean), D3 (drop self from top_values), D4 (persistence scenario), D7 (wrapper set) in `src/contest` | any trust in weekly/daily cards |
| R2 | spec + implement daily card: Yahoo same-day closes → holder inference → MIRROR_VIS (primary) + TMC P(win) table (diagnostic); weekday ~16:50 KST timer; fail-closed to HOLD | daily policy |
| R3 | predeclared closed-loop evaluation grid (§3 axes) as a reusable harness; report mean/min/q25, not era averages | future decisions |
| R4 | test lock-when-leading hybrid (MIRROR behind / TMC ahead) out of sample | endgame |
