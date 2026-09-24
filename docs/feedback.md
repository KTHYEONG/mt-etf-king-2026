# Contest Strategy Feedback Log (2026 MT ETF Contest)

> SUPERSEDED IN PART (2026-09-24): §2-§3 static-hold conclusions and P1 levels are revised by `contest-closed-loop-reanalysis-2026-09-24.md` (harness defects D1-D8, closed-loop daily policies).

Purpose: machine-readable record of the live strategy, the evidence behind it, rejected alternatives, and open questions for future research. All P1 values = P(rank 1) from crowd-relative forward simulation unless noted.

## 1. Context (as of 2026-09-23)

| key | value |
|---|---|
| contest_window | 2026-09-21 .. 2026-11-13 (36 sessions; 9/24-9/27 Chuseok closed) |
| participants | 1,268 total (자율형 495); leaderboard JSON = top-50 only |
| our_nickname | lkthl |
| our_standing_2026-09-23 | rank 22, +3.714% (holding 122630 KODEX 레버리지 ~0.95) |
| top50_cutoff_2026-09-23 | +1.70% |
| inferred_leaders_2026-09-23 | ranks 3/4/6 → 494310 (SEMI2); 7/8/17 → 0197X0 (HY2I); 9 → 488080; rank 5 ambiguous (HY2 or K2/SEMI2 mix) |
| win_bar_2025 | winner +47.82%, 2nd +44.64% (≈ semis 2x buy-and-hold) |
| objective | maximize P(rank1), NOT P(R>30%) threshold (ADR-01 superseded by ADR-08 for live) |

## 2. Live Strategy

| item | value |
|---|---|
| action | 2026-09-28 09:00 open: sell 122630 → buy 0193T0 (SK하이닉스 2x) at ~100% |
| stop_loss | none (stops cut right tail; rank objective) |
| cadence | weekly re-decision (Sat 10:00 KST card, execute next session open) |
| candidates | HOLD, HY2, HY2I, SS2, SEMI2, K2, Q2, K2I (+ MIMIC_<leader> in last 8 sessions) |
| switch_rule | argmax mean P1; switch only if gain ≥ 0.05 |
| endgame | last 8 sessions: mimic strongest competitor's vehicle if leading |
| our_equity_source | leaderboard entry → else `--our-return` (manual) → else estimate from last visible entry + held vehicle (commit 5cd7ab2) |
| code | src/contest/*, src/cli/commands/contest.py, configs/contest.yaml, deploy/systemd/mt-etf-contest-* |
| entry_guard | edge valid while 0193T0 open < 12,000 × (our NAV/10억) |

## 3. Evidence: Strategy Comparison (forward sim from real 9/23 leaderboard)

Setup: 1,200-agent calibrated crowd, 4 eras {E1 recent raw, E2 recent drift-neutral, E3 long raw, E4 long drift-neutral}, block bootstrap mean 10, leadersB_churn (explicit top holders, 10%/day churn).

| strategy | P1_mean | P1_range | verdict |
|---|---|---|---|
| HY2 hold | 0.165 | 0.12-0.24 | ADOPTED |
| HY2 hold + late lock | +0.005 vs hold | - | adopted (endgame) |
| KOSPI20 direction HY2/HY2I weekly | 0.147 | 0.12-0.20 | rejected (lower mean) |
| HY2I hold | 0.097 | 0.02-0.16 | rejected (blocked by existing HY2I leaders) |
| KOSPI5 direction daily | 0.119 | 0.09-0.15 | rejected |
| own-trend direction daily (3d/10d) | 0.076-0.078 | 0.06-0.09 | rejected |
| momentum rotation daily (3/5d, long+short) | 0.055-0.062 | 0.05-0.08 | rejected |
| momentum rotation 10d daily/weekly | 0.026-0.028 | 0.02-0.03 | rejected |
| positional / rank-reactive weekly | 0.09-0.28 (replay) | - | no better than hold |
| max-vol weekly | 0.12-0.27 (replay) | - | no better than hold |
| champion (P27 incl.) | 0.01-0.05 | - | rejected |
| K2 hold (status quo) | ≤0.007 | - | rejected |
| SEMI2 / SS2 switch | ~0.0-0.02 | - | rejected (behind existing holders) |

Noisy-oracle upper bound (daily HY2/HY2I direction guessed with accuracy a):

| a | 0.50 | 0.52 | 0.55 | 0.60 |
|---|---|---|---|---|
| P1_mean | 0.159 | 0.201 | 0.277 | 0.426 |

Findings:
- F1: Edge is positional + max variance, not expected return. Random daily coin-flip ≈ hold.
- F2: All tested timing rules score below coin-flip (whipsaw + chasing vehicles already held by leaders).
- F3: Active trading beats hold only with ≥52-55% sustained daily direction accuracy; no such signal exists in repo.

## 4. Evidence: Failure Mode (Hynix falls)

| HY2 remaining return | share | crowd_median | winner_median | winner_holds_inverse | P1_HY2_hold | P1_HY2I_hold |
|---|---|---|---|---|---|---|
| ≤ -50% | 2-9% | -3% | +65~128% | 68-82% | 0 | 0.24-0.41 |
| -50~-20% | 13-23% | -1% | +39~74% | 46-78% | 0 | 0.09-0.45 |
| 0~+30% | 20-31% | +1% | +35~54% | 19-33% | 0.03-0.12 | ~0 |
| ≥ +80% | 5-25% | +4% | +116~143% | 2-5% | 0.62-0.66 | 0 |

- F4: HY2 ends ≤0 in 30-51% of worlds → P1 = 0 there (accepted cost; no strategy wins both directions).
- F5: Crowd median falls, but winner bar rises in crash worlds (inverse holders win).
- F6: After bad week 1 (HY2 -10~-25%): continue HY2 P1 0.09-0.20 > switch to HY2I/others 0.01-0.07. Comeback path = Hynix rebound, not late inverse.
- F7 (fixed): tool went blind outside top-50 → fixed with manual/estimated equity.

## 5. Model Limitations

| id | limitation | direction of bias |
|---|---|---|
| L1 | block bootstrap (mean 10d) weakens multi-week trends | understates trend-following |
| L2 | crowd model calibrated on 2025 + 2026 day-3 top-50 only | P1 level uncertain ±5-10pp |
| L3 | leader holdings inferred from daily return match | rank-5 = HY2 forever → HY2 P1 ~1% |
| L4 | pre-listing single-stock 2x returns are synthetic (listed 2026-05-27) | vol decay / tracking approximated |
| L5 | fees, tie-break rules unknown | small |
| L6 | multiple trials (~20+ strategies) without haircut | top differences (1-3pp) within noise |

## 6. Open Research Questions

| id | question | how to test |
|---|---|---|
| Q1 | Can any signal reach ≥52% daily HY2 direction accuracy OOS? | walk-forward on hynix daily; require perturbation-invariant PIT |
| Q2 | Do real leader holdings (HINT 보유종목베스트10 / 매매집중종목) change P1 ranking? | ingest if visible when logged in; replace inference |
| Q3 | Does weekly re-decision (myopic hold-to-end scoring) lose vs a true dynamic policy? | nested sim: policy re-evaluated inside worlds |
| Q4 | Sensitivity to longer-regime bootstrap (block 20-40, regime-conditioned) | rerun §3 with larger blocks |
| Q5 | Post-contest: did realized crowd match the model? | compare archived daily leaderboards vs sim quantiles |

## 7. Reproduction

| artifact | path |
|---|---|
| forward sim | scratch/probe_contest_forward.py `<ERA> base <W> 7 10 leadersB_churn` |
| failure bins | scratch/probe_contest_failure.py (same args) |
| active/oracle | scratch/probe_contest_active.py (same args) |
| week-1 playbook | scratch/probe_contest_playbook.py |
| replay (historical) | scratch/probe_contest_replay.py |
| leaderboard archive | data/contest/leaderboard/YYYYMMDD/ (or-vps) |
| weekly cards | results/contest_weekly/<session>.{json,md} (or-vps) |

Note: scratch/ is purgeable; copy probes before cleanup if re-research is planned. Memory budget: 6k worlds ≈ 0.5GB RSS; run eras sequentially.
