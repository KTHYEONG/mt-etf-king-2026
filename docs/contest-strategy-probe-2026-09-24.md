# Contest strategy probe: leader-relative daily monitoring

Research cutoff: 2026-09-23 close. The next contest session is 2026-09-28. This note evaluates a candidate policy; it does not change the live recommendation or execution configuration.

## Material correction to the prior feedback

The scratch forward simulation underlying `docs/feedback.md` modeled our first three sessions as a 95% KODEX leverage position from the first open. That path ends at equity **1.06708589**, while the archived leaderboard reports **1.03714170** for `lkthl`. The prior scratch P(rank 1) levels therefore had about **2.99 percentage points of excess starting equity**. The live `decide_week` implementation already rescales candidate paths to observed account equity; the issue is in the scratch research baseline. The probes below multiply our simulated equity by `1.03714170 / 1.06708589 = 0.97193835` at every comparison. Prior absolute P1 estimates, including 16.5% for holding HY2, should not be used as live probabilities.

## Current rival exposure

The archived 2026-09-23 leaderboard puts fifth place at equity **1.1010822** with a daily return of **+2.71436%**. KODEX SK Hynix 2x (0193T0) rose from 11,550 to 11,865 KRW, **+2.72727%**. A single KODEX fund explanation implies approximately **99.53%** weight. TIGER SK Hynix 2x (0195S0) also returned about **+2.70%** that day, within the return-match tolerance. The two wrappers' absolute daily market-return difference had a **0.42 percentage point median** over 80 paired local trading days, so matching only KODEX would often miss a TIGER holder. Current HY2-family exposure is plausible, not proven: another ETF or a mixed portfolio can produce a similar daily return. The fifth-place participant's cumulative +10.10822% exceeds HY2's first-three-session +1.28%, so they did not hold only HY2 from contest inception.

## Omitted static vehicles

The 2026-09-22 local ETF table has 243880 TIGER 200IT leverage among the most volatile liquid funds, with approximately 35.5 billion KRW in 20-session average trading value. It was absent from the live candidates. The static screen also included SS2I, BAT2, Q2I, and 50/50 splits. All candidates start from the actual account equity and use the same 2026-09-23 top-50 snapshot, four market eras, 6,000 worlds per era, mean bootstrap block 10, and a 10% daily leader churn profile. These are model scenarios, not independent out-of-sample observations.

| Future position | Mean P1 | Mean P(top 2) |
|---|---:|---:|
| HY2 | 12.76% | 20.61% |
| HY2I | 7.10% | 13.54% |
| SS2I | 5.23% | 8.98% |
| BAT2 | 4.30% | 7.51% |
| HY2 + IT2, equal capital | 2.70% | 6.58% |
| IT2 | 1.56% | 3.60% |

SS2I's recent trading value is below the live 10 billion KRW average-value gate. The IT2 panel has one missing 2026-08-28 ETF bar approximated from its underlying index; its result should be treated as a screen rather than an execution score. The omitted static positions did not beat HY2.

## Candidate daily policy

Track the *same named fifth-place participant* from the public daily top-50 table. Count 2026-09-23 as one HY2-family return match. At each later close, compare their published daily return with both KODEX HY2 at 99.5% weight and TIGER HY2 at 100% weight, within 0.02 percentage points. When at least two consecutive sessions match either wrapper, the participant remains visible and ahead of us, hold HY2I from the following open. Otherwise hold HY2. This uses only prior-close information. A third or fourth required match delays the first possible switch and reduces false positives.

The probe starts with HY2 at the 2026-09-28 open. It observes the 2026-09-28 closing leaderboard and can first switch at the 2026-09-29 open. The simulated policy changes vehicle about twice per world after initial entry. Costs below are hypothetical 0.2% or 0.5% of account equity for each additional switch; both sides of a switch, spread, and impact are approximated by that single charge.

| Fifth-place behavior | Policy | P1 before extra cost | P1 at 0.2%/switch | P1 at 0.5%/switch |
|---|---|---:|---:|---:|
| HY2 now, 10% daily churn; base, block 10 | HY2 hold | 10.16% | 10.16% | 10.16% |
| Same | Two return matches, HY2/HY2I | 17.21% | 16.86% | 16.22% |
| Same | Three return matches, HY2/HY2I | 17.07% | 16.60% | 16.06% |
| Same | Four return matches, HY2/HY2I | 16.84% | 16.58% | 16.18% |
| Fifth place holds TIGER HY2, 10% daily churn | HY2 hold | 10.13% | 10.13% | 10.13% |
| Same | Two return matches against both wrappers | 17.08% | 16.72% | 16.08% |
| Fifth place holds K2 initially, 10% daily churn | HY2 hold | 13.03% | 13.03% | 13.03% |
| Same | Two return matches against both wrappers | 13.84% | 13.81% | 13.78% |

Each row averages four eras with 5,000 worlds per era. Fifth-place weight is 99.5% in the KODEX and K2 paths and 100% in the TIGER path. The TIGER leader path uses its listed market returns from 2026-05-28 onward, with a HY2 proxy before listing and on missing-data days. As additional market/crowd stresses for the KODEX HY2-initial profile, block 20 yielded **10.43% hold vs 15.84% policy after 0.2% switch cost**; a more aggressive crowd yielded **8.88% vs 14.28%**. If fifth place holds KODEX HY2 permanently at 100% weight, HY2 hold falls to **0.27% P1**, while the observable two-match policy reaches **22.61% before cost**. This extreme case illustrates the positional mechanism and is not a probability forecast.

P(top 2) also rises in the KODEX HY2-initial churn scenario: **19.10% hold vs 22.74%** for the two-match policy before additional costs. Official contest prizes reward overall first and second place, with a separate category prize under the published rules.

## Interpretation and limits

The policy has a clear causal reason to help: when a rival with a 6.4 percentage point head start holds our exact vehicle, remaining in that vehicle cannot close the gap. An inverse position can pass that rival during a decline; returning to HY2 after the rival falls behind can capture a rebound. Weekly hold-to-end scoring cannot represent this path-dependent action.

The numerical gain remains conditional on a synthetic rival and crowd model. The public table shows only 50 participants and no individual holdings. Daily return matching can misidentify a mixed portfolio; ETF premiums, tracking error, cash weights, and execution at the next open can break a match. The 2026-09-23 TIGER quote was checked against public market pages rather than the local KRX silver table, which ends on 2026-09-22. The historical market eras overlap, the rule was chosen after exploratory trials, and transaction costs were stressed rather than measured. The existing scratch crowd pin also leaves a modeled copy of our own leaderboard entry among competitors. These limitations prevent treating the reported P1 values as calibrated live odds.

The research priority is to confirm the fifth-place holder from an authorized HTS holdings view or repeated return matches, then evaluate this rule against archived daily leaderboards and actual next-open fill data. Until that evidence is available, the result supports **daily monitoring as a promising candidate**, while the 2026-09-28 initial HY2 choice remains the strongest tested static action.

Reproduction uses `scratch/probe_contest_newscope.py` and `scratch/probe_contest_leader_monitor.py`, backed by `scratch/etfRankTotal_20260923.json` and `scratch/contest_panel.npz`. The `scratch/` directory is purgeable.
