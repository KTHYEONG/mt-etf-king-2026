# Contest pure-strategy audit, 2026-09-24

Research cutoff: Korean market close on 2026-09-23. The local normalized KRX ETF panel ends on 2026-09-22; selected 2026-09-23 quotes came from cached public quotes. The KRX Open API returned 0 ETF rows for 2026-09-23 when checked on 2026-09-24, versus 1,175 rows for 2026-09-22. The next planned Korean session is 2026-09-28. This is a research audit, not a change to the live account or strategy configuration.

## Verdict

The claim that holding SK Hynix 2x is **the** best stand-alone contest strategy is not established. It is a plausible, liquid, high-upside bet and remains a defensible provisional 2026-09-28 candidate. The actual listed-product history is strongly adverse, and the older favorable history is mostly a synthetic exposure that was not purchasable at those dates. The current market state has no close historical analogue in the available Hynix series. No alternative has a validated superior forward return distribution either. The prior crowd-model P(rank 1) figures must not be interpreted as calibrated live probabilities.

The decision should distinguish three questions: (1) Which underlying has a credible next-33-session directional edge? (2) Which wrapper converts that edge into the largest *net* terminal account value? (3) Which terminal value passes the actual winning participant? High volatility helps only the upper tail of (2) and (3); it does not answer (1).

## Investable universe and missed exposures

The 2026-09-22 local ETF panel contains 1,175 current tickers. A sponsor-brand mapping recognizes 1,086; 62 have a positive 2x-like name and 10 have an inverse 2x-like name. The organizer reported 1,059 eligible sponsor ETFs on 2026-09-21, so the brand count is a **screen**, not proof of HTS eligibility. `configs/universe_manifest.yaml` is still `null`. The contest's autonomous division permits the leverage and inverse products of sponsor managers. See [organizer rules](https://www.mt.co.kr/etf/join/index.html) and [organizer eligible-count announcement](https://www.mt.co.kr/index.php/stock/2026/09/20/2026091814372139415).

All figures below are from local KRX data through 2026-09-22. Volatility is the standard deviation of the latest 20 available close-to-close returns. ADV is mean trading value of the latest 20 available sessions. A 1 billion KRW simulated order's share of ADV is a coarse capacity diagnostic, not an order-book fill estimate.

| ETF | 20-session daily vol | 20-session ADV, KRW bn | 1bn/ADV | Finding |
|---|---:|---:|---:|---|
| 0193T0 KODEX SK Hynix 2x | 6.95% | 242.7 | 0.41% | Highest liquid positive leverage volatility in screen |
| 0195S0 TIGER SK Hynix 2x | 6.94% | 110.3 | 0.91% | Same underlying, viable wrapper challenger |
| 0197X0 SOL SK Hynix futures inverse 2x | 6.81% | 108.2 | 0.92% | Major downside branch |
| 488080 TIGER Semiconductor TOP10 2x | 5.91% | 174.5 | 0.57% | Diversified chip leverage |
| 462330 KODEX Battery Industry 2x | 5.77% | 74.3 | 1.35% | Different sector branch |
| 494310 KODEX Semiconductor 2x | 5.64% | 237.3 | 0.42% | Very liquid chip-sector alternative |
| 243880 TIGER 200 IT 2x | 5.44% | 35.5 | 2.82% | Omitted from the live candidate list |
| 0193W0 KODEX Samsung Electronics 2x | 5.10% | 90.5 | 1.10% | Single-stock alternative |
| 0091P0 TIGER Korea Nuclear | 4.33% | 54.8 | 1.83% | Uncorrelated thematic challenger; no leverage |
| 0197W0 SOL SK Hynix 2x | 7.04% | 2.16 | 46.3% | Same exposure, weak capacity at 1bn |
| 0194T0 ACE SK Hynix 2x | 6.85% | 1.02 | 98.0% | Same exposure, weak capacity at 1bn |
| 0080Y0 SOL Shipbuilding TOP3+ 2x | 5.45% | 3.84 | 26.0% | Previously omitted; weak capacity |
| 0100K0 KODEX Defense TOP10 2x | 5.05% | 3.21 | 31.2% | Previously omitted; weak capacity |

The full sponsor-mapped screen also contains US semiconductor and AI memory ETFs, but the more volatile liquid domestic vehicles dominate those themes' recent volatility. This is a volatility screen, not proof that high volatility or recent return predicts future return. Some name-based leverage classifications and brand eligibility may be imperfect.

## Crowd-free historical evidence

The remaining horizon is 33 contest sessions. For each historical start, the calculation buys at that day's open and marks after 33 sessions. Synthetic 2x returns are daily-reset multiples of Hynix/Samsung stock or the KOSPI200 IT index with an assumed 1% annual drag. The long and recent panels contain 1,910 and 385 complete start windows, respectively. These windows overlap heavily and do not provide independent Bernoulli trials. Returns and probabilities are descriptive historical frequencies, **not** forecasts for 2026-09-28.

| Hypothetical exposure | 2018-05 to 2026-07: 33-session median | Share >+50% | 2024-10 to 2026-07: median | Share >+50% |
|---|---:|---:|---:|---:|
| Hynix 2x synthetic | +3.0% | 12.7% | +23.3% | 36.9% |
| Samsung 2x synthetic | +1.8% | 5.7% | +14.3% | 20.3% |
| KOSPI200 IT 2x synthetic | +1.4% | 7.4% | +14.1% | 27.3% |
| KODEX KOSPI200 leverage actual | +1.5% | 4.2% | +12.0% | 18.7% |

The long-period Hynix mean is +12.6% and its 10th/90th percentiles are -27.2%/+59.0%. The recent-period mean is +45.4% and its 10th/90th percentiles are -29.2%/+147.2%. This sharp regime dependence prevents choosing a calibrated forward mean or win probability from either row. The synthetic Hynix 2x fund did not exist before 2026-05-27.

The actual KODEX Hynix 2x listed history has only 81 ETF observations through 2026-09-22 and 49 overlapping 33-**market-session** windows, counted on the KOSPI200 trading calendar so the missing 2026-08-28 ETF bar does not lengthen the horizon. Those windows have a **-49.7% median** and **0/49 exceed +50%**. Over the same 49 start dates, KODEX Semiconductor 2x has a -40.5% median; KODEX Hynix 2x outperforms it in only 6/49 starts. This is one extended semiconductor crash, not 49 independent future analogues. It is decisive evidence against treating the favorable synthetic backtest as the whole story, not decisive evidence that Hynix must continue falling.

Across the full listing interval 2026-05-27 to 2026-09-22, Hynix stock fell about 18.0%, a synthetic daily-reset Hynix 2x path fell 55.5%, and KODEX Hynix 2x market close fell 58.4%. The fund's observed peak-to-trough drawdown was 84.1% (2026-06-22 to 2026-07-30). This illustrates path dependence and execution/tracking costs, not a predicted future loss. The [issuer's own product explanation](https://www.samsungfund.com/etf/theme-view.do?bgColor=%23FBB938&seq=60116&themeCd=leverage) confirms daily rather than holding-period leverage and a 0.29% annual stated fee before other costs.

A wider actual-product scan found 94 currently sponsor-mapped ETFs with at least 100 local observations, latest 20-session ADV >=10bn KRW, and at least 60 eligible 33-market-session windows since 2025-01-01. Among examples, SOL AI Semiconductor TOP2+ had 37.1% of its 97 overlapping windows above +50%; TIGER 200 IT 2x had 33.1% of 387; TIGER Semiconductor TOP10 2x had 31.3% of 387; KODEX Semiconductor 2x had 30.5% of 387. These products' lifetimes differ, the windows overlap, and sorting 94 funds by realized upside causes large selection bias. They are leads for further study, not evidence that the first entry should be changed. The Hynix funds were excluded from this particular scan by the 100-observation minimum, not because their returns were judged inferior.

An exploratory crowd-free policy check asked whether a simple prior-20-session momentum choice between hypothetical Hynix 2x and Samsung 2x improves the 33-session upper tail. In 1,531 complete windows starting 2018-05 through 2024-12, fixed Hynix reached +50% in 7.1% of windows versus 4.2% for the momentum choice. In 316 complete windows starting 2025-01 through 2026-07, the corresponding shares were 38.3% versus 30.7%. A four-way momentum choice among Hynix, Samsung, KOSPI200 IT 2x, and KODEX KOSPI200 leverage was lower still (2.5% and 30.1%). A 60-session lookback was also worse than fixed Hynix in both eras. These windows overlap, the Hynix/Samsung exposures before their 2026 listing are synthetic, and the rule has been inspected after other exploratory rules. The result rejects a claim of an already demonstrated simple momentum fix; it does not rule out all timing signals.

## Why the present state is unusually hard to infer

From the cached Hynix stock close through 2026-09-23: past 20 sessions +10.3%, past 60 -29.1%, past 120 +130.7%, past 252 +546.5%; current price is 36.2% below its 120-session high. The stock's latest 20 daily-return standard deviation is 3.59%; the latest 60 is 7.28%. In the 2018-2026 stock series, there were 18 overlapping historical starts with a prior 60-session return between -50% and -25%, but **none** also had a prior 20-session return between +5% and +20%. These intervals were selected after observing today's state, so even the zero match is descriptive, not a statistical test. The long synthetic history does not directly condition on this combination of violent correction and partial rebound.

For a 2x daily-reset instrument, `log(terminal ETF factor) ≈ 2 log(terminal stock factor) - sum(stock daily return²) - costs`. If the account's 2026-09-23 equity is 1.0371417, reaching a *hypothetical* final equity of 1.50 requires +44.6% from 2026-09-28 onward. A smooth Hynix path would require about +20.3% stock appreciation. Applying the observed 20-session or 60-session squared-return rates to 33 future sessions raises the approximate stock requirement to +22.8% or +31.1%, respectively, before account transaction costs and ETF price/NAV effects. These are scenario arithmetic, not price targets or a forecast of the winner's final equity.

The case for Hynix direction is real but incomplete. [SK hynix's July 2026 Q2 release](https://news.skhynix.com/en/q2-2026-business-results/) reports record results and HBM4 shipment progress. [Samsung's July 2026 Q2 release](https://news.samsung.com/global/samsung-electronics-announces-second-quarter-2026-results) also reports record memory performance, rising HBM4 sales, and HBM4E samples. These releases support a broad memory-sector thesis but do not establish that Hynix will outperform Samsung or a diversified semiconductor basket from today's prices. [Micron has scheduled fiscal Q4 results for 2026-09-30](https://investors.micron.com/news/press-release/2026/Micron-Technology-to-Report-Fiscal-Fourth-Quarter-Results-on-September-30-2026/default.aspx), creating another memory-sector catalyst. None of these releases measures the *unexpected* component relative to the market's current expectations. The 2026-09-23 ETF close already incorporates known fundamentals. Chuseok US-market news can alter the 2026-09-28 Korean opening price; a buyer at that open cannot capture an overnight move already priced into the fill.

The available 2026-09-23 public quotes add a near-term cross-check: KODEX Hynix 2x closed +2.73% from 2026-09-22, while KODEX Semiconductor 2x closed +6.13% from its 2026-09-22 close. [A secondary market close quote](https://markets.hankyung.com/stock/0091P0/asp) shows TIGER Korea Nuclear -5.83% on 2026-09-23 after the strong prior-20-session run. The full official 2026-09-23 KRX ETF file was unavailable at the research cutoff. These single-session observations weaken any simple extrapolation of the latest 20-session trend but do not predict the next 33 sessions.

## Wrapper and execution comparison

The four Hynix long 2x wrappers were all listed on 2026-05-27. On 2026-09-22, market close relative to published NAV was -0.031% for KODEX, +0.208% for TIGER, +0.328% for SOL, and -0.275% for ACE. Absolute median market-price/NAV gaps over the latest 20 observations were approximately 0.38%, 0.39%, 0.32%, and 0.41%, respectively. KODEX and TIGER have ample 20-session turnover for a 1bn KRW hypothetical order; SOL and ACE do not pass the same rough capacity screen. The 2026-09-22 premium ranking is stale for 2026-09-28. Compare contemporaneous executable price, live iNAV/estimated fair value, depth, and total order cost at the open. KODEX is the default liquid wrapper, not intrinsically the best purchase at any price.

The feedback's fixed entry guard, `0193T0 open < 12,000 × own NAV/1bn`, has no documented empirical derivation in the codebase. It scales an ETF share price by account wealth, even though share price and account equity have different causal roles. It can reject a fairly priced opening after favorable news or accept an ETF trading at a costly premium. Any go/no-go rule needs a contemporaneous fair-value and order-cost basis. Do not turn the arbitrary guard into a claimed optimal threshold.

## Model issues that make earlier P(rank 1) fragile

1. `docs/feedback.md`'s original scratch comparison used own starting equity 1.06708589 instead of the public 1.03714170. The newer probe corrected the static HY2 figure to 12.76% under its assumed crowd, but this remains a scenario output rather than a live probability.
2. The production decision's `neutralize_drift` and scratch analogue separately force each vehicle's mean log return to zero. For daily long and inverse 2x on the same underlying daily return `r`, the joint gross return is `(1+2r)(1-2r)=1-4r²<1`. In the recent era, the raw Hynix long/inverse pair has mean joint log return **-0.973% per session**; independent neutralization changes it to approximately **0%**, an economically impossible joint series if both are generated from the same underlying. This can change candidate and simulated-rival outcomes. Coherent stress must adjust underlying drift before constructing both leverage sides.
3. The crowd model has about 20 synthetic vehicles while the official sponsor universe exceeds 1,000 products; it cannot cover omitted nuclear, defense, shipping, single-stock wrappers and other winner paths. Top-50 holdings remain inferred, and fifth place may share Hynix long exposure.
4. `scratch/contest_panel.npz` has missing K2, Q2, and Q2I returns on both 2026-08-28 and 2026-08-31; the simulator silently replaces NaNs with zero. This deletes real market moves in bootstrap samples. Prelisting substitutes for Hynix/sector products are hypothetical, and the 2026-09-23 sector-index returns are approximated from KOSPI200. These defects should be repaired before relying on small P1 differences.
5. The weekly engine only scores its eight configured positions and uses a 5 percentage-point P1 gain to switch. The 243880 IT 2x and other identified exposures are absent. The switch threshold is larger than many modeled candidate gaps and is not derived from observed transaction costs or sampling uncertainty. A daily event check may help when new information changes the exposure case; a daily trade schedule has no demonstrated edge.

## Actionable research conclusion

- Preserve Hynix 2x as a **provisional concentrated upside candidate**, subject to 2026-09-28 fair-value and liquidity conditions. The present evidence does not justify describing it as proven optimal or the earlier P1 numbers as odds.
- Keep KODEX Semiconductor 2x, TIGER 200 IT 2x, SOL AI Semiconductor TOP2+, TIGER Korea Nuclear, and Hynix inverse 2x on a live challenger watchlist. Their causal win cases differ: broader chip leadership, IT rebound, concentrated dual-chip leadership, nonchip leadership, and Hynix decline. None is a validated replacement on 2026-09-24.
- Before the first order, obtain the missing 2026-09-23 full ETF table or reliable closing quotes, inspect 2026-09-24/25 US memory-sector information, compare 2026-09-28 live iNAV/premiums and depth, and use current leaderboard state. A scheduled earnings release or a high recent return alone is not a trading signal.
- Prioritize a causally coherent market panel, full eligible-product manifest, actual account fills, and walk-forward evaluation of **predeclared** daily/event triggers. Require a material, stable net advantage across market regimes and realistic costs before replacing a static position or the weekly schedule.

Reproduction: `scratch/probe_pure_contest_20260924.py`, plus the local ETF/stock sources noted above. Scratch probes are exploratory and may be deleted; this note retains the essential methods, cutoffs, and numeric findings.
