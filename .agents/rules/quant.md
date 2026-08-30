---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(etf|portfolio|allocation|rebalance|execution|data|backtest|alpha|feature).*"
  - on_file_path_glob: ["src/**/etf/**/*.py", "src/**/portfolio/**/*.py", "src/**/allocation/**/*.py", "src/**/rebalance/**/*.py", "src/**/data/**/*.py", "src/**/alpha/**/*.py", "src/**/backtest/**/*.py"]
priority: 10
---

# Tournament Quant & ETF Engineering Directives

This document defines quantitative and financial directives for the **36-session ETF Tournament Research & Trading System** (Money Today ETF King 2026).

## 0. Tournament Objective & Decision Philosophy
1. **Distribution Optimization Over Scalar Metrics:**
   - The primary objective is maximizing prize expectation $\mathbb{E}[\text{Prize}]$ by lifting right-tail outcome probability ($P(R_{36d} > 30\%\sim 40\%)$) while capping lower-tail catastrophe risk ($P(R_{36d} < -25\%) \le 5\%$).
   - Mean, Sharpe ratio, or long-term CAGR are secondary diagnostic metrics; they must never override 36-session return distribution criteria (G1~G5 gates).
2. **Strict Layer Separation (ARCH-1 / INV-24):**
   - Keep `Signal (Alpha)` $\to$ `Portfolio (Allocation/Sizing)` $\to$ `Tournament Policy (Regime/Overlay)` strictly decoupled.
   - Alpha models select theme/index keys; exposure selectors choose leverage vehicles (1X, 2X, -1X, -2X).
3. **Fail-Closed & Empirical Realism:**
   - Never extrapolate unverified rules. Treat missing/unknown rules as discrete scenarios (primary vs fallback).

## 1. Portfolio Sizing & Effective Exposure
- **Concentrated Momentum Allocation:**
   - Optimize for concentrated sizing (Top 1~3 holdings, max single weight $\le 80\%$, min cash buffer $\ge 5\%$) to capture explosive sector moves.
   - Dynamic exit on leadership deterioration (momentum score drop $\ge 30\%$, drawdowns $\le -15\%$).
- **Effective Gross Exposure Invariant (INV-18):**
   - Apply exposure constraints to **effective leverage exposure**, NOT raw weights:
     $$\text{Gross Exposure} = \sum |w_i \times \text{multiplier}_i| \le 1.60$$
   - Enforce single-holding constraint per underlying index family (INV-17): $\max(\text{count}(\text{family})) \le 1$.
- **Leverage & Inverse Realism (INV-14, INV-19):**
   - Use actual ETF price series for leverage/inverse backtesting; NEVER synthesize via index return $\times$ multiplier (preserves volatility drag).
   - If multiplier confidence is low or unverified, fail-closed to 1X exposure.

## 2. Universe & Eligibility Invariants
- **Sponsor Universe Boundary (INV-20, INV-21):**
   - Strategy adoption, ML models, and live execution must operate strictly on the **10 Tournament Sponsor ETF Universe** (`configs/sponsor_brands.yaml`).
   - Structural analysis across all KRX ETFs is permitted only for hypothesis generation.
- **Liquidity & Execution Capacity (INV-11):**
   - Ensure capital capacity (1.0B KRW initial capital) conforms to ADV limits:
     $$\text{Order Value} \le \text{ADV}_{20} \times \text{participation\_rate} \quad (\text{participation\_rate} \le 1\%\sim 2\%)$$
   - Discard illiquid candidates before calculating signals.

## 3. Microstructure & Point-in-Time Execution
- **Strict Timestamp & Execution Mechanics (INV-10):**
   - Signals generated at $close(t)$ MUST execute at $open(t+1)$ (next-day market opening). Same-bar look-ahead cheats are strictly prohibited.
- **Realistic Friction & Cost Drag (INV-12):**
   - Apply transaction taxes, broker commissions, and bid-ask / liquidity slippage directly at the fill event.
- **Point-in-Time (PIT) Data Integrity (INV-7, INV-8, INV-9):**
   - Use strict XKRX session calendar only (`core/calendar.py`). Avoid generic business-day generators (`bdate_range`).
   - Cross-sectional ranking, z-scores, and normalizations must strictly use cross-sectional data available at date $t$.

## 4. Deterministic & Safe Numerical Computation
- **Safe Vectorized Operations:**
   - Guard against zero-division and empty arrays using `np.divide(..., where=...)` with explicit zero-fill.
- **Distribution Stability & Non-Stationarity:**
   - Calculate rolling 36-session statistics using overlapping window corrections ($n_{\text{effective}}$).
   - Track giveback metrics (median & q90) and drawdown profiles for every strategy evaluation.
