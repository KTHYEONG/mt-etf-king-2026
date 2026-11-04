---
paths:
  - "src/**"
  - "tests/**"
  - "docs/specs/**"
---
# Quantitative Engineering Core Directives

> **Financial correctness, temporal causality, and execution realism without phantom alpha. Contest objective: maximize $P(M \ge W)$, the probability that the net 8-week multiple reaches the ex-ante prize line, using only edge validated against a zero-edge twin; mock capital carries no loss penalty.**

## 1. Temporal Causality
- **Point-in-Time Availability:** Every signal, feature, universe selection, and portfolio decision at $T$ uses only data observable at or before $T$.
- **Zero Lookahead:** Executing bar $T$ on signals from its own close, or referencing unreleased future data, is a fatal causality violation.
- **Perturbation Invariance:** Corrupting or randomizing future data ($t > T$) must leave decisions at or before $T$ unchanged.

## 2. Universe & Selection Integrity
- **Historical Universe Reconstruction:** Never judge historical windows by present-day survival criteria; the universe at $T$ includes later delisted, bankrupt, or suspended names.
- **Selection Bias Elimination:** Screens and dynamic universes ingest only information certified as known at rebalance time $T$.

## 3. Execution Friction & Capacity
- **Net Friction Integrity:** Never evaluate under zero-friction illusions; metrics include commissions, statutory taxes, bid-ask spreads, slippage, and financing/borrow costs.
- **Market Impact & Capacity Constraints:** Never assume infinite liquidity or instant fills; size within depth and ADV participation limits.
- **Zero Magic Numbers:** Thresholds, fee schedules, and filter cutoffs are typed `Spec`/`Config` contracts, never numeric literals in business logic.

## 4. Multiple Testing & Overfitting
- **Statistical Humility:** Never present trial-and-error metrics as independent discoveries without explicit haircuts for trial multiplicity.
- **Parameter Surface Robustness:** No knife-edge alpha; require stable plateaus across neighboring regimes.
- **Strict Out-of-Sample Isolation:** Keep discovery/training and evaluation temporally separated; never leak validation statistics into model formulation.

## 5. Integrity & Tail Modeling
- **Conservation Law:** Ledgers (cash, asset units, fees, taxes) reconcile exactly in integer base units (KRW/cents) or `Decimal`; float valuations use `rtol`/`atol` (`pytest.approx`), never exact equality.
- **Deterministic Fail-Closed:** On unrecoverable data anomalies or feed disruption, never substitute arbitrary defaults; abort to `NO_TRADE` (an operational safeguard against acting on bad data, not a risk preference).
- **Non-Gaussian Estimation:** Never assume Gaussian returns; estimate outcome distributions with fat tails, volatility regimes and regime shifts.
- **No Real-Money Preferences:** Do not add loss constraints, median/log-growth objectives, or drawdown breakers; never raise or cut exposure based on live rank or distance to the prize line.
