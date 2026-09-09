> ⚠️ **DEPRECATED (2026-09-09)**: 이 산출물의 `executable_oracle_p50`/`capture_50` 컬럼은 P27 자신의 스코어 후보(모델조건부)를 시장오라클로 오인해 계산된 결함이 있습니다 (INACTIVE 오라클=0.0, LOTTERY_ON capture=1.0으로 나타난 원인). 정정본은 [championship_regime_lift_20260909.md](./championship_regime_lift_20260909.md) 를 참조하세요. 본 파일은 감사 이력 보존을 위해 삭제하지 않고 유지합니다.

# Championship Regime Lift — Sleeve Diagnostic (2026-09-08)

**Analysis ID:** `championship_regime_lift_20260908`  
**P27 baseline run:** `20260907T232810Z_sticky.mom60_raw_20180102_20260827_0300_0500_0010`  
**Machine-readable:** [championship_regime_lift_20260908.json](./championship_regime_lift_20260908.json)

## Gate status

| Flag | Value |
|------|-------|
| `CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE` | `False` |
| `crash_rebound_abs_mom_bypass` (YAML) | not parseable; default `False` |
| P27 behavior with CRASH_REBOUND sleeve + bypass=True | still `CASH_INTENT` (no-op) |

## P27 tail vs sleeve (2,088 windows)

| Sleeve | n | P>50 | P>30 | Ruin<-25% |
|--------|---|------|------|-----------|
| LOTTERY_ON | 564 | **18.8%** | 25.4% | 5.3% |
| INACTIVE | 1,447 | 0.0% | 2.8% | 1.7% |
| CRASH_REBOUND | 17 | 0.0% | 0.0% | 0.0% |
| UNCERTAIN | 60 | 0.0% | 0.0% | 0.0% |

Overall P>50: **106 / 2088 = 5.08%**

## P>50 by year

| Year | Windows | P>50 hits | Rate |
|------|---------|-----------|------|
| 2018–2024 | 1,721 | 4 | ~0.2% |
| 2025 | 242 | 48 | 19.8% |
| 2026 YTD | 125 | 54 | 43.2% |

## 2026-08-27 probe vector

- Inputs: mom60=-22%, mom20=+23%, rv20_daily≈0.056 → rv_ann≈0.90
- Sleeve: **CRASH_REBOUND**
- `audit_regime_label`: **high_vol_reversal** (annualized rv required)

## Implementation check

- `lean_check --spec championship_regime_lift_contract.json`: **PASS** (diff-coverage 100%)
