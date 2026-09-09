# Win-Bar Candidate Ranking v2 (20260910_fixedrun, market_wide_candidate_pool 근본수정 반영)

**이전 산출물 대체:** `winbar_candidate_ranking_20260909.md` - 본 문서가 production 함수(`market_wide_session_candidates`, 커밋 1058453)와 100% 동일 경로로 재계산한 정본입니다.

레짐조건부 우승바(rank=5위, ratio=오라클x0.621) 초과율 기준 재랭킹. 절대임계 P>50/P>30, G2a 파멸위험(<-25%) 병기.

| Strategy | n | P>50(θ) | P>30(θ) | Ruin<-25%(G2a) | Capture@50(시장오라클) | WinBar rank 초과율 | n(rank) | WinBar ratio 초과율 | n(ratio) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sticky.mom60_raw(P27) | 2086 | 4.99% | 8.72% | 2.64% (OK) | 30.13% | 6.17% | 2026 | 7.75% | 2026 |
| sticky.p27_complement_switch | 2086 | 2.64% | 4.17% | 2.73% (OK) | 10.26% | 5.68% | 2026 | 6.07% | 2026 |
| baseline.mom20_top1(B1) | 2086 | 2.40% | 3.84% | 1.77% (OK) | 9.94% | 4.59% | 2026 | 4.79% | 2026 |
| baseline.regime_gated_theme | 2086 | 1.77% | 4.46% | 0.53% (OK) | 7.37% | 3.90% | 2026 | 4.05% | 2026 |
| baseline.mom20_ma_gate | 2086 | 1.39% | 2.64% | 1.63% (OK) | 9.29% | 3.50% | 2026 | 2.96% | 2026 |
| convex.lottery_impulse | 2086 | 4.17% | 5.61% | 0.81% (OK) | 27.24% | 2.52% | 2026 | 2.37% | 2026 |
| sticky.mom60_runner_reversal | 2086 | 3.55% | 6.09% | 0.96% (OK) | 20.51% | 2.47% | 2026 | 2.62% | 2026 |
| sticky.mom60_hold | 2086 | 2.44% | 3.69% | 1.01% (OK) | 13.46% | 1.68% | 2026 | 1.92% | 2026 |
| sticky.mom60_abs_cash | 2086 | 2.44% | 3.69% | 1.01% (OK) | 13.46% | 1.68% | 2026 | 1.92% | 2026 |
| baseline.buy_hold | 2086 | 0.48% | 2.54% | 0.43% (OK) | 3.21% | 0.00% | 2026 | 0.00% | 2026 |
