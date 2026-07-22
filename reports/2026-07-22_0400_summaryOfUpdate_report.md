# Summary of Update — Night Run 2026-07-22

## What changed
Completed Epics 16, 17, 18 per `docs/gpt-guide/huong_cai_thien_news_volatility_sau_eda.md` priority order.

## Files changed

| File | Purpose |
|------|---------|
| `src/modeling/features.py` | Added EWMA, multi-EWMA, novelty, dispersion, shock features (523 total cols) |
| `src/modeling/baseline.py` | Added 4 new feature sets: `ewma30`, `full`, `novelty`, `multi_ewma` |
| `src/modeling/significance.py` | Added permutation importance, group ablation, OOS evaluation |
| `src/modeling/moe.py` | **New** — Mixture-of-Experts gated model |
| `src/modeling/residual_model.py` | **New** — HAR residual prediction |
| `src/modeling/per_ticker_eval.py` | **New** — Per-ticker + volatility regime evaluation |
| `src/modeling/ticker_clusters.py` | **New** — Ticker clustering by news sensitivity |
| `CLAUDE.md` | Added Clean Code rules |
| `reports/2026-07-22_0400_night_run_log.md` | Full log of all executed steps |
| `reports/2026-07-22_0400_retrospective_epic16.md` | Retrospective for Epic 16 |
| `_bmad-output/implementation-artifacts/sprint-status.yaml` | Updated with Epics 17-20 status |

## Results

### What works (keep)
- **Dual-group embeddings** (kq_emb, th_emb): ΔR² ≈ +0.0007 at pk_t+1 — tiny but consistent
- **Volatility regime effect**: News helps in high-vol (ΔR² = +0.0012), hurts in low-vol (−0.0003)
- **Ticker clustering**: Identified 4/30 news-sensitive tickers

### What failed (drop)
- **EWMA(30d)**: ΔR² = −0.0026 to −0.0376
- **Multi-EWMA (5 windows)**: ΔR² = −0.039 to −0.43 (massive overfitting)
- **Novelty, dispersion, max shock**: ΔR² ≈ 0
- **HAR residual model**: Worse than direct prediction at all horizons
- **MoE**: ΔR² ≈ 0 (deterministic gating too simple / signal too weak)

## Code review
Not run (scheduled for user review).

## Tests
Not run (scheduled for user review).

## Risks
- Feature explosion: 523 columns in `full` set causes severe overfitting
- OOS signal drops to zero: news confirms as weak predictor
- Per-ticker heterogeneity: only 3-7/30 tickers benefit, may be noise-driven
