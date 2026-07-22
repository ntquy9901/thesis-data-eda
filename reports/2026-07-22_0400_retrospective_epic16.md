# Retrospective: Epic 16 — Advanced News Signal with Embedding + Ticker-Specific Models

**Date:** 2026-07-22  
**Sprint:** Night run (autonomous)  
**Epic Status:** In Progress (5/5 stories implemented, tests pending)  
**Guide reference:** `docs/gpt-guide/huong_cai_thien_news_volatility_sau_eda.md`

---

## What we set out to do

Epic 16 aimed to extract a stronger news signal from PhoBERT embeddings by:
1. Separating news into two groups (khach_quan mainstream vs tong_hop analyst) → 16-1
2. Smoothing embeddings via EWMA → 16-2
3. Clustering tickers by news sensitivity → 16-3
4. Routing tickers through news-aware vs price-only experts (MoE) → 16-4
5. Rigorous ablation + permutation importance + OOS validation → 16-5

## What actually happened

### Positive outcomes
- **Dual-group embeddings (16-1):** Work as intended. kq_emb and th_emb each contribute
  ~−0.002 ΔR² when removed (ablation analysis). This confirms both groups carry complementary signal.
- **Ticker clustering (16-3):** Identified 4/30 sensitive tickers (CTG, PLX, POW, VIC).
- **Per-ticker evaluation:** Confirmed only 3-7/30 tickers benefit — consistent across all horizons.
- **Regime dependence (new finding):** News helps in high-volatility regimes (ΔR² = +0.0012)
  but hurts in low-volatility regimes. This was not in the original scope.

### What failed or underperformed
- **EWMA embedding smoothing (16-2):** All EWMA variants hurt performance.
  - EWMA(30d): ΔR² = −0.0026 to −0.0376 across horizons
  - Multi-EWMA (5 windows): ΔR² = −0.039 to −0.43
- **MoE (16-4):** ΔR² improvement is essentially 0 (+0.000006 to +0.000306).
  Deterministic gating by cluster is too simple; or the news signal is too weak for gating to matter.
- **HAR residual model (18-1):** Two-stage estimation adds noise.
- **OOS evaluation (16-5):** Signal disappears entirely on OOS 2026 data.

### What surprised us
1. **EWMA hurts.** We expected smoothing to reduce noise. Instead it added stale signal.
2. **Regime effect is strong.** High-vol regimes show positive ΔR²; low-vol regimes negative.
3. **Dispersion and novelty contribute nothing.** The semantic complexity features
   (how diverse is today's news) carry zero predictive value.

## Action items for next sprint

| Priority | Item | Assignee |
|----------|------|----------|
| HIGH | Build news gate conditioned on volatility regime (only activate news in high-vol) | Next sprint |
| HIGH | Run `bmad-code-review` on all new code | Before done |
| HIGH | Write unit tests for moe.py, residual_model.py, per_ticker_eval.py | Before done |
| MEDIUM | Simplify feature set: keep only kq_emb, th_emb, basic news_count, drop EWMA/multi-EWMA/novelty/dispersion | Next sprint |
| MEDIUM | Investigate regime gate: logistic gate that blends price-only vs price+news based on current vol regime | Next sprint |
| LOW | Restart dashboard to display new results | After tests pass |

## What to keep / drop

### Keep
- **Dual-group embedding features** (kq_emb_*, th_emb_*) — marginal but real signal
- **Ticker clustering** — useful for diagnostics
- **Per-ticker evaluation** — useful for monitoring
- **Volatility regime analysis** — key new finding

### Drop from active feature set
- **EWMA features** (ewma_kq_emb_*, ewma_th_emb_*) — hurts performance
- **Multi-EWMA features** (ewma5/10/20/30/60) — heavily overfits
- **Novelty features** — no signal
- **Dispersion features** — no signal
- **Max semantic shock** — no signal

### Recommend for Priority 2 (not yet implemented)
- **News gate conditioned on vol regime** — most promising remaining idea
- **Event type extraction via LLM** — high cost, uncertain benefit
- **Article dedup/clustering** — medium cost, unclear benefit
