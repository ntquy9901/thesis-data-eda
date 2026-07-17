# Research: Data-Mining/NLP Methods for News→Volatility (Next Steps)

**Date:** 2026-07-15
**Scope:** Methods NOT yet applied in this repo. Builds on existing pipeline: rule-based topic flags
(`phase04_news_eda.py`), Granger/cross-corr/event study (`phase05/06`), and the new PhoBERT `[CLS]`
embedding + PCA cache (`src/features/news_embeddings.py`, `src/nlp/embeddings.py`).
**Known result to design against:** effect is weak, T+10-only, ~25% of tickers (Epic 9). Any new
method should either (a) explain *which* tickers/news are the sensitive 25%, or (b) sharpen the
weak long-horizon signal — not just add more features to an already-null short-horizon regression.

---

## 1. Topic modeling: BERTopic / embedding-clustering vs. current keyword flags

**What it is:** BERTopic clusters document embeddings (e.g. UMAP + HDBSCAN) then labels clusters via
class-based TF-IDF, replacing LDA's bag-of-words generative model with semantic clustering.

**Fit here:** Mixed evidence — no Vietnamese-financial-specific benchmark exists; general studies
split (BERTopic wins on marketing/Chinese-English corpora, LDA wins on Belgian-Dutch coherence
scores; a Vietnamese *legal*-text paper found neither alone beats an ensemble). More importantly:
**you already have PhoBERT embeddings cached per article** — BERTopic is almost free to try because
step 1 (embedding) is done. It would replace the current binary keyword `topic_*_count` flags (which
can't detect topics the keyword dictionary didn't anticipate) with data-driven clusters. Honest caveat:
your topic flags aren't the bottleneck — the bottleneck is small effect size — so this improves
*interpretability of which topics matter* more than it improves raw predictive power.

**Effort:** New phase module (`phase11_topic_clustering.py` or similar), reusing the cached
`raw_*` embedding columns from `news_emb_articles_{group}.parquet` directly (skip re-embedding).
Needs new library (`bertopic`, plus its `umap-learn`/`hdbscan` deps — moderately heavy). ~1 story.

---

## 2. NER / structured event extraction (beyond keyword topic flags)

**What it is:** Extract structured tuples (entity, event-type, direction) from text instead of a
single topic flag, e.g. `(FPT, earnings_beat, positive)` vs `(FPT, dividend_cut, negative)`. Recent
literature (EFSA, 2024) argues event-level signal dominates entity-level or plain sentiment because
the same entity can carry opposite sentiment depending on the *event type* (price move vs. profit
forecast).

**Fit here:** Directionally the most likely lever for the "why only 25% of tickers" question — if the
sensitive tickers are the ones with more *earnings/M&A/regulatory* events vs. routine market
commentary, an event-typed feature could isolate that signal that a flat topic-flag or embedding
average smears out. Vietnamese financial NER models are scarce (no off-the-shelf PhoBERT-NER-finance);
realistic path is zero/few-shot LLM extraction (prompt an LLM to output event type + polarity per
article) rather than training a tagger — feasible at this data volume (thousands, not millions, of
articles) but adds an LLM-API dependency your other phases don't have.

**Effort:** New phase module + new dependency (LLM API call or a pretrained multilingual NER en/vi
model like `xlm-roberta` NER fine-tune). Medium-large: bigger than a features tweak, more like Epic 8
scale (new feature category + revalidate against target).

---

## 3. Causal/event-study refinements

**What it is:** Sharper event-study design — e.g. matched-control synthetic control per ticker,
staggered-adoption-robust DiD (if grouping tickers by "news-sensitive" cohort), or a
regression-discontinuity around scheduled events (earnings dates) vs. unscheduled news.

**Fit here:** Your Epic 9 event-level t-test came back null (p 0.27–0.86) with a symmetric
CAR-vs-non-CAR-day comparison. Standard event-study refinements (synthetic control, matched-firm
benchmarks) mostly correct for confounding in *cross-sectional* settings with many comparable firms —
here you have 30 tickers and a single market, so the marginal gain is likely small. A more promising
refinement given your existing null: separate scheduled (earnings/dividend calendar — knowable in
advance) vs. unscheduled news, since scheduled-event "surprises" are the theoretically cleaner test
of information content; unscheduled-news noise may be diluting the pooled t-test.

**Fit verdict:** Marginal — you already tested the obvious version and got a clean null; a fancier
econometric wrapper is unlikely to flip that unless paired with the scheduled/unscheduled split above.

**Effort:** Extends `phase06_event_study.py` — needs an earnings-calendar data source you don't
currently have (not in your two mandatory data dirs) — likely infeasible without new data acquisition,
which conflicts with the "no alternative data sources" project rule. **Low priority.**

---

## 4. Text-based volatility/uncertainty index (Baker-Bloom-Davis style)

**What it is:** BBD's EPU index is a simple frequency count: normalized monthly count of articles
containing an (Economy ∧ Policy ∧ Uncertainty) keyword triple, validated against human reading and
correlated with VIX (r=0.73 for their equity-uncertainty variant).

**Fit here:** This is *directly* transplantable and cheap — build a Vietnamese equivalent
(`kinh tế`/`chính sách`/`bất định|không chắc chắn`-style keyword triple, or reuse your existing
Vietnamese keyword-topic infrastructure from phase04) as a **daily aggregate uncertainty index**, then
correlate it with realized/Parkinson volatility the same way you already test topic flags. This is
different in kind from your current per-article features: it's a *market-wide* daily index, not a
per-ticker news feature, so it plugs into the T+10 finding as an additional macro control or
alternative target-side regressor. Given your existing Granger/cross-corr machinery
(`phase05_relationship.py`), testing this needs no new statistical infra — just a new feature
column. Low-risk, genuinely novel angle (not yet tried), directly answers "does more RSS/news chatter
mean more upcoming volatility" independent of sentiment/direction.

**Effort:** Small — extends `phase04_news_eda.py`/`phase05_relationship.py` with one new daily
aggregate series. No new library. Reuses existing keyword-matching code pattern already in the repo.
**Cheapest high-value item on this list.**

---

## 5. Embedding-based novelty/staleness detection

**What it is:** For each article about ticker X, compute cosine similarity to prior articles about X
(e.g. past 5 trading days); classify as "fresh" (low max-similarity) vs. "stale" (high max-similarity,
i.e. rehashed news). Tetlock and follow-up work find stale news gets weaker/faster price response than
fresh news, and pure republication inflates naive news-volume counts without adding information.

**Fit here:** Strong conceptual fit — you already have the embeddings needed (PhoBERT `raw_*` cached
per article, keyed by url) so novelty = pairwise cosine similarity within a rolling per-ticker window,
computable directly from the existing cache with no new encoding cost. This could plausibly explain
part of the weak/heterogeneous signal: if the sensitive 25% of tickers are the ones getting more
*fresh* (vs. republished/aggregated) coverage, a novelty-weighted feature could sharpen the T+10 result
where the raw embedding average did not. This is the best "reuse what you built" opportunity on this
list.

**Effort:** Small-medium — new function in `src/features/news_embeddings.py` (or a sibling module)
computing rolling cosine-similarity-to-past-articles per ticker from the existing article cache, then
feed as a feature into `src/modeling/features.py`. No new library (cosine sim is `numpy`/`sklearn`
one-liner).

---

## 6. Cross-lingual/multilingual embedding alternatives to PhoBERT

**What it is:** Swap/augment PhoBERT `[CLS]` with multilingual sentence-embedding models
(`multilingual-e5`, `LaBSE`, `gte-multilingual`) tuned for semantic similarity/retrieval rather than
PhoBERT's MLM-pretraining objective (which produces token representations not optimized for
sentence-level similarity).

**Fit here:** The 2025 VN-MTEB benchmark (arXiv 2507.21500) directly measured this and found
**multilingual-e5-base (62.4) and gte-multilingual-base (65.2) clearly beat PhoBERT-based sentence
embeddings (50.6)** on Vietnamese semantic tasks (retrieval/classification/clustering/STS) — PhoBERT's
`[CLS]` vector was never trained for sentence-similarity use, unlike e5/LaBSE which use
contrastive/instruction-tuned objectives. This matters for THIS project specifically because #5
(novelty detection) and #1 (clustering) both hinge on embedding cosine-similarity quality — a metric
PhoBERT wasn't optimized for. Swapping the embedding backbone is a plausible quality lever independent
of finding new features. No financial-domain-specific Vietnamese benchmark exists yet, so this is
inferred from general-purpose Vietnamese STS, not proven for finance text — reasonable but not certain.

**Effort:** Small — `src/nlp/embeddings.py` already isolates the embedding call
(`extract_phobert_embeddings`); add a sibling `extract_e5_embeddings` (via `sentence-transformers`,
new dependency) and re-run the incremental cache under a new cache filename. Cheap to A/B since the
caching/PCA/downstream code is already backbone-agnostic.

---

## 7. Temporal decay / half-life weighting of news impact

**What it is:** Instead of same-day / fixed-window news aggregation, weight each article's
contribution by `w = λ^d` (d = days old), calibrated per forecast horizon — recent zero-shot financial
NLP work explicitly tunes decay per horizon (short-horizon = steep decay, long-horizon = flatter).

**Fit here:** Directly relevant to your core finding: news matters at T+10 but not T+1/T+5. That's
consistent with a slow information-diffusion / decay story rather than instantaneous pricing — decay
weighting operationalizes "how many days of news history matter for THIS horizon" instead of assuming
a fixed window (which your current features likely use, e.g. daily/rolling counts). This is a feature
*construction* change on top of existing news-aggregation code, testable directly against the
existing T+1/T+5/T+10 HAR-Ridge setup (`src/modeling/dataset.py`, `baseline.py`) with no new
statistical/DM-test infra needed — reuses Epic 9's significance framework as-is.

**Effort:** Small-medium — modify feature aggregation in `src/modeling/features.py` to replace
flat rolling windows with exponentially-decayed sums (one new parameter, λ, grid-searched per
horizon). No new library.

---

## 8. Graph-based methods: ticker co-mention networks

**What it is:** Build a graph where tickers co-mentioned in the same article are connected (edge
weight = co-mention frequency), then use graph structure (centrality, community, or a GNN) to model
spillover — e.g. "does news about VCB move BID because they're co-mentioned/sector-linked."

**Fit here:** Conceptually appealing (you already extract per-article ticker mentions in
`_explode_tickers` in `news_embeddings.py`) but a genuine scope risk for a thesis: full GNN spillover
modeling (per 2025 literature: DGDNN, attention-graph momentum spillover) is a substantial modeling
project on its own, and one cited 2025 volatility-spillover GNN paper found multi-hop spillover
**did not clearly improve accuracy** over simpler baselines — i.e. even well-resourced papers get
mixed results here. Given your effect sizes are already small and per-ticker heterogeneous, a full GNN
is high implementation cost for uncertain payoff. A *much* cheaper partial version: a static co-mention
count (how many other VN30 tickers are mentioned in the same article as ticker X) as a scalar feature,
no graph library needed — testable in an afternoon, but this is a shadow of true graph modeling, not
GNN-level insight.

**Effort:** Full GNN — large, new library (`torch-geometric` or similar), new modeling paradigm,
likely a whole additional epic. Cheap version (co-mention count feature) — small, extends
`_explode_tickers`. **Recommend only the cheap version, if any.**

---

## Other techniques considered, not detailed (lower fit)

- **FinBERT-style domain-adaptive pretraining of PhoBERT on Vietnamese financial corpus:** would help
  general sentiment/embedding quality but requires a large in-domain corpus + GPU pretraining budget —
  disproportionate for thesis scope; the multilingual-e5 swap (#6) is a cheaper way to test "is the
  backbone the bottleneck."
- **LLM-based zero-shot volatility/sentiment scoring (prompt an LLM per article):** interesting per
  2026 "Can News Predict the Market?" paper, but that same paper's finding is limited zero-shot
  performance without careful weighting (which overlaps with #4/#7 above) — treat as an alternative
  implementation of NER/decay ideas, not a separate method.
- **Financial-STS / subtle-semantic-shift detection:** 2024 paper found existing embeddings (PhoBERT
  and general LLM embeddings alike) **fail** at detecting subtle financial narrative shifts
  (intensified sentiment, plan realization, etc.) — explicitly not solved by swapping embeddings; would
  need a purpose-built model. Out of scope for thesis effort budget.

---

## Ranked Top 3 Recommendations

### 1. Embedding-based novelty/staleness feature (#5)
**Why it wins:** Zero new encoding cost — reuses the PhoBERT cache you already built and maintain.
Directly testable against the existing DM/bootstrap significance framework (Epic 9) with almost no new
infrastructure (cosine similarity + a rolling window). Best match to the actual finding: it's a
plausible *mechanism* for why only some tickers/news show T+10 sensitivity (fresh vs. rehashed news),
turning a "weak, unexplained heterogeneity" result into a testable hypothesis.

### 2. Vietnamese uncertainty index, BBD-style (#4)
**Why it wins:** Cheapest to build (keyword frequency, no ML), completely novel axis versus everything
tried so far (market-wide daily signal vs. per-article/per-ticker features), and has a 50-year-adjacent
track record in the finance literature of correlating with realized volatility (BBD's own r=0.73 with
VIX). Given the project's honest small-effect-size finding, an aggregate uncertainty index is the kind
of "different lens" most likely to surface a signal the per-article approaches missed — and it's a
half-day of work reusing existing keyword-matching + relationship-testing code.

### 3. Temporal decay / half-life feature weighting (#7)
**Why it wins:** Speaks directly to the project's single strongest empirical fact — the effect exists
at T+10 and not T+1/T+5 — by giving the model a *mechanism* (slow decay) instead of treating horizon as
just "which column of the target we regress on." It reuses 100% of existing modeling/significance
infrastructure (`dataset.py`, `baseline.py`, `significance.py`); the only change is how news features
are aggregated before they enter the same pipeline, making it the lowest-risk way to test whether
decay-shape, not just topic content, explains the horizon-specific finding.

**Why these beat the rest:** All three (a) reuse existing pipeline components rather than requiring
new modeling paradigms (ruling out full GNN, LLM-NER pipelines, domain pretraining), (b) are directly
testable against the already-built DM/bootstrap significance machinery without new infra, and
(c) each targets a *specific, already-observed* pattern in the results (heterogeneity, null aggregate
sentiment, horizon-specificity) rather than adding generic "more features" — consistent with a
thesis-scale project that should extend, not replace, ten epics of working EDA.
