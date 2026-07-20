# Epic 14 Retrospective — Level 1/2 Feature Evaluation

**Date:** 2026-07-20
**Epic:** 14 (Level 1/2 Feature Evaluation per `docs/gpt-guide`)
**Stories:** 14-1 (Level-1 Significance), 14-2 (Level-2 Event Study by Type)
**Status:** ✅ Both done

---

## What Went Well

- **sentiment5 T+1 significant** (p=0.033) — distinct signal từ embedding-based `news_adv` (T+10), mở ra hướng hybrid feature fusion
- **event_type** operationalized từ `TOPIC_CATEGORIES` — không cần labeled data
- Vectorized scoring (`_vectorized_lexicon_scores`) xử lý được 1.45M-row corpus
- Mid-story bugs (source-name collision, `_one_sample_ttest` crash) đều được phát hiện và fix kịp

## Challenges

- **Backfill 1.45M rows mid-story** — gây gián đoạn, phát sinh bug không mong đợi
- **PhoBERT re-encode mất ~28 phút trên CPU** — chờ đợi, không tận dụng được GPU
- **ΔR² median âm** sau re-encode — phát hiện quan trọng nhưng chưa có hướng xử lý

## Action Items

| # | Action | Owner | Deadline | Status |
|---|--------|-------|----------|--------|
| 1 | Tài liệu hóa embedding output format (`data/features/news_emb_articles_*.parquet`: schema, columns, cách load) để project khác tái sử dụng | Dev | Epic 15 start | Open |
| 2 | Điều tra ΔR² trung vị âm theo ticker (PCA dim, regularization, per-ticker feature selection) | Dev | TBD | Open |
| 3 | Xác nhận thêm tín hiệu sentiment T+1 (GBM ablation đã làm, cần thêm phương pháp khác) | Dev | TBD | Open |

## Technical Notes (for next epic)

- **GPU re-encode:** `src/nlp/embeddings.py:27` đã auto-detect CUDA — chỉ cần `pip install torch` với CUDA support, không sửa code
- **Embedding output:** parquet files ở `data/features/news_emb_articles_{source}.parquet`, mỗi file có raw 768-dim + metadata (url, source, text)
- **Epic 15 chưa định nghĩa** — ứng cử viên: GNN/entity-linking, event severity, hoặc phân tích ΔR² trung vị âm
