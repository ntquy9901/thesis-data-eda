# Phiên tự chủ — Tổng kết (Epics 1–9)

**Ngày:** 2026-07-12 | **Repo:** github.com/ntquy9901/thesis-data-eda | **Commits:** e9b2d05 → 941d530

## Đã hoàn tất (tự chủ, không cần approve)
- **Epic 7** Modeling: HAR-Ridge baseline, leakage-safe, comparison → news không giúp (linear, 5 tickers).
- **Epic 8** Advanced: scale 30 tickers + advanced news features (event-weighted, sentiment strength, topic flags) + GBM → vẫn null.
- **Epic 9** Significance: Diebold-Mariano + bootstrap CI + per-ticker heterogeneity + event t-test → **phát hiện tinh tế**.
- **Charts**: +5 news charts (12 total). **Report tổng hợp** + **chương luận văn tiếng Việt**.

## Phát hiện cốt lõi (nuanced)
| Horizon | DM p | Kết luận |
|---------|------|---------|
| pk_t+1 | 0.99 | KHÔNG |
| pk_t+5 | 0.39 | KHÔNG |
| **pk_t+10** | **0.0008** | **CÓ** (ΔR² CI [+0.0007, +0.0022]) |

- News giúp **7–8/30 ticker** (max ΔR² ≈0.036) — 25% nhạy cảm.
- Event abnormal vol: KHÔNG có ý nghĩa (p 0.27–0.86).
- **Kết luận:** tin tức tài chính tiếng Việt = dự báo yếu, dài hạn (T+10), theo-ticker cho biến động Parkinson; null ở ngắn hạn + event level.

## Chất lượng
- 112 tests pass, ruff clean, diff-cover ≥91% (nhiều epic 100%).
- Code review 3-layer mỗi epic, bắt nhiều critical bug (inverted date, sentiment key, FDR NaN-poisoning, leakage dead-code…).
- mypy sạch trên code dự án (numpy stub env issue).

## Deliverables
- `docs/THESIS_CHAPTER_NEWS_VOLATILITY.md` — chương luận văn (VN).
- `eda_output/report/eda_final_report.md` — báo cáo tổng hợp + Thesis Conclusion.
- `eda_output/modeling/{metrics.csv, comparison_report.md, significance_report.md}`.
- 12 charts trong `eda_output/`.
- `src/eda` (10 phase) + `src/modeling` (dataset, features, baseline, significance).

## Hướng phát triển (đề xuất)
Text embeddings (PhoBERT/LLM), intraday/LOB, LSTM/Transformer, đặc trưng hóa nhóm ticker nhạy cảm tin tức, per-source analysis.
