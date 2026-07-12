# Vietnam Stock Market — News × Volatility EDA + Modeling (Project Context)

**Last Updated:** 2026-07-12
**Repo:** https://github.com/ntquy9901/thesis-data-eda (branch `main`, public, account `ntquy9901`)
**Status:** ✅ **10 EPICS COMPLETE** — full EDA pipeline + modeling + statistical significance + web dashboard
**Commits:** e9b2d05 → d47f4ab

---

## 🎯 Mục tiêu & câu hỏi nghiên cứu
*Liệu tin tức tài chính tiếng Việt có dự đoán được biến động giá cổ phiếu (volatility) hay không?*
Phân tích theo horizon (T+1/T+5/T+10), phương pháp (tuyến tính/phi tuyến), và heterogeneity per-ticker.
**Target:** Parkinson volatility `(ln(H/L))²/(4·ln2)` — đúng baseline của dự án song hành `stock_vol_prediction01` (HAR/CryptoMamba dự đoán Parkinson, KHÔNG phải realized vol).

## 🔑 Phát hiện cốt lõi (nuanced — Epic 9 significance)
| Horizon | Diebold-Mariano p | Kết luận |
|---------|-------------------|---------|
| pk_t+1 | 0.99 | ❌ KHÔNG có ý nghĩa |
| pk_t+5 | 0.39 | ❌ KHÔNG |
| **pk_t+10** | **0.0008** | ✅ **CÓ ý nghĩa** (ΔR² CI [+0.0007, +0.0022]) |
- **Heterogeneity:** news giúp **7-8/30 ticker** (max ΔR² ≈0.036) — ~25% nhạy cảm
- **Event abnormal vol:** KHÔNG có ý nghĩa (t-test p 0.27-0.86)
- **Kết luận:** tin tức tiếng Việt = dự báo **yếu, dài hạn (T+10), theo-ticker** cho Parkinson vol; null mạnh ở ngắn hạn + event level.

## 📦 Epics đã hoàn thành (sprint-status.yaml)
1. **Epic 1** Foundation — Phase 1 profiling + Phase 2 quality
2. **Epic 2** Price EDA (Phase 3) — returns/ATR/realized vol/**Parkinson**/leakage-safe targets
3. **Epic 3** News EDA (Phase 4 + 7) — coverage/publish-time/effective_trading_date/sentiment/topics + sparse panel
4. **Epic 4** Relationship (Phase 5 + 6) — Pearson/Spearman/MI/Granger/cross-corr/FDR + event study
5. **Epic 5** Validation (Phase 8 + 9) — feature validation + explicit leakage list
6. **Epic 6** Viz + Report (Phase 10) — 12 charts + final report
7. **Epic 7** Modeling — HAR-Ridge baseline (news không giúp, 5 tickers)
8. **Epic 8** Advanced — 30 tickers + advanced news features + GBM (vẫn null)
9. **Epic 9** Significance — Diebold-Mariano + bootstrap + heterogeneity + event t-test → phát hiện tinh tế
10. **Epic 10** Dashboard — Streamlit web (5 trang, plotly)

## 🗂️ Cấu trúc code
- **`src/eda/`** — 10 phase modules (phase01…phase10) + common.py + report.py. Chạy: `python -m src.eda.<phase>` (theo thứ tự).
- **`src/modeling/`** — dataset.py (HAR + split), features.py (advanced news), baseline.py (Ridge/GBM), significance.py (DM/bootstrap).
- **`src/dashboard/`** — data.py (loaders) + app.py (Streamlit). **Chạy dashboard:** `uv run streamlit run src/dashboard/app.py`
- **`config/__init__.py`** — `EDA_TICKERS = VN30_TICKERS` (30), `EDA_OUTPUT_DIR`, paths.
- **`eda_output/`** — tất cả artifacts (profiling, quality, price, news, relationship, feature_engineering, leakage, modeling, report).
- **`docs/`** — `EDA_Guide…`, `PRD.md` (v1.2), `Technical_Architecture.md` (v1.2), `THESIS_CHAPTER_NEWS_VOLATILITY.md`.
- **`_bmad-output/`** — planning-artifacts (epics.md, PRD/Arch refs) + implementation-artifacts (sprint-status.yaml, stories/).

## 📊 Chất lượng
- **120 tests** pass (`tests/unit/`), ruff sạch, diff-cover ≥90% mọi epic.
- Code review 3-layer (`/bmad-code-review`) mỗi epic — bắt nhiều critical bug (inverted date format, sentiment key, FDR NaN-poisoning, leakage dead-code).
- BMAD workflow: mỗi story có acceptance, sprint-status tracking, summary report.
- mypy sạch trên code dự án (numpy stub env issue).

## ⚠️ bmad-loop orchestrator
`bmad-loop run` KHÔNG dùng được trên Windows native (không mux backend — tmux POSIX, psmux unimplemented; WSL2 bị chặn ở BIOS virtualization). Toàn bộ dev làm **direct implementation** (orchestrator-ready nếu fix env). Quy trình BMAD (sprint/story/review/commit) vẫn tuân thủ thủ công.

## 🔧 Tech stack
Python 3.13 (venv qua uv), pandas/polars, scikit-learn (Ridge/HistGBR), statsmodels (Granger/Ljung-Box), scipy, plotly, **streamlit** (dashboard), matplotlib. NLP: rule-based Vietnamese sentiment (`src/sprint1/task1_3_vietnamese_nlp.py`).

## 💡 Hướng phát triển (đề xuất)
Text embeddings (PhoBERT/LLM) thay count/sentiment; intraday/LOB; LSTM/Transformer (sibling project); đặc trưng hóa nhóm ticker nhạy cảm tin tức; per-source analysis; mở rộng nguồn tin (VietStock PDF 238MB).

## 📁 Data sources (MANDATORY — per CLAUDE.md, không sửa)
- News: `D:\bmad-projects\crawl_data\data` (SSI ISO, cafef ISO, vndirect DD/MM — chuẩn hóa per-source trong phase04)
- Price: `D:\bmad-projects\stock_vol_prediction01\data\raw\prices\{TICKER}_ohlcv.csv`
