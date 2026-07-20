# Vietnam Stock Market — News × Volatility EDA + Modeling (Project Context)

**Last Updated:** 2026-07-18
**Repo:** https://github.com/ntquy9901/thesis-data-eda (branch `main`, public, account `ntquy9901`)
**Status:** ✅ **14 EPICS COMPLETE** — full EDA pipeline (18 phases) + modeling + Level 1/2 feature evaluation + web dashboard
**Latest commits touch:** Epic 13 (night-run pipeline) + Epic 14 (Level 1/2 evaluation, 2026-07-18 session)

---

## 🎯 Mục tiêu & câu hỏi nghiên cứu
*Liệu tin tức tài chính tiếng Việt có dự đoán được biến động giá cổ phiếu (volatility) hay không?*
Phân tích theo horizon (T+1/T+5/T+10), phương pháp (tuyến tính/phi tuyến), và heterogeneity per-ticker.
**Target:** Parkinson volatility `(ln(H/L))²/(4·ln2)` — đúng baseline của dự án song hành `stock_vol_prediction01`.

## 🔑 Phát hiện cốt lõi (cập nhật 2026-07-18 CUỐI PHIÊN, sau backfill + re-encode PhoBERT đầy đủ — xem báo cáo `reports/2026-07-18_1301_bao_cao_tien_do_phan_tich.md`)

### Embedding PhoBERT (tin tức tổng hợp, "news_adv") — re-encode xong, số liệu FINAL
Encode lại toàn bộ: khach_quan 4→**3197** bài, tong_hop 2221→**2527** bài (chỉ 0.3% của 1.45M bài mới nhắc VN30 ticker — filter theo ticker TRƯỚC KHI encode, xem mục môi trường bên dưới).

| Horizon | DM p | Kết luận | ΔR² 95% CI |
|---|---|---|---|
| T+1 | **0.0491** | ✅ Có ý nghĩa (mới xuất hiện sau re-encode) | [-0.00003, 0.00103] |
| T+5 | **0.0059** | ✅ Có ý nghĩa | [0.00031, 0.00160] |
| T+10 | **0.0204** | ✅ Có ý nghĩa | [0.00030, 0.00252] |

**QUAN TRỌNG — đọc cùng heterogeneity:** cả 3 horizon giờ đều significant ở cấp tổng thể, NHƯNG số ticker có ΔR²>0 giảm còn **3-4/30** (trước đó 6-7/30) và **ΔR² TRUNG VỊ ÂM** ở cả 3 horizon (T+1: -0.012, T+5: -0.024, T+10: -0.033) trong khi max vẫn dương (T+10: 0.024). Nghĩa là: ý nghĩa thống kê tổng thể đến từ 1 nhóm nhỏ ticker cải thiện mạnh, còn phần lớn ticker khác bị mô hình dự báo TỆ HƠN khi thêm embedding (khả năng overfit nhẹ do tăng chiều đặc trưng). Diễn giải "tin tức giúp dự báo" cần đi kèm caveat này, KHÔNG nên chỉ trích dẫn DM p-value.

### Sentiment 5-score (Level 1, Epic 14) — không đổi (không phụ thuộc PhoBERT)
- `price+sentiment5` (Ridge): **có ý nghĩa ở T+1** (p=0.033) — NGƯỢC hướng với embedding trước đây (giờ embedding cũng significant T+1 nên bớt "ngược" hơn, nhưng vẫn là 2 cơ chế riêng). `event_type` một mình: không ý nghĩa. Gộp 2 nhóm: mất ý nghĩa.
- **Đã kiểm định bằng GBM**: DM p=1.0 tuyệt đối ở CẢ 9 tổ hợp (kể cả sentiment5 riêng lẻ) → **bác bỏ giả thuyết đa cộng tuyến**. Diễn giải đúng: tín hiệu Ridge T+1 là tuyến tính, rất nhỏ, GBM coi là nhiễu — cần xác nhận thêm trước khi coi là kết luận chắc chắn.
- Level 1 (84 cặp feature×target, 5 phép đo Pearson/Spearman/Kendall/MI/dcor): 16/84 Pearson-significant nhưng 79/84 MI≈0 — đa số "significant" chỉ vì cỡ mẫu lớn (r=0.04-0.08), không phải quan hệ mạnh.
- Event Study by Type (Level 2, CAR): chỉ 1/21 tổ hợp (macro × T+10) có ý nghĩa.

### Kết luận tổng thể (final, cuối phiên 2026-07-18)
Tin tức tiếng Việt = tín hiệu **tồn tại thật về mặt thống kê tổng thể** (embedding significant cả 3 horizon) nhưng **không đồng đều mạnh giữa các ticker** (median ΔR² âm, chỉ 3-4/30 ticker hưởng lợi thật) — không phải hiệu ứng lan tỏa toàn thị trường, tập trung ở nhóm nhỏ ticker nhạy cảm tin tức.

## 📦 Epics đã hoàn thành (sprint-status.yaml)
1-10: xem lịch sử trước — Foundation, Price/News EDA, Relationship+Event Study, Validation+Leakage, Viz+Report, Modeling (5→30 tickers), Significance, Dashboard.
11. **Epic 11** News Embedding (PhoBERT, khách_quan/tổng_hợp, drop sentiment cũ)
12. **Epic 12** Novelty/Uncertainty/Temporal-decay features (Phase 12-16)
13. **Epic 13** Night-run comprehensive pipeline (`src/eda/run_night_analysis.py --skip-tests`)
14. **Epic 14** Level 1/2 Feature Evaluation (Phase 17-18) — sentiment5 + event_type theo `docs/gpt-guide/news_feature_evaluation_guideline.md`, GBM ablation kiểm định đa cộng tuyến

## 🗂️ Cấu trúc code
- **`src/eda/`** — 18 phase modules (phase01…phase18) + common.py + report.py + run_night_analysis.py.
- **`src/modeling/`** — dataset.py (HAR + split), features.py (advanced news/embedding), baseline.py (Ridge/GBM, FEATURE_SETS bao gồm sentiment5/event_type), significance.py (DM/bootstrap + `_ablation_block` cho per-family Ridge/GBM).
- **`src/features/`** — news_embeddings.py (PhoBERT, ticker pre-filter TRƯỚC encode), sentiment_scores.py (5 sentiment + 7 event-type, vectorized).
- **`src/dashboard/`** — data.py (loaders) + app.py (Streamlit, 13 trang, MỌI biểu đồ/bảng có `_insight()` giải thích+gợi ý). **Chạy:** `uv run streamlit run src/dashboard/app.py`
- **`config/__init__.py`** — `EDA_TICKERS = VN30_TICKERS` (30), `EDA_OUTPUT_DIR`, paths.
- **`docs/gpt-guide/`** — kim chỉ nam Level 1/2 evaluation + kế hoạch News-HAR-GNN (tham khảo, KHÔNG phải toàn bộ đã implement — GNN/entity-linking chưa làm).
- **`_bmad-output/`** — sprint-status.yaml + stories/ (14-1, 14-2 mới nhất).

## ⚠️ Vấn đề dữ liệu quan trọng (2026-07-18)
- **Backfill lớn:** crawl_data tăng từ ~6.900 → **~1.450.798 dòng** (tuoitre/thanhnien/vietnamplus mới). Chỉ **0,3% (4.890 bài)** thực sự nhắc VN30 ticker.
- **Bug đã sửa (`src/data/discover_news.py`):** (1) tên nguồn trùng (`objective/news_unenriched_thanhnien_records.csv` vs `thanhnien_articles.csv`) từng bị silently drop 1 file — giờ disambiguate bằng suffix `_root`/`_objective`; (2) `cafef_articles.csv` dùng cột `article_url` thay vì `url` → từng bị loại 100% khỏi pipeline embedding, giờ đã rename.
- **Tối ưu quan trọng (`src/features/news_embeddings.py::_load_group`):** filter theo ticker-mention TRƯỚC KHI encode PhoBERT (không phải sau) — giảm job re-embed từ ước tính 37 giờ xuống **~28 phút thực tế** trên CPU (không có GPU trên máy này; ước tính ban đầu ~8 phút thiếu overhead reload model 12 lần/nguồn), không đổi kết quả cuối vì downstream vốn đã loại bài không match ticker.
- **Embedding ĐÃ re-encode xong** (2026-07-18, cuối phiên) trên toàn bộ data mới — xem số liệu ở mục "Phát hiện cốt lõi" trên. Lệnh: `python -m src.features.news_embeddings` (~28 phút).
- **Bug `pyarrow` bị Windows Application Control Policy chặn** (không phải lỗi cài đặt — reinstall không giải quyết được): mọi `pd.read_parquet`/`to_parquet` lỗi `ImportError: DLL load failed... Application Control policy has blocked this file`. Fix: `uv add fastparquet` (đã thêm vào pyproject.toml) — pandas tự fallback sang fastparquet khi `engine="auto"` (mặc định), KHÔNG cần sửa code gọi read/write parquet ở đâu khác trong repo.
- **Môi trường uv/numpy dễ bị corrupt dist-info** nếu nhiều `uv run` chạy song song → lỗi "Access is denied" khi xóa file .dll đang bị khóa bởi process khác (dashboard). Fix: đóng hết `python.exe` rồi `uv sync --extra dev --reinstall-package numpy` (hoặc package khác bị corrupt — dấu hiệu: `dist-info` folder chỉ có `licenses/`, thiếu METADATA/RECORD).

## 📊 Chất lượng
- Unit test hiện có marker `slow` (đăng ký sẵn trong pyproject.toml) cho 3 smoke test xử lý full 1.45M-row corpus — chạy nhanh: `uv run pytest tests/unit -m "not slow"`.
- Toàn bộ dashboard (13 trang) đã verify qua AppTest (0 exception) + `test_dashboard.py` (18/18 pass).

## 🔧 Tech stack
Python 3.13 (venv qua uv), pandas/polars, scikit-learn (Ridge/HistGBR), statsmodels (Granger/Ljung-Box/Kendall qua scipy), scipy, `dcor` (Distance Correlation, mới thêm Epic 14), plotly, streamlit, matplotlib. NLP: PhoBERT (`vinai/phobert-base`, CPU-only, không có GPU) + rule-based Vietnamese sentiment lexicon (`src/features/sentiment_scores.py`).

## 💡 Hướng phát triển (đề xuất, chưa làm)
- **Điều tra ΔR² trung vị âm theo ticker** (mục "Phát hiện cốt lõi" trên) — ưu tiên cao nhất hiện tại, vì ảnh hưởng trực tiếp tới cách diễn giải "tin tức có ý nghĩa". Hướng thử: giảm PCA dim, regularization mạnh hơn cho Ridge, hoặc per-ticker feature selection.
- Event severity/confidence (Level 1 đầy đủ) — cần thiết kế nhãn riêng.
- Xác nhận thêm tín hiệu sentiment T+1 (cỡ mẫu lớn hơn/phương pháp khác) trước khi đưa vào kết luận chính.
- GNN/entity-linking chính thống (IDS/HOSE) — theo `docs/gpt-guide`, chỉ nên làm SAU khi non-graph fusion đã validate xong (hiện đã validate: tín hiệu yếu).

## 📁 Data sources (MANDATORY — per CLAUDE.md, không sửa)
- News: `D:\bmad-projects\crawl_data\data` (discover tự động qua `src/data/discover_news.py`, KHÔNG hardcode filename)
- Price: `D:\bmad-projects\stock_vol_prediction01\data\raw\prices\{TICKER}_ohlcv.csv`
