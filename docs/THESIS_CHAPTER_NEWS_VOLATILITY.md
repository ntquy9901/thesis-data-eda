# Chương — Tin tức tài chính và Dự đoán Biến động Cổ phiếu Việt Nam

> Chương tổng hợp EDA + mô hình hóa. Nguồn: `eda_output/` + `src/eda` + `src/modeling`.
> Dữ liệu: 30 mã VN30, tin tức SSI/CafeF/VNDirect (≈13.000 bài), 2006–2026.
> Mọi kết quả có thể tái lập: `python -m src.eda.phase01_profiling` … `python -m src.modeling.significance`.

---

## 1. Mục tiêu và câu hỏi nghiên cứu

**Câu hỏi chính:** *Tin tức tài chính tiếng Việt có dự đoán được biến động giá cổ phiếu (volatility) hay không?*

Phân tích theo 3 trục:
- **Horizon**: ngắn hạn (T+1, T+5) vs dài hạn (T+10).
- **Phương pháp**: tuyến tính (HAR-Ridge) vs phi tuyến (Gradient Boosting).
- **Heterogeneity**: có ticker nào nhạy cảm với tin tức không?

Target dự đoán: **Parkinson volatility** `(ln(H/L))²/(4·ln2)` — đúng định nghĩa baseline trong dự án song hành `stock_vol_prediction01` (HAR, CryptoMamba đều dự đoán Parkinson, không phải realized vol).

---

## 2. Dữ liệu

| Nguồn | Quy mô | Phạm vi |
|-------|--------|---------|
| News (SSI/CafeF/VNDirect) | ≈13.000 bài (gộp) | 2016–2026 |
| Price (30 VN30 OHLCV) | ≈100.000 dòng-giao-dịch | 2006–2026 |
| Macro (DXY, SBV rates) | phụ trợ | 2006–2026 |

**Đặc thù đã xử lý:**
- **Encoding UTF-8** bắt buộc (văn bản tiếng Việt).
- **Chuẩn hóa ngày theo nguồn**: CafeF + news_articles = ISO `2026-07-04T…`; SSI/VNDirect = `DD/MM/YYYY` (đã fix bug đảo dayfirst trong review).
- **Schema khác nhau**: CafeF dùng `section`/`article_url`, các nguồn khác `category`/`url`.
- **effective_trading_date**: tin trước 15h (đóng cửa) → cùng ngày giao dịch; sau 15h/cuối tuần → ngày giao dịch kế tiếp.

---

## 3. Phương pháp

**EDA 10 phase** (theo EDA Guide): profiling → chất lượng → price EDA → news EDA → quan hệ → event study → sparse news → feature validation → leakage detection → trực quan.

**Leakage-safe** (ADR-006): target `pk_t+h` dùng `parkinson.shift(-h)` (chỉ giá tương lai); feature dùng rolling trailing; train/test split theo thời gian (train ≤ 2024, test ≥ 2025), không shuffle.

**Mô hình:**
- **HAR features** (Heterogeneous Autoregressive): `har_daily/weekly/monthly` = rolling mean 1/5/22 ngày của Parkinson (trailing, không look-ahead).
- **Ridge** (tuyến tính HAR) + **HistGradientBoosting** (phi tuyến).
- 3 bộ feature: `price` / `+news_basic` (count, sentiment) / `+news_adv` (event-weighted, sentiment strength, topic flags).

**Kiểm định ý nghĩa:** Diebold-Mariano trên loss differential + bootstrap CI (1000) + per-ticker ΔR² + t-test abnormal vol.

---

## 4. Kết quả chính

### 4.1 Sentiment (rule-based tiếng Việt)
Mean = 0.151; positive 30%, negative 14%, neutral 56%. → tín hiệu sentiment có tồn tại nhưng lệch về trung tính.

### 4.2 Tương quan news ↔ vol (Phase 5, có FDR)
Yếu ở mức daily. Chỉ `neg_news vs rv_t+10` còn significant sau hiệu chỉnh đa phép thử.

### 4.3 Mô hình — news có giúp gì?

| Horizon | Ridge R² (price) | ΔR² (+news) | GBM R² (price) |
|---------|------------------|-------------|----------------|
| pk_t+1 | 0.28 | ≈ +0.0007 | 0.18 |
| pk_t+5 | 0.17 | ≈ +0.001 | 0.11 |
| pk_t+10 | 0.12 | ≈ +0.001 | 0.08 |

→ ΔR² từ news **rất nhỏ**; GBM gần như bỏ qua feature tin tức (cây không split trên tín hiệu yếu).

### 4.4 Ý nghĩa thống kê (Epic 9 — mấu chốt)

| Horizon | Diebold-Mariano p | Kết luận |
|---------|-------------------|----------|
| pk_t+1 | **0.99** | KHÔNG có ý nghĩa |
| pk_t+5 | **0.39** | KHÔNG có ý nghĩa |
| pk_t+10 | **0.0008** | **CÓ ý nghĩa** (ΔR² CI [+0.0007, +0.0022]) |

- **Heterogeneity**: news giúp **7–8/30 ticker** (max ΔR² ≈ 0.036) — khoảng 25% ticker nhạy cảm tin tức.
- **Event study**: abnormal vol **không** có ý nghĩa trung bình (t-test p = 0.27–0.86).

---

## 5. Bàn luận

Phát hiện **tinh tế**, không phải "có/không" đơn giản:

1. **Ngắn hạn: null mạnh.** Ở T+1/T+5, tin tức tiếng Việt (dạng daily count/sentiment/topic) **không** cải thiện dự đoán Parkinson vol so với baseline HAR giá — ổn định qua tuyến tính và phi tuyến.
2. **Dài hạn (T+10): có hiệu ứng thật, nhỏ.** DM significant với ΔR² dương nhưng biên độ ~0.1–0.2% — ý nghĩa thống kê nhưng **ý nghĩa thực tiễn nhỏ**.
3. **Tính chất ticker-specific.** ~25% ticker hưởng lợi tin tức (có thể là ticker có tin tức nhiều/nhạy cảm ngành); đa số không.
4. **Event-level: không có phản ứng trung bình.** Các "sự kiện tin tức lớn" không đẩy vol lên đáng kể trên trung bình — có thể do thị trường VN hấp thụ nhanh hoặc noise lớn.

**Giải thích khả dĩ:** ở tần suất daily, tin tức tài chính VN quá thưa/noisy để tạo tín hiệu dự đoán mạnh; HAR giá (vol hôm nay = dự báo tốt nhất vol ngày mai) thống trị. Tín hiệu mạnh hơn có thể nằm ở: embedding văn bản (PhoBERT/LLM), dữ liệu intraday, hoặc gắn theo sự kiện cụ thể chứ không phải tổng hợp daily.

---

## 6. Kết luận

- **Trả lời câu hỏi nghiên cứu:** Tin tức tài chính tiếng Việt là một **dự báo yếu, dài hạn (T+10), và theo-ticker** cho biến động Parkinson — không phải tín hiệu mạnh tức thời. Null ở ngắn hạn là ổn định; hiệu ứng T+10 là điểm duy nhất tin tức có giá trị ý nghĩa thống kê.
- **Giá trị thực hành:** feature tin tức nên được giữ cho horizon dài + các ticker nhạy cảm; không đáng cho dự báo ngắn hạn.
- **Giá trị phương pháp:** quy trình EDA + leakage-safe + kiểm định ý nghĩa (DM/bootstrap) cho phép đưa ra kết luận có sức mạnh thống kê, không chỉ "nhìn ΔR²".

## 7. Hướng phát triển
1. Feature văn bản sâu: PhoBERT/sentence-transformer embeddings thay cho count/sentiment rule-based.
2. Dữ liệu intraday /LOB để đo phản ứng sự kiện chính xác hơn.
3. Mô hình chuỗi (LSTM/Transformer) — dự án song hành cho thấy có cải thiện capacity.
4. Phân tích ticker-specific: đặc trưng hóa nhóm ticker nhạy cảm tin tức (ngành, vốn hóa, thanh khoản).
5. Mở rộng nguồn tin (VietStock PDF 238MB, thêm nguồn) + cập nhật daily.

---

## Phụ lục: tái lập
- Pipeline: `src/eda/phase01…phase10` + `src/eda/report.py`.
- Mô hình: `src/modeling/{dataset,features,baseline,significance}.py`.
- Artifacts: `eda_output/{profiling,quality,price,news,relationship,feature_engineering,leakage,modeling,report}/`.
- 112 unit/integration tests pass, ruff clean, diff-cover ≥91% mọi epic.
