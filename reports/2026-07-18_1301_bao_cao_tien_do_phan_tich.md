# Báo cáo tiến độ — Dự báo biến động cổ phiếu VN30 từ tin tức tiếng Việt

**Ngày báo cáo:** 2026-07-18
**Phạm vi:** Cập nhật dữ liệu, sửa lỗi thu thập dữ liệu, hoàn thiện đánh giá đặc trưng tin tức theo khung Level 1/Level 2, kiểm định giả thuyết đa cộng tuyến bằng mô hình phi tuyến.

---

## 1. Tóm tắt

- Dữ liệu tin tức được mở rộng đáng kể (**~1,45 triệu dòng**, tăng từ ~7.000 dòng), đồng thời phát hiện và khắc phục ba lỗi kỹ thuật khiến một phần dữ liệu bị loại bỏ âm thầm (mục 3.2).
- Embedding PhoBERT đã được tính lại trên toàn bộ dữ liệu mở rộng (số bài có nhắc mã VN30 dùng để encode tăng từ 4 lên 3.197 bài đối với nhóm "khách quan", và từ 2.221 lên 2.527 bài đối với nhóm "tổng hợp").
- Sau khi chạy lại toàn bộ pipeline trên embedding mới: đặc trưng embedding PhoBERT có ý nghĩa thống kê ở **cả ba horizon T+1, T+5, T+10** (trước đây chỉ T+10, sau đó T+5+T+10 với dữ liệu chưa tính lại embedding) — bằng chứng cho hiệu ứng của tin tức được củng cố đáng kể, nhưng số mã cổ phiếu hưởng lợi thực sự lại giảm (mục 5.5), cần diễn giải thận trọng.
- Bổ sung 5 chỉ số sentiment (Positive/Negative/Fear/Optimism/Uncertainty) theo khung đánh giá Level 1/Level 2 tham khảo (`docs/gpt-guide/news_feature_evaluation_guideline.md`): sentiment có ý nghĩa thống kê ở T+1 (ngắn hạn), ngược hướng với embedding (thể hiện rõ ở mọi horizon nhưng đặc biệt là dài hạn).
- Giả thuyết ban đầu cho rằng việc mất ý nghĩa khi gộp sentiment5 + event_type là do đa cộng tuyến đã được kiểm định bằng mô hình phi tuyến (GBM) và **bị bác bỏ** (mục 5.2).
- Hiệu ứng tổng thể của tin tức vẫn nhỏ và không đồng đều theo mã cổ phiếu; phần lớn loại sự kiện (earnings/dividend/M&A...) không có ảnh hưởng trung bình có ý nghĩa thống kê.
- Kết luận tổng thể của đề tài không thay đổi về bản chất: tin tức tiếng Việt là tín hiệu dự báo yếu, mang tính đặc thù theo mã cổ phiếu — nhưng bằng chứng thống kê cho sự tồn tại của tín hiệu này nay chắc chắn hơn nhờ dữ liệu và embedding được cập nhật đầy đủ.

---

## 2. Bối cảnh và mục tiêu

Đề tài phân tích mối quan hệ giữa tin tức tài chính tiếng Việt (CafeF, VnExpress, Tuổi Trẻ, Thanh Niên, VietnamPlus, SSI, VNDirect, Vietstock, VSDC, HSC, NLĐ) và biến động giá (Parkinson volatility) của 30 mã VN30. Nội dung công việc trong giai đoạn này gồm:

1. Cập nhật pipeline phân tích với dữ liệu thu thập mới.
2. Bổ sung Level 1 (kiểm định thống kê đặc trưng) và Level 2 (event study theo loại sự kiện) theo quy trình đánh giá đặc trưng chuẩn trước khi đưa vào mô hình HAR/GNN.
3. Kiểm định giả thuyết đa cộng tuyến phát sinh từ kết quả Level 1 bằng mô hình phi tuyến.

---

## 3. Dữ liệu

### 3.1 Quy mô dữ liệu sau cập nhật

| Nguồn | Số dòng | Ghi chú |
|---|---|---|
| vnstock (Vietstock reports) | 14.836 | Đầy đủ 2001–2026 |
| ssi | 1.867 | Đầy đủ |
| vndirect | 969 | Đầy đủ trong giới hạn của trang nguồn |
| tuoitre | 283.568 | Bổ sung mới, từ 2011 |
| thanhnien | 387.169 | Bổ sung mới, từ 2011 |
| vietnamplus | 773.152 | Bổ sung mới |
| cafef | 4.067 | Chưa thu thập đầy đủ lịch sử |
| vnexpress | ~103 | Chưa thu thập đầy đủ (bị chặn crawl) |
| **Tổng** | **~1.450.798** | Tăng từ ~6.900 dòng |

### 3.2 Lỗi dữ liệu phát hiện và khắc phục

Ba vấn đề được phát hiện và xử lý trong quá trình cập nhật:

1. **Trùng tên nguồn:** khi dữ liệu mới (`thanhnien_articles.csv`, `tuoitre_articles.csv`...) được bổ sung vào cùng thư mục với dữ liệu đã có (`objective/news_unenriched_thanhnien_records.csv`...), hai tệp có tên nguồn suy luận trùng nhau (cùng được gán tên "thanhnien"). Cơ chế phát hiện nguồn tin tự động trong pipeline trước đó giữ lại duy nhất một trong hai tệp, khiến dữ liệu của tệp còn lại bị loại bỏ mà không có cảnh báo. Kiểm tra cho thấy hai tệp này không phải bản sao của nhau (0% trùng URL). Cơ chế phát hiện nguồn đã được sửa để giữ lại cả hai nguồn.
2. **Sai tên cột:** `cafef_articles.csv` sử dụng cột `article_url` thay vì `url` như các nguồn khác, khiến toàn bộ dữ liệu CafeF bị loại khỏi các pipeline dùng cơ chế phát hiện nguồn tự động (embedding, sentiment, uncertainty index) do thao tác loại bỏ dòng thiếu `url`. Đã chuẩn hóa tên cột khi nạp dữ liệu.
3. **Hiệu năng tính embedding:** kiểm tra cho thấy chỉ **0,3% (4.890/1.452.945 bài)** trong dữ liệu mở rộng thực sự nhắc đến một mã VN30. Pipeline trước đó tính embedding PhoBERT cho toàn bộ bài viết rồi mới lọc theo mã cổ phiếu ở bước sau, dẫn đến ước tính thời gian xử lý khoảng 37 giờ trên CPU (máy không có GPU). Đã điều chỉnh để lọc theo mã cổ phiếu trước khi tính embedding — kết quả cuối không đổi (bước lọc sau vốn đã loại các bài không khớp), thời gian xử lý giảm còn khoảng 28 phút.

---

## 4. Phương pháp

Pipeline gồm 18 phase (profiling dữ liệu → EDA giá/tin tức → correlation → event study → feature validation → leakage check → embedding PhoBERT → Level 1/Level 2), tiếp theo là:

- **Mô hình baseline:** Ridge (tuyến tính) và GradientBoosting — GBM (phi tuyến), dự báo Parkinson volatility tại các horizon T+1/T+5/T+10.
- **Kiểm định thống kê:** Diebold-Mariano test (kiểm định ý nghĩa thống kê của chênh lệch sai số dự báo), bootstrap confidence interval, và kiểm tra tính đồng nhất của hiệu ứng qua từng mã cổ phiếu.

### Bổ sung — Level 1 (đánh giá thống kê đặc trưng)

5 chỉ số sentiment (Positive/Negative/Fear/Optimism/Uncertainty score, tính bằng từ điển từ khóa tiếng Việt) và Event type (7 loại: earnings/dividend/M&A/management/regulation/macro/sector), đánh giá bằng 5 phép đo thống kê để không bỏ sót quan hệ phi tuyến: Pearson (tuyến tính), Spearman và Kendall Tau (đơn điệu), Mutual Information và Distance Correlation (quan hệ dạng bất kỳ).

### Bổ sung — Level 2 (Event Study theo loại sự kiện)

Thay vì gộp chung mọi loại tin như trong phân tích trước đó, từng loại sự kiện được phân tích riêng biệt trong cửa sổ T-5..T0..T+10, bổ sung tính Cumulative Abnormal Return (CAR) so với benchmark thị trường (bình quân toàn VN30) — phân tích trước đó chỉ có abnormal volatility.

---

## 5. Kết quả

### 5.1 Mô hình embedding PhoBERT (tin tức tổng hợp)

Kết quả sau khi tính lại embedding trên toàn bộ dữ liệu mở rộng:

| Horizon | DM p-value | Kết luận | ΔR² 95% CI |
|---|---|---|---|
| T+1 | 0.0491 | Có ý nghĩa | [-0.00003, 0.00103] |
| T+5 | 0.0059 | Có ý nghĩa | [0.00031, 0.00160] |
| T+10 | 0.0204 | Có ý nghĩa | [0.00030, 0.00252] |

Trên tập dữ liệu ban đầu, chỉ T+10 đạt ý nghĩa thống kê. Sau khi mở rộng dữ liệu văn bản nhưng chưa tính lại embedding, T+5 cũng đạt ý nghĩa. Sau khi tính lại embedding trên toàn bộ dữ liệu mở rộng, **cả ba horizon đều đạt ý nghĩa thống kê**. Kết quả này cần được đọc cùng với mục 5.5 — mức độ đồng đều của hiệu ứng giữa các mã cổ phiếu giảm so với trước, dù kiểm định tổng thể mạnh hơn.

### 5.2 Sentiment 5-score (Level 1) và kiểm định giả thuyết đa cộng tuyến

| Feature set | T+1 (Ridge) | T+5 (Ridge) | T+10 (Ridge) | T+1/T+5/T+10 (GBM) |
|---|---|---|---|---|
| Sentiment 5-score | Có ý nghĩa (p=0.033) | Không | Không | Không (p=1.0 cả 3) |
| Event-type (7 loại) | Không | Không | Không | Không (p=1.0 cả 3) |
| Gộp cả 2 nhóm | Không | Không | Không | Không (p=1.0 cả 3) |

Với mô hình Ridge, sentiment 5-score đạt ý nghĩa thống kê tại T+1 (ngắn hạn) — trái ngược với embedding (chỉ có ý nghĩa ở dài hạn). Khi gộp sentiment5 với event_type, ý nghĩa thống kê tại T+1 biến mất, dẫn tới giả thuyết ban đầu về đa cộng tuyến giữa hai nhóm đặc trưng khi đưa cùng lúc vào mô hình tuyến tính.

Giả thuyết này được kiểm định bằng GBM — mô hình cây quyết định, không chịu ảnh hưởng của đa cộng tuyến vì tách nhánh trên từng đặc trưng độc lập. Kết quả: DM p-value = 1.0 (dự đoán giống hệt nhau về mặt số học) ở toàn bộ 9 tổ hợp, bao gồm cả khi sentiment5 được dùng riêng lẻ (không gộp với event_type). Điều này cho thấy GBM không tách nhánh trên sentiment5 hoặc event_type trong bất kỳ trường hợp nào, không phụ thuộc vào việc gộp chung hay không — **giả thuyết đa cộng tuyến bị bác bỏ**.

Diễn giải phù hợp hơn: tín hiệu mà Ridge phát hiện tại T+1 là tín hiệu tuyến tính có biên độ rất nhỏ, chỉ được mô hình tuyến tính đơn giản ghi nhận, trong khi mô hình phi tuyến coi là nhiễu không đáng tách nhánh. Kết quả này cần được xác nhận thêm (cỡ mẫu lớn hơn, phương pháp kiểm định khác) trước khi được coi là một kết luận vững chắc của đề tài.

### 5.3 Kiểm định Level 1 (84 cặp đặc trưng × mục tiêu)

- 16/84 cặp có tương quan tuyến tính (Pearson) đạt ý nghĩa sau hiệu chỉnh đa kiểm định.
- 79/84 cặp có Mutual Information xấp xỉ 0 — phần lớn đặc trưng sentiment/event-type không mang thông tin dự báo tương lai đáng kể.
- `positive_score` có tương quan tương đối mạnh với lợi suất cùng ngày (r=0.21, có ý nghĩa) — đây nhiều khả năng là quan hệ đồng thời (tin tích cực xuất hiện sau khi giá đã tăng) chứ không phải tín hiệu dự báo tương lai.

### 5.4 Event Study theo loại sự kiện (Level 2)

Trong 21 tổ hợp (7 loại sự kiện × 3 horizon) được kiểm định bằng t-test, chỉ 1/21 đạt ý nghĩa thống kê: tin vĩ mô (macro) tại T+10 có CAR trung bình khác 0 có ý nghĩa (p=0.04, n=66 sự kiện). Các loại sự kiện khác (earnings, dividend, M&A...) không có hiệu ứng trung bình rõ rệt trong dữ liệu hiện tại.

### 5.5 Tính không đồng đều theo mã cổ phiếu

Sau khi tính lại embedding trên dữ liệu mở rộng, số mã cổ phiếu có ΔR² dương (tin tức cải thiện dự báo) là **3/30 (T+1), 4/30 (T+5), 4/30 (T+10)** — thấp hơn kết quả trước đó (6-7/30). Đáng chú ý, **ΔR² trung vị hiện là số âm** ở cả ba horizon (T+1: -0.0120, T+5: -0.0236, T+10: -0.0328), trong khi ΔR² tối đa vẫn dương (T+10: 0.0241). Nghĩa là: kiểm định tổng thể (DM test, dựa trên tổng sai số dự báo trên toàn bộ 30 mã) đạt ý nghĩa thống kê, nhưng hiệu ứng này đến từ một số ít mã có cải thiện rõ rệt, trong khi phần lớn mã còn lại có kết quả dự báo kém hơn khi thêm đặc trưng embedding (nhiều khả năng do số chiều đặc trưng tăng gây overfitting nhẹ trên các mã ít có tin tức liên quan). Đây là điểm cần diễn giải thận trọng: ý nghĩa thống kê ở cấp độ tổng thể không đồng nghĩa với việc tin tức có ích cho đa số cổ phiếu.

---

## 6. Kết luận

Sau khi mở rộng dữ liệu, tính lại embedding trên toàn bộ dữ liệu mở rộng, bổ sung khung đánh giá Level 1/Level 2, và kiểm định giả thuyết đa cộng tuyến bằng mô hình phi tuyến, kết luận tổng thể của đề tài được củng cố thêm bằng chứng mà không thay đổi về bản chất:

Tin tức tài chính tiếng Việt là một tín hiệu dự báo biến động tồn tại thật về mặt thống kê ở cấp độ tổng thể (embedding PhoBERT có ý nghĩa ở cả ba horizon sau khi cập nhật dữ liệu), nhưng có biên độ nhỏ và **không đồng đều giữa các mã cổ phiếu** — số mã thực sự hưởng lợi giảm khi mô hình dùng nhiều đặc trưng hơn, cho thấy hiệu ứng tập trung ở một nhóm nhỏ cổ phiếu nhạy cảm với tin tức hơn là một hiệu ứng lan tỏa trên toàn thị trường.

Một điểm đáng chú ý: hai cách biểu diễn tin tức khác nhau (sentiment từ khóa và embedding ngữ nghĩa) dường như phản ánh hai khung thời gian tác động khác nhau của tin tức lên biến động giá — đây là hướng có thể được xem xét thêm trong các bước tiếp theo.

---

## 7. Hạn chế

- Sentiment 5-score và event-type được tính bằng từ điển từ khóa thủ công, chưa dùng mô hình học máy đã huấn luyện — có thể bỏ sót sắc thái ngôn ngữ.
- Event severity/confidence (mức độ nghiêm trọng, độ tin cậy của sự kiện) chưa được triển khai, do cần thiết kế nhãn dữ liệu riêng.
- Tính không đồng đều của hiệu ứng embedding giữa các mã cổ phiếu (mục 5.5) chưa được lý giải sâu — cần phân tích thêm để xác định liệu đây là overfitting do tăng số chiều đặc trưng hay phản ánh đặc điểm thực sự của từng mã.
- CAR sử dụng benchmark thị trường tự tính (bình quân đơn giản 30 mã VN30), chưa đối chiếu với chỉ số VN-Index chính thức.

## 8. Hướng tiếp theo

1. Phân tích nguyên nhân ΔR² trung vị âm theo mã cổ phiếu sau khi tính lại embedding (mục 5.5) — cân nhắc giảm số chiều PCA hoặc dùng regularization mạnh hơn.
2. Thiết kế event severity/confidence nếu cần tăng độ chi tiết trước khi triển khai mô hình GNN.
3. Xem xét khả năng xác nhận tín hiệu sentiment T+1 bằng phương pháp bổ sung (cỡ mẫu lớn hơn, kiểm định khác) trước khi đưa vào kết luận chính của đề tài.
4. Kết quả đầy đủ có thể xem trực quan tại dashboard: `uv run streamlit run src/dashboard/app.py`.

---

*Báo cáo tổng hợp từ các artifact trong `eda_output/`, `reports/`, và story `_bmad-output/implementation-artifacts/stories/14-1-*`, `14-2-*`. Số liệu lấy trực tiếp từ kết quả chạy thực tế, có thể tái lập bằng lệnh `uv run python -m src.eda.run_night_analysis --skip-tests`.*
