# Lộ trình tiếp theo sau Epic 16–18

## 1. Tóm tắt kết quả hiện tại

### Epic đã hoàn thành

#### Epic 16 — News Modeling

- 16-1: Dual-group embeddings — DONE
- 16-2: EWMA + multi-EWMA + novelty/dispersion/shock — DONE
- 16-3: Ticker clustering — DONE
- 16-4: Mixture-of-Experts — DONE
- 16-5: Ablation + permutation + OOS — DONE

#### Epic 17 — News Feature Quality

- Toàn bộ feature đã được triển khai.
- Kết quả đã được ghi nhận.
- Epic hoàn thành.

#### Epic 18 — HAR Residual

- Đã triển khai residual model.
- Đã đánh giá theo ticker.
- Đã đánh giá theo volatility regime.
- Epic hoàn thành.

---

## 2. Kết quả chính

| Feature / Model | ΔR² `pk_t+1` |
|---|---:|
| Basic dual-group | +0.0007 |
| EWMA 30 ngày | −0.0026 |
| Full 523 features | −0.0396 |
| Novelty | +0.0007 |
| HAR residual | −0.0002 |
| Mixture-of-Experts | ≈ 0.0000 |

### Phát hiện mới

News chỉ có tác dụng trong **high-volatility regime**:

\[
\Delta R^2 = +0.0012
\]

Trong low-volatility regime, news features làm kết quả xấu hơn.

---

## 3. Diễn giải kết quả

### Basic dual-group: `+0.0007`

Có incremental signal nhưng rất nhỏ.

Kết quả này cho thấy:

- Embedding cơ bản có thể chứa một phần thông tin.
- Signal không đủ mạnh để tạo cải thiện lớn.
- Cần kiểm tra độ ổn định qua fold, ticker và thời gian.

### Novelty: `+0.0007`

Novelty có giá trị tương đương basic dual-group.

Điều này gợi ý:

- Độ mới của thông tin có ích hơn aggregation dài hạn.
- News effect có thể đến từ semantic shock thay vì sentiment hoặc news volume.
- Nên giữ novelty trong feature set tối giản.

### EWMA 30 ngày: `−0.0026`

EWMA 30 ngày làm kết quả xấu hơn.

Giải thích có thể gồm:

- Làm mượt quá mạnh khiến tín hiệu ngắn hạn biến mất.
- Tin cũ tiếp tục ảnh hưởng feature dù thị trường đã phản ánh.
- Một decay window cố định không phù hợp cho mọi loại sự kiện.
- Signal thực tế mang tính shock hoặc regime-dependent.

### Full 523 features: `−0.0396`

Feature explosion gây overfitting nghiêm trọng.

Kết quả cho thấy:

- Noise lớn hơn signal.
- Collinearity giữa các cửa sổ EWMA và interaction rất cao.
- Feature selection không thể hoàn toàn cứu được signal yếu.
- Dataset chưa đủ lớn để hỗ trợ hàng trăm news features.

### HAR residual: `−0.0002`

News không giải thích được residual của HAR một cách ổn định trên toàn bộ dữ liệu.

Điều này cho thấy:

- News không tạo incremental contribution tổng quát.
- Residual vẫn chủ yếu là noise.
- News effect có thể chỉ xuất hiện trong một số regime hoặc ticker.

### Mixture-of-Experts: `≈ 0`

MoE không cải thiện đáng kể.

Kết luận:

- Vấn đề không nằm ở thiếu model capacity.
- Model phức tạp hơn không tạo thêm signal.
- Cần xác minh điều kiện xuất hiện của signal thay vì tăng complexity.

### High-volatility regime: `+0.0012`

Đây là phát hiện quan trọng nhất.

Kết luận kỹ thuật:

> News signal cho `pk_t+1` là sparse, conditional và regime-dependent, không phải tín hiệu ổn định trên toàn bộ dataset.

---

# 4. Quyết định chiến lược

## 4.1 Dừng mở rộng news model tổng quát cho T+1

Không nên tiếp tục các hướng sau ở thời điểm hiện tại:

- Thêm nhiều embedding dimensions.
- Thêm nhiều EWMA windows.
- Tăng từ 523 lên hàng nghìn features.
- Làm Mixture-of-Experts phức tạp hơn.
- Chuyển ngay sang LSTM hoặc Transformer lớn.
- Fine-tune PhoBERT trực tiếp theo target volatility.
- Thêm nhiều interaction chưa có giả thuyết rõ ràng.

Các Epic 16–18 đã đủ để cho thấy:

- Complexity không giải quyết được signal yếu.
- Feature engineering quá rộng làm hiệu năng xấu đi.
- News chỉ có khả năng hữu ích trong một số điều kiện cụ thể.

## 4.2 Giữ baseline news tối giản

Feature set ban đầu nên giới hạn khoảng 10–30 features.

Nhóm đề xuất:

```text
HAR price-only
Basic dual-group embedding
News novelty
Maximum semantic shock
News dispersion
Has-news mask
Days since last news
High-vol regime flag
Sensitive ticker flag
Novelty × high-vol regime
Shock × high-vol regime
```

---

# 5. Epic 19 — Regime-Conditional News Validation

## Mục tiêu

Xác minh xem hiệu ứng:

\[
\Delta R^2 = +0.0012
\]

trong high-volatility regime có:

- Ổn định theo thời gian.
- Ổn định theo ticker.
- Tốt hơn placebo.
- Có ý nghĩa thống kê.
- Có ý nghĩa thực tế.

## Stories

```text
19-1 Define ex-ante volatility regimes
19-2 Threshold sensitivity analysis
19-3 High-vol-only news model
19-4 Sensitive-ticker × high-vol analysis
19-5 Nested walk-forward validation
19-6 Statistical and economic significance
19-7 Final keep/drop decision
```

---

## 5.1 Định nghĩa high-volatility regime không leakage

Regime phải được xác định chỉ bằng dữ liệu có sẵn tại ngày \(t\).

Ví dụ:

\[
\text{high\_vol}_t =
I\left(
\text{index\_vol}_{20,t}
>
Q_{0.8,t}
\right)
\]

Trong đó:

- `index_vol_20d` được tính tại thời điểm \(t\).
- \(Q_{0.8,t}\) là percentile 80% của lịch sử trước ngày \(t\).
- Không dùng percentile tính trên toàn bộ dataset.

Nên thử các threshold:

```text
Rolling percentile 60%
Rolling percentile 70%
Rolling percentile 80%
Rolling percentile 90%
```

### Continuous regime score

Ngoài hard gate, có thể thử:

\[
g_t =
\sigma\left(
a \cdot z(\text{index\_vol}_{20,t}) + b
\right)
\]

Dự báo:

\[
\hat y_t =
\hat y^{HAR}_t
+
g_t \cdot \Delta^{news}_t
\]

Tuy nhiên, hard gate nên được triển khai trước vì:

- Dễ giải thích.
- Dễ debug.
- Ít tham số.
- Giảm nguy cơ overfitting.

---

## 5.2 Kiểm tra stability theo thời gian

Cần báo cáo theo từng giai đoạn:

```text
2025 H1
2025 H2
2026 H1
Từng walk-forward fold
Các high-volatility episodes riêng biệt
```

Câu hỏi cần trả lời:

- Signal có xuất hiện trên nhiều fold không?
- Hay chỉ xuất hiện trong một giai đoạn cực đoan?
- Có một volatility event nào chi phối toàn bộ kết quả không?
- Dấu của ΔR² có ổn định không?

---

## 5.3 Kiểm tra stability theo ticker

Cần chạy:

```text
30 tickers riêng biệt
4 sensitive tickers
26 non-sensitive tickers
Leave-one-ticker-out
Leave-one-sensitive-ticker-out
```

Tạo bảng phân tích:

| Nhóm | Low/normal vol | High vol |
|---|---:|---:|
| Sensitive ticker | ΔR² | ΔR² |
| Non-sensitive ticker | ΔR² | ΔR² |

Cần xác định:

> High-vol signal tồn tại trên nhiều ticker hay chỉ do một hoặc hai ticker tạo ra?

Nếu bỏ một ticker mà toàn bộ signal biến mất, chưa thể xem đây là signal tổng quát.

---

## 5.4 Statistical validation cho high-vol subset

Tối thiểu cần báo cáo:

```text
Number of high-vol observations
Delta R2
RMSE delta
MAE delta
QLIKE delta
DM p-value
Block-bootstrap 95% CI
Per-ticker win rate
Per-fold win rate
```

### Block bootstrap

Target volatility có serial dependence và các horizon có thể overlap.

Do đó nên dùng:

- Moving block bootstrap.
- Stationary bootstrap.
- Block length được chọn dựa trên autocorrelation hoặc sensitivity analysis.

Không dùng bootstrap ngẫu nhiên từng row độc lập.

---

## 5.5 Placebo tests

Cần chạy tối thiểu các placebo sau:

### Placebo 1 — Shuffle news theo block thời gian

Giữ autocorrelation gần đúng nhưng phá liên kết news–target.

### Placebo 2 — Time-shift news

Dịch news features:

```text
−10 ngày
−5 ngày
+5 ngày
+10 ngày
```

Signal thật phải mạnh hơn các offset không hợp lý.

### Placebo 3 — Randomize ticker mapping

Gán news của ticker này sang ticker khác.

Nếu kết quả không giảm, feature có thể chỉ đang phản ánh market-wide regime.

### Placebo 4 — Fake regime thresholds

Dùng các threshold ngẫu nhiên hoặc regime flag bị dịch thời gian.

### Placebo 5 — Random feature cùng distribution

Tạo random feature có mean, std và autocorrelation gần giống novelty.

Model thật phải tốt hơn placebo một cách ổn định.

---

# 6. Epic 20 — Conditional Lightweight Model

## Mục tiêu

Xây mô hình có fallback an toàn:

```text
Low/normal volatility
    -> HAR price-only

High volatility
    -> HAR price-only + small news adjustment
```

## Stories

```text
20-1 Reduce to 10–30 stable features
20-2 HAR fallback architecture
20-3 High-vol hard gate
20-4 Optional soft gate
20-5 Nested walk-forward OOS
20-6 Model calibration and clipping
20-7 Operational fallback
```

---

## 6.1 Kiến trúc đề xuất

\[
\hat y_t =
\begin{cases}
\hat y^{HAR}_t + \Delta^{news}_t,
& \text{high-vol regime} \\
\hat y^{HAR}_t,
& \text{low/normal-vol regime}
\end{cases}
\]

Có thể thêm shrinkage:

\[
\hat y_t =
\hat y^{HAR}_t
+
\lambda g_t \Delta^{news}_t
\]

Trong đó:

- \(g_t \in \{0,1\}\) với hard gate.
- \(g_t \in [0,1]\) với soft gate.
- \(\lambda\) là hệ số co nhỏ để tránh news adjustment quá mạnh.

### Clipping news adjustment

\[
\Delta^{news}_t =
\operatorname{clip}
\left(
\Delta^{news}_t,
-q,
+q
\right)
\]

Trong đó \(q\) được học trên training fold.

Mục tiêu:

- Tránh một số extreme feature tạo adjustment quá lớn.
- Bảo vệ HAR baseline.
- Giảm ảnh hưởng outlier.

---

## 6.2 Feature set tối giản

### Semantic content

```text
dual_group_pc0
dual_group_pc1
centroid_coherence
novelty
maximum_shock
dispersion
```

### Availability

```text
has_news
days_since_last_news
unique_event_count
```

### Context interaction

```text
novelty_x_high_vol
shock_x_high_vol
pc0_x_index_vol_20d
novelty_x_abnormal_volume
```

### Ticker context

```text
sensitive_ticker_flag
ticker_cluster
```

Bắt đầu với khoảng 15 features.

Không đưa toàn bộ 523 features trở lại model.

---

## 6.3 Forward ablation

Chạy theo thứ tự:

```text
HAR
HAR + semantic content
HAR + semantic content + availability
HAR + semantic content + availability + interactions
HAR + selected ticker context
HAR + high-vol gate
```

Mỗi feature group chỉ được giữ nếu:

- Cải thiện OOS.
- Dấu ổn định.
- Không gây hại rõ trong low-vol regime.
- Không phụ thuộc vào một fold hoặc một ticker.

Feature selection phải được thực hiện bên trong từng training fold.

---

# 7. Epic 21 — Horizon và Target Expansion

## Mục tiêu

Kiểm tra xem news signal có phù hợp hơn với:

- Horizon dài hơn.
- Volatility spike.
- Abnormal volatility.

## Stories

```text
21-1 Repeat for T+5
21-2 Repeat for T+10
21-3 Repeat for T+22
21-4 Volatility spike classification
21-5 Abnormal-volatility magnitude
21-6 Multi-horizon comparison
21-7 Target robustness
```

---

## 7.1 Chuyển trọng tâm sang horizon dài hơn

Kết quả EDA trước đó cho thấy news có xu hướng hữu ích hơn ở T+10.

Cần chạy cùng một pipeline cho:

```text
pk_t+5
pk_t+10
pk_t+22
```

Bảng báo cáo chính:

| Horizon | All regimes | High-vol | Low/normal-vol |
|---|---:|---:|---:|
| T+1 | ΔR² | ΔR² | ΔR² |
| T+5 | ΔR² | ΔR² | ΔR² |
| T+10 | ΔR² | ΔR² | ΔR² |
| T+22 | ΔR² | ΔR² | ΔR² |

Giả thuyết:

> News novelty không dự báo tốt volatility ngay ngày kế tiếp, nhưng trong high-vol regime có thể dự báo biến động tích lũy trong 5–22 ngày.

Nếu T+10 hoặc T+22 ổn định hơn, đó nên trở thành horizon chính của nghiên cứu.

---

## 7.2 Volatility spike classification

News có thể không dự báo tốt mức volatility liên tục, nhưng có thể dự báo spike.

Target:

\[
\text{vol\_spike}_{t+h}
=
I\left(
y_{t+h} > Q_{0.8,t}
\right)
\]

Có thể thử các threshold:

```text
70th percentile
80th percentile
90th percentile
Top-decile abnormal volatility
```

Metrics:

```text
PR-AUC
ROC-AUC
F1
Recall at fixed precision
Precision at top K
Brier score
Calibration error
```

Với class imbalance, PR-AUC quan trọng hơn accuracy.

---

## 7.3 Abnormal volatility target

\[
\text{abnormal\_vol}_{t+h}
=
y_{t+h}
-
\hat y^{HAR}_{t+h}
\]

Có thể dùng mô hình hai bước:

### Bước 1 — Spike probability

```text
P(volatility spike | market + news)
```

### Bước 2 — Spike magnitude

```text
Expected abnormal volatility magnitude
```

Dự báo cuối:

\[
E[\text{abnormal vol}]
=
P(\text{spike})
\times
E[\text{magnitude} \mid \text{spike}]
\]

News có thể phù hợp với bài toán event detection hơn là point forecast liên tục.

---

# 8. Epic 22 — Final Research Decision

## Mục tiêu

Đưa ra quyết định cuối cùng:

- Giữ news branch.
- Chỉ giữ trong high-vol regime.
- Chỉ giữ cho một số ticker.
- Chỉ giữ cho horizon dài.
- Hoặc loại bỏ hoàn toàn khỏi production model.

## Stories

```text
22-1 Statistical significance
22-2 Economic significance
22-3 Stability across ticker and time
22-4 Compute-cost assessment
22-5 Keep/drop news branch
22-6 Final report
22-7 Reproducibility package
```

---

## 8.1 Statistical significance

Cần xem xét:

```text
DM p-value
Block-bootstrap confidence interval
Multiple-testing correction
Per-fold consistency
Per-ticker consistency
Placebo comparison
```

Không kết luận chỉ dựa vào một ΔR² dương nhỏ.

---

## 8.2 Economic significance

Ngay cả khi có ý nghĩa thống kê, cần đánh giá:

- Mức giảm RMSE/QLIKE có đủ lớn không?
- Có cải thiện ranking volatility không?
- Có giúp cảnh báo spike không?
- Chi phí tạo PhoBERT embedding có đáng không?
- Độ trễ pipeline có chấp nhận được không?
- Signal có dùng được trong quyết định thực tế không?

Nếu chi phí cao nhưng ΔR² chỉ khoảng `+0.001`, có thể không phù hợp cho production.

---

## 8.3 Compute-cost assessment

Báo cáo:

```text
Embedding generation cost
Feature generation time
Storage size
Inference latency
Retraining time
Maintenance complexity
Incremental metric gain
```

Tạo chỉ số:

\[
\text{Value Ratio}
=
\frac{\text{Incremental predictive gain}}
{\text{Compute and maintenance cost}}
\]

---

# 9. Stopping criteria

Để tránh tiếp tục tốn thời gian vào signal quá yếu, cần đặt điều kiện dừng trước.

News branch chỉ nên được giữ nếu đáp ứng phần lớn các điều kiện sau:

- `ΔR² > 0` trên đa số walk-forward folds.
- Block-bootstrap CI chủ yếu lớn hơn 0.
- QLIKE và ít nhất một metric khác cùng cải thiện.
- Tối thiểu 50–60% ticker high-vol không bị giảm hiệu năng.
- Kết quả không phụ thuộc vào một ticker duy nhất.
- Kết quả tốt hơn placebo tests.
- Cải thiện xuất hiện trong ít nhất hai giai đoạn thời gian.
- Feature importance hoặc coefficient đủ ổn định.
- Signal tái lập được với seed khác.
- Mức cải thiện đủ lớn để bù chi phí tạo embedding.

Nếu không đạt, kết luận hợp lý là:

> News không mang lại incremental predictive value đủ ổn định cho Parkinson volatility trong dataset hiện tại.

Đây vẫn là một kết quả nghiên cứu có giá trị.

---

# 10. Acceptance criteria cho Epic 19

Epic 19 được xem là hoàn thành khi có:

- Định nghĩa regime không leakage.
- Ít nhất bốn threshold được kiểm tra.
- Kết quả theo fold.
- Kết quả theo ticker.
- Leave-one-ticker-out analysis.
- Block-bootstrap CI.
- DM test cho high-vol subset.
- Tối thiểu ba placebo tests.
- Báo cáo số lượng sample trong từng regime.
- Kết luận keep/drop rõ ràng.

---

# 11. Acceptance criteria cho Epic 20

Epic 20 được xem là hoàn thành khi:

- Feature set không vượt quá 30 features.
- Có HAR fallback.
- Có hard gate.
- Có optional soft gate.
- Có clipping hoặc shrinkage cho news adjustment.
- Chạy nested walk-forward.
- Không giảm mạnh trong low-vol regime.
- So sánh trực tiếp với HAR price-only.
- Báo cáo compute overhead.
- Có operational fallback nếu news pipeline lỗi.

---

# 12. Acceptance criteria cho Epic 21

Epic 21 được xem là hoàn thành khi:

- Chạy đủ T+5, T+10 và T+22.
- Có bảng all/high/low-vol theo horizon.
- Có volatility spike target.
- Có abnormal volatility target.
- Có PR-AUC và calibration cho classification.
- Có QLIKE và ΔR² cho regression.
- Không dùng threshold tính từ toàn bộ dataset.
- Có kiểm tra target overlap và leakage.

---

# 13. Acceptance criteria cho Epic 22

Epic 22 được xem là hoàn thành khi có quyết định cuối cùng cho từng trường hợp:

| Trường hợp | Quyết định |
|---|---|
| T+1 all regimes | Keep / Drop |
| T+1 high-vol | Keep / Drop |
| T+5 high-vol | Keep / Drop |
| T+10 high-vol | Keep / Drop |
| T+22 high-vol | Keep / Drop |
| Sensitive tickers | Keep / Drop |
| Volatility spike | Keep / Drop |
| Production deployment | Yes / No |

Báo cáo cuối phải phân biệt:

- Statistical significance.
- Predictive significance.
- Economic significance.
- Operational value.

---

# 14. Thứ tự triển khai khuyến nghị

## Bước 1 — Xác minh high-vol signal

Ưu tiên cao nhất:

```text
Rolling regime thresholds
High-vol subset evaluation
Block bootstrap
DM test
Ticker stability
Placebo tests
```

## Bước 2 — Xây conditional model tối giản

```text
HAR fallback
10–30 news features
High-vol hard gate
Shrinkage
Walk-forward OOS
```

## Bước 3 — Mở rộng horizon

```text
T+5
T+10
T+22
```

## Bước 4 — Đổi target

```text
Volatility spike
Abnormal volatility
Two-stage prediction
```

## Bước 5 — Đưa ra quyết định dừng hoặc giữ

Không phát triển thêm model phức tạp trước khi hoàn thành các bước trên.

---

# 15. Khuyến nghị cuối cùng

Các Epic 16–18 đã cho thấy:

- News không phải tín hiệu tổng quát mạnh cho `pk_t+1`.
- Feature explosion gây overfitting.
- EWMA dài hạn làm mất signal.
- HAR residual không tạo incremental value ổn định.
- Mixture-of-Experts không giải quyết được vấn đề.
- Signal đáng chú ý nhất chỉ xuất hiện trong high-volatility regime.

Do đó, hướng tiếp theo nên là:

1. Xác nhận `high-vol regime ΔR² = +0.0012` có thực sự ổn định.
2. Xây `HAR fallback + high-vol news gate`.
3. Giảm feature set xuống 10–30 features.
4. Chạy lại cho T+5, T+10 và T+22.
5. Thử volatility spike và abnormal volatility.
6. Dừng hướng news cho T+1 nếu không vượt qua bootstrap, placebo và stability tests.

Kết luận nghiên cứu có thể là:

> Tin tức tiếng Việt không tạo giá trị dự báo ổn định cho Parkinson volatility trong toàn bộ thị trường. Tuy nhiên, một lượng incremental signal nhỏ có thể xuất hiện trong high-volatility regimes, ở một số ticker và horizon cụ thể. Signal này cần được xác nhận bằng nested walk-forward validation, block bootstrap và placebo testing trước khi cân nhắc sử dụng trong production.
