# Hướng cải thiện tiếp theo sau EDA tin tức và biến động cổ phiếu

## 1. Mục tiêu

Kết quả EDA hiện tại cho thấy:

- Tin tức không cải thiện rõ dự báo biến động ở T+1 và T+5.
- Tín hiệu có ý nghĩa thống kê xuất hiện ở T+10 nhưng mức cải thiện rất nhỏ.
- T+22 chưa có bằng chứng rõ ràng về đóng góp của news features.
- GBM hiện tại gần như không sử dụng các news features.
- Tín hiệu tốt nhất đến từ PhoBERT embedding kết hợp với trạng thái biến động thị trường.
- Chỉ khoảng 25% ticker thể hiện độ nhạy đáng kể với tin tức.

Vì vậy, hướng phát triển tiếp theo không nên bắt đầu bằng việc thay Ridge hoặc GBM bằng mô hình deep learning lớn. Giá trị cao nhất nằm ở:

1. Cải thiện cách biểu diễn nội dung tin tức.
2. Cải thiện aggregation theo thời gian.
3. Tách incremental contribution của news khỏi price model.
4. Mô hình hóa sự khác biệt giữa các ticker.
5. Siết chặt quy trình đánh giá out-of-sample.

---

## 2. Kiểm tra lại ý nghĩa của `centroid norm`

PhoBERT embedding có thể không đẳng hướng. Vì vậy, độ lớn của vector centroid chưa chắc phản ánh trực tiếp “cường độ thông tin”.

Nên chạy ablation với các phiên bản sau:

| Phiên bản | Cách tính |
|---|---|
| Raw norm | `norm(mean(raw_embeddings))` |
| Normalized norm | L2-normalize từng bài rồi mới lấy centroid |
| Whitened norm | Whitening embedding trước khi tính centroid |
| PCA score | Dùng PC0–PC31 thay cho một scalar norm |
| Centered norm | Trừ global hoặc market centroid rồi mới tính norm |

### Normalized centroid

Với embedding bài báo \(e_i\):

\[
\tilde e_i = \frac{e_i}{\|e_i\|_2}
\]

Centroid trong ngày:

\[
C_t = \frac{1}{N_t}\sum_{i=1}^{N_t}\tilde e_i
\]

Semantic coherence:

\[
\text{coherence}_t = \|C_t\|_2
\]

Sau khi từng embedding đã được L2-normalize:

- Giá trị gần 1: các tin trong ngày tập trung vào cùng một hướng ngữ nghĩa.
- Giá trị thấp: nội dung tin phân tán, khác chủ đề hoặc mâu thuẫn.

Do đó, feature này nên được diễn giải là **news semantic coherence** thay vì mặc định là “news strength”.

---

## 3. Thêm `news novelty`

Một bài báo có embedding mạnh nhưng chỉ lặp lại thông tin cũ có thể không tạo thêm tác động đến thị trường.

Có thể đo độ mới của news centroid ngày \(t\) so với lịch sử:

\[
\text{novelty}_t =
1 -
\cos\left(
C_t,
\operatorname{EWMA}_{30}(C_{t-1})
\right)
\]

Diễn giải:

- Novelty thấp: tin mới tương tự thông tin đã xuất hiện.
- Novelty cao: xuất hiện sự kiện hoặc nội dung mới.
- Novelty cao kết hợp volatility cao có thể là tín hiệu mạnh hơn centroid norm.

Các feature nên thử:

```text
news_novelty_5d
news_novelty_20d
news_novelty_60d
news_novelty_x_hist_vol_20d
news_novelty_x_abnormal_volume
news_novelty_x_market_stress
```

---

## 4. Không chỉ dùng trung bình: đo dispersion và extreme news

Daily centroid có thể làm mất thông tin khi một ngày xuất hiện nhiều tin trái chiều.

Ví dụ:

- Một bài rất tích cực.
- Một bài rất tiêu cực.
- Vector trung bình có thể trở nên gần trung lập.

### 4.1 Semantic dispersion

\[
\text{dispersion}_t =
\frac{1}{N_t}
\sum_i
\left(
1 - \cos(e_{t,i}, C_t)
\right)
\]

Dispersion cao cho thấy các bài báo trong ngày thiếu đồng thuận về mặt ngữ nghĩa.

### 4.2 Maximum semantic shock

Các feature nên thử:

```text
max_distance_from_rolling_centroid
top1_novelty
top3_mean_novelty
maximum_event_severity
```

### 4.3 Agreement và concentration

```text
positive_negative_disagreement
embedding_directional_consistency
article_cluster_count
dominant_cluster_share
```

Một ngày có 10 bài cùng lặp lại một sự kiện khác hoàn toàn ngày có 10 bài thuộc nhiều sự kiện không liên quan, dù `news_count` giống nhau.

---

## 5. Thử nhiều tốc độ suy giảm thay vì chỉ EWMA 30 ngày

EWMA 30 ngày có thể quá chậm với một số sự kiện và quá nhanh với các sự kiện dài hạn.

Nên tạo nhiều cửa sổ:

```text
ewma_3d
ewma_5d
ewma_10d
ewma_20d
ewma_30d
ewma_60d
```

### News momentum

\[
\text{news\_momentum}_t =
\text{EWMA}_{5,t}
-
\text{EWMA}_{30,t}
\]

Các feature mở rộng:

```text
novelty_ewma_5d_minus_30d
coherence_ewma_5d_div_30d
pc0_ewma_5d_minus_20d
dispersion_ewma_5d_minus_30d
```

Các feature này giúp phân biệt:

- Tín hiệu mới tăng đột ngột.
- Chủ đề đã kéo dài nhiều tuần.
- Tín hiệu đang suy giảm.

Việc chọn window phải được thực hiện trong training fold, không chọn bằng toàn bộ dataset.

---

## 6. Trích xuất loại sự kiện thay vì chỉ dùng sentiment

Sentiment tích cực, tiêu cực hoặc trung lập quá tổng quát.

Hai tin đều tích cực nhưng có thể tạo tác động volatility rất khác nhau:

- Doanh thu tăng nhẹ.
- Công bố thương vụ M&A.
- Cổ đông lớn mua cổ phiếu.
- Phê duyệt dự án.
- Điều tra pháp lý.
- Thay đổi lãnh đạo.
- Phát hành thêm.
- Chia cổ tức.
- Cảnh báo thanh khoản.
- Thay đổi chính sách ngành.

Nên trích xuất structured event:

```json
{
  "ticker": "VIC",
  "event_type": "M&A",
  "direction": "positive",
  "status": "announced",
  "magnitude": "high",
  "certainty": 0.87,
  "time_horizon": "medium",
  "novelty": 0.74,
  "ticker_relevance": 0.92
}
```

Các feature có thể tạo:

```text
ma_event_count_20d
legal_event_severity_30d
earnings_surprise_event
event_novelty_x_hist_vol_20d
event_severity_x_ticker_sensitivity
event_certainty_x_relevance
```

LLM có thể dùng để gán nhãn offline, nhưng nên:

- Lưu confidence.
- Kiểm tra thủ công một tập mẫu.
- Định nghĩa taxonomy sự kiện cố định.
- Version prompt và model.
- Không để LLM nhìn target hoặc dữ liệu tương lai.

---

## 7. Thêm `ticker relevance`

Không phải mọi bài báo nhắc đến ticker đều có mức liên quan như nhau.

Một bài có thể:

- Chỉ nhắc tên doanh nghiệp trong một câu.
- Tập trung hoàn toàn vào doanh nghiệp.
- Nói về toàn ngành.
- Nói về công ty mẹ, công ty con hoặc đối tác.
- Sao chép lại tin từ nguồn khác.

Có thể tạo weighted centroid:

\[
C_t =
\frac{\sum_i w_i e_i}
{\sum_i w_i}
\]

Trong đó:

\[
w_i =
\text{ticker relevance}
\times
\text{source quality}
\times
\text{novelty}
\times
\text{time decay}
\]

Feature tối thiểu nên có:

```text
ticker_mention_count
ticker_mention_density
headline_mentions_ticker
primary_subject_flag
market_wide_news_flag
sector_news_flag
duplicate_cluster_size
source_quality_score
```

---

## 8. Deduplicate và cluster các bài cùng sự kiện

News volume hiện tại có thể bị nhiễu do:

- Nhiều báo đăng lại cùng một thông tin.
- Một nguồn cập nhật lại bài cũ.
- Headline khác nhau nhưng nội dung gần giống nhau.
- Một sự kiện được chia thành nhiều bài ngắn.

Quy trình đề xuất:

1. L2-normalize embedding.
2. Tính cosine similarity.
3. Cluster theo ticker và time window.
4. Gộp các bài có similarity cao thành một event cluster.
5. Chọn bài đại diện hoặc weighted centroid cho cluster.
6. Dùng số event cluster thay cho raw news count.

Các feature:

```text
raw_article_count
unique_event_count
duplicate_ratio
largest_cluster_size
dominant_event_share
event_cluster_entropy
```

---

## 9. Dự báo residual thay vì dự báo trực tiếp volatility

HAR price-only đã giải thích phần lớn tín hiệu. News model nên tập trung dự báo phần còn lại.

### Bước 1: Price-only model

\[
\hat y^{price}_{t+h}
=
HAR(X^{price}_t)
\]

### Bước 2: Residual target

\[
r_{t+h}
=
y_{t+h}
-
\hat y^{price}_{t+h}
\]

Lưu ý: residual phải được tạo bằng out-of-fold prediction trên training set để tránh leakage.

### Bước 3: News residual model

\[
\hat r_{t+h}
=
f(
X^{news}_t,
X^{interaction}_t
)
\]

### Bước 4: Kết hợp

\[
\hat y_{t+h}
=
\hat y^{price}_{t+h}
+
\hat r_{t+h}
\]

News branch lúc này trả lời:

> Tin tức làm volatility cao hoặc thấp hơn mức price-only model dự báo bao nhiêu?

Đây là thử nghiệm ưu tiên cao vì phù hợp với kết luận EDA: market features mạnh, news chỉ có incremental signal nhỏ.

### Log-volatility target

Có thể thử:

\[
z_{t+h}
=
\log(y_{t+h} + \epsilon)
\]

Lợi ích:

- Giảm ảnh hưởng của outlier.
- Tránh dự báo volatility âm sau khi inverse transform.
- Ổn định variance tốt hơn.

---

## 10. Dùng news gate

Vì chỉ khoảng 25% ticker nhạy với tin tức, không nên luôn ép news branch đóng góp.

Gate:

\[
g_t =
\sigma\left(
f(
\text{ticker},
\text{novelty},
\text{event type},
\text{market regime}
)
\right)
\]

Dự báo cuối:

\[
\hat y_t =
\hat y^{price}_t
+
g_t \cdot \Delta^{news}_t
\]

Diễn giải:

- \(g_t \approx 0\): bỏ qua news adjustment.
- \(g_t \approx 1\): kích hoạt news adjustment.
- \(\Delta^{news}_t\): mức điều chỉnh từ news model.

Nên bắt đầu với:

1. Logistic Regression gate.
2. LightGBM gate.
3. Soft gate bằng xác suất.
4. Mixture-of-Experts chỉ sau khi gate đơn giản chứng minh được giá trị.

---

## 11. Học hệ số theo ticker nhưng vẫn chia sẻ thông tin

Không nên huấn luyện 30 model độc lập nếu dữ liệu chỉ có khoảng hai năm.

Thứ tự thử nghiệm:

1. Global model.
2. Global model + ticker one-hot.
3. Global model + ticker embedding.
4. Global model + ticker-news interactions.
5. Hierarchical model.
6. Cluster-specific model.
7. Mixture-of-Experts.

Ví dụ ticker-specific interaction:

\[
y =
\beta X_{news}
+
\sum_k
\gamma_k
\left(
I_{\text{ticker}=k} X_{news}
\right)
\]

Regularization giúp \(\gamma_k\) co về 0 nếu ticker không có đủ bằng chứng.

Có thể cluster ticker theo:

```text
news_sensitivity
sector
market_cap
liquidity
average_volatility
foreign_ownership
news_frequency
```

---

## 12. Tách market, sector và company-specific news

Các feature như:

```text
Tong_hop PC0 x index_vol_20d
Khach_quan PC0 x index_vol_20d
```

có thể đang phản ánh tin thị trường chung thay vì tin riêng của ticker.

Nên xây ba nhánh:

```text
Market-wide news embedding
    -> market news factors

Sector news embedding
    -> sector news factors

Ticker-specific news embedding
    -> idiosyncratic news factors
```

Có thể residualize ticker news theo market news:

\[
C^{idio}_{i,t}
=
C_{i,t}
-
\hat B_i C^{market}_t
\]

Mục tiêu là tách:

- Tin vĩ mô ảnh hưởng toàn thị trường.
- Tin ngành ảnh hưởng một nhóm ticker.
- Tin riêng của doanh nghiệp.
- Hiệu ứng lan truyền giữa các ticker liên quan.

---

## 13. Thêm market regime

News effect có thể chỉ xuất hiện trong một số regime nhất định.

Các regime nên thử:

```text
low_volatility
normal_volatility
high_volatility
market_uptrend
market_downtrend
high_liquidity
low_liquidity
event_stress
```

Feature interaction:

```text
news_novelty_x_high_vol_regime
event_severity_x_market_downtrend
ticker_news_coherence_x_index_vol_20d
news_dispersion_x_low_liquidity
```

Regime phải được xác định chỉ từ dữ liệu có sẵn tại thời điểm dự báo.

---

## 14. Fine-tune PhoBERT theo domain tài chính

Không nên fine-tune trực tiếp theo target volatility ngay từ đầu vì:

- Dataset nhỏ.
- Target nhiễu.
- Nguy cơ overfitting cao.
- Khó xác định model học ngữ nghĩa hay học pattern thời gian.

Lộ trình phù hợp:

1. Frozen PhoBERT.
2. L2 normalization.
3. Whitening hoặc PCA.
4. Linear probe.
5. Event classification.
6. Ticker relevance classification.
7. Contrastive fine-tuning.
8. Cuối cùng mới thử end-to-end volatility forecasting.

### Contrastive learning

Positive pairs:

- Các bài cùng event cluster.
- Các bài cùng ticker và cùng sự kiện.
- Bản tin gốc và bản đăng lại.

Negative pairs:

- Khác ticker và khác sự kiện.
- Cùng ticker nhưng khác loại sự kiện.
- Tin thị trường chung và tin doanh nghiệp riêng.

Không nên coi mọi bài cùng ticker là positive pair.

---

## 15. Multi-task learning theo horizon

T+5, T+10 và T+22 có thể chia sẻ representation nhưng khác head dự báo.

Kiến trúc:

```text
Shared market encoder
Shared news encoder
        |
        +--> Head T+5
        +--> Head T+10
        +--> Head T+22
```

Loss:

\[
L =
\lambda_5 L_{T+5}
+
\lambda_{10} L_{T+10}
+
\lambda_{22} L_{T+22}
\]

Có thể đặt trọng số cao hơn cho T+10 và T+22 vì EDA cho thấy news signal có xu hướng xuất hiện ở horizon dài hơn.

Chỉ nên thử multi-task sau khi feature pipeline đơn giản chứng minh được incremental signal ổn định.

---

## 16. Siết chặt quy trình evaluation

Vì \(\Delta R^2\) rất nhỏ, pipeline đánh giá phải đủ nghiêm ngặt để tránh chọn nhầm noise.

### 16.1 Walk-forward evaluation

Không dùng random split.

Ví dụ:

```text
Fold 1: Train 01-06, Validate 07, Test 08
Fold 2: Train 01-07, Validate 08, Test 09
Fold 3: Train 01-08, Validate 09, Test 10
```

### 16.2 Mọi transformation phải fit trong training fold

Bao gồm:

- Scaler.
- PCA.
- Whitening.
- Feature selection.
- Hyperparameter selection.
- Event taxonomy mapping nếu có học tham số.
- Imputation.
- Residual model.
- EWMA initialization nếu dùng state từ lịch sử.

### 16.3 Out-of-fold residual

Không được dùng fitted prediction trên chính training rows để tạo residual cho news model.

Phải tạo:

```text
price_model OOF prediction
    -> OOF residual
    -> train news residual model
```

### 16.4 Multiple-testing correction

Khi thử hàng trăm feature, một số feature có thể có p-value thấp do ngẫu nhiên.

Nên dùng:

- Benjamini-Hochberg FDR.
- Holm correction.
- Block permutation test.
- Stability selection.

### 16.5 Kiểm tra stability

Một feature chỉ nên được giữ nếu:

- Dấu hệ số ổn định qua các fold.
- Feature importance không biến mất hoàn toàn.
- Cải thiện xuất hiện trên nhiều giai đoạn.
- Không chỉ phụ thuộc vào một ticker.
- Không chỉ phụ thuộc vào một vài ngày extreme.

---

## 17. Bộ metric nên báo cáo

Không nên chỉ dùng \(R^2\).

```text
R2
Delta R2
MAE
RMSE
QLIKE
Spearman correlation
DM p-value
Bootstrap confidence interval
Per-ticker win rate
Per-period win rate
Coefficient stability
Feature selection frequency
```

### QLIKE

QLIKE thường phù hợp cho volatility forecasting vì đánh giá sai số theo tỷ lệ và phạt mạnh dự báo volatility không phù hợp.

Cần bảo đảm prediction và target dương trước khi tính QLIKE.

---

## 18. Kiểm tra leakage

Các điểm phải kiểm tra:

- News timestamp có sau market close không.
- Tin đăng cuối ngày có bị dùng để dự báo cùng ngày không.
- Target T+10 hoặc T+22 có overlap mạnh giữa các sample không.
- PCA có fit trên toàn dataset không.
- Feature selection có nhìn test period không.
- Duplicate article có xuất hiện ở cả train và test không.
- Rolling/EWMA có dùng future rows không.
- Residual target có được tạo từ in-sample fitted prediction không.
- Ticker sensitivity có được tính bằng toàn bộ thời gian không.
- Event label do LLM tạo có vô tình chứa thông tin tương lai không.

---

## 19. Thứ tự triển khai đề xuất

### Ưu tiên 1 — chi phí thấp, giá trị cao

1. L2-normalize embedding.
2. Kiểm tra whitening và PCA.
3. Thêm novelty.
4. Thêm dispersion.
5. Thêm maximum semantic shock.
6. Thử EWMA 5/10/20/30/60.
7. Thêm `no_news_mask`.
8. Dự báo HAR residual.
9. Đánh giá theo ticker.
10. Đánh giá theo volatility regime.

### Ưu tiên 2 — nâng chất lượng nội dung

11. Deduplicate bài cùng sự kiện.
12. Cluster event.
13. Tính ticker relevance.
14. Tách market, sector và company news.
15. Trích xuất event type.
16. Trích xuất event severity, certainty và status.
17. Thêm source quality.

### Ưu tiên 3 — mô hình nâng cao

18. Ticker-specific coefficients.
19. News gate.
20. Hierarchical model.
21. Cluster experts.
22. Multi-task T+5/T+10/T+22.
23. Mixture-of-Experts.
24. Domain fine-tuning cho PhoBERT.
25. End-to-end deep fusion nếu các bước trước đã chứng minh được signal.

---

## 20. Bộ thí nghiệm tiếp theo

| Experiment | Price branch | News branch | Mục đích |
|---|---|---|---|
| E1 | HAR | Raw centroid norm | Baseline hiện tại |
| E2 | HAR | Normalized centroid coherence | Kiểm tra embedding geometry |
| E3 | HAR | Whitened/PCA embedding | Giảm anisotropy |
| E4 | HAR | Novelty + dispersion | Kiểm tra semantic shock |
| E5 | HAR | Deduplicated event features | Loại ảnh hưởng đăng lại |
| E6 | HAR | Event + ticker relevance | Kiểm tra structured news |
| E7 | HAR | News dự báo HAR residual | Tách incremental contribution |
| E8 | HAR | Residual + ticker interactions | Kiểm tra heterogeneity |
| E9 | HAR | Residual + regime gate | Chỉ dùng news khi phù hợp |
| E10 | HAR | Market/sector/ticker branches | Tách loại news signal |
| E11 | HAR | Multi-task horizons | Chia sẻ representation |
| E12 | HAR | Mixture-of-Experts | Mô hình cuối nếu E7–E10 tốt |

---

## 21. Acceptance criteria cho news features

Một nhóm news features chỉ nên được giữ nếu đáp ứng phần lớn các điều kiện sau:

- Cải thiện out-of-sample trên ít nhất hai metric.
- \(\Delta R^2 > 0\) ổn định qua nhiều fold.
- DM test có ý nghĩa sau điều chỉnh multiple testing.
- Bootstrap confidence interval không chứa 0 hoặc nghiêng rõ về phía cải thiện.
- Cải thiện không chỉ đến từ một ticker.
- Có dấu hệ số ổn định.
- Không bị permutation importance đưa về 0.
- Không phụ thuộc vào một vài ngày extreme.
- Có giải thích kinh tế hợp lý.
- Pipeline không có leakage.
- Kết quả tái lập được với random seed khác.
- Cải thiện đủ lớn để bù chi phí vận hành feature pipeline.

---

## 22. Khuyến nghị chính

Bước tiếp theo tốt nhất không phải là chuyển ngay sang LSTM, Transformer hoặc một kiến trúc deep learning phức tạp.

Pipeline nên ưu tiên:

```text
PhoBERT embedding
    -> L2 normalization / whitening
    -> deduplication và event clustering
    -> novelty + dispersion + coherence
    -> EWMA multi-scale
    -> interaction với volatility regime
    -> HAR residual prediction
    -> ticker-specific regularization
    -> strict walk-forward evaluation
```

Giả thuyết cần kiểm chứng là:

> Tin tức không trực tiếp dự báo phần lớn volatility, nhưng có thể giải thích một phần nhỏ residual của price-only model khi nội dung đủ mới, đủ liên quan và xuất hiện trong đúng ticker hoặc market regime.

Chỉ nên triển khai news gate, hierarchical model, Mixture-of-Experts hoặc domain fine-tuning sau khi pipeline feature đơn giản tạo ra cải thiện out-of-sample ổn định và có ý nghĩa thực tế.
