# Epic 20 — Conditional Lightweight Model


Reduced feature set (13 features): ['news_count_1d', 'news_count_3d', 'days_since_last_news', 'kq_emb_0', 'th_emb_0', 'kq_emb_norm', 'th_emb_norm', 'kq_novelty_30d', 'th_novelty_30d', 'kq_dispersion', 'th_dispersion', 'kq_max_semantic_shock', 'th_max_semantic_shock']


## hard gate — pk_t+1

- All: R²=0.278805, RMSE=0.00049108
- High-vol (n=40): R²=-0.112608
- Low-vol  (n=21088): R²=0.277837
- HAR baseline: R²=0.278804, RMSE=0.00049108
- ΔR² all: 1e-06
- ΔR² high-vol: 0.0

## hard gate — pk_t+5

- All: R²=0.173202, RMSE=0.00052786
- High-vol (n=40): R²=-0.274008
- Low-vol  (n=20848): R²=0.173168
- HAR baseline: R²=0.173202, RMSE=0.00052786
- ΔR² all: 0.0
- ΔR² high-vol: 0.0

## hard gate — pk_t+10

- All: R²=0.121657, RMSE=0.00054646
- High-vol (n=40): R²=-1.520241
- Low-vol  (n=20548): R²=0.123178
- HAR baseline: R²=0.121657, RMSE=0.00054646
- ΔR² all: 0.0
- ΔR² high-vol: -8e-06

## hard gate — pk_t+22

- All: R²=0.055157, RMSE=0.0005738
- High-vol (n=40): R²=-1.282758
- Low-vol  (n=19828): R²=0.056561
- HAR baseline: R²=0.055157, RMSE=0.0005738
- ΔR² all: 0.0
- ΔR² high-vol: -0.000251

## soft gate — pk_t+1

- All: R²=0.278805, RMSE=0.00049108
- High-vol (n=40): R²=-0.112608
- Low-vol  (n=21088): R²=0.277837
- HAR baseline: R²=0.278804, RMSE=0.00049108
- ΔR² all: 1e-06
- ΔR² high-vol: 0.0

## soft gate — pk_t+5

- All: R²=0.173202, RMSE=0.00052786
- High-vol (n=40): R²=-0.274008
- Low-vol  (n=20848): R²=0.173168
- HAR baseline: R²=0.173202, RMSE=0.00052786
- ΔR² all: 0.0
- ΔR² high-vol: 0.0

## soft gate — pk_t+10

- All: R²=0.121657, RMSE=0.00054646
- High-vol (n=40): R²=-1.520241
- Low-vol  (n=20548): R²=0.123178
- HAR baseline: R²=0.121657, RMSE=0.00054646
- ΔR² all: 0.0
- ΔR² high-vol: -8e-06

## soft gate — pk_t+22

- All: R²=0.055159, RMSE=0.0005738
- High-vol (n=40): R²=-1.282739
- Low-vol  (n=19828): R²=0.056563
- HAR baseline: R²=0.055157, RMSE=0.0005738
- ΔR² all: 2e-06
- ΔR² high-vol: -0.000232