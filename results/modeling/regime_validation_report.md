# Epic 19 — Regime-Conditional News Validation


## 19-1: Ex-ante Volatility Regimes

- Threshold 60%: 23896/201230 high-vol observations
- Threshold 70%: 14170/201230 high-vol observations
- Threshold 80%: 7630/201230 high-vol observations
- Threshold 90%: 3778/201230 high-vol observations

## 19-2: Threshold Sensitivity


### pk_t+1

| Threshold | Regime | n | price_r2 | news_r2 | delta_r2 |
|-----------|--------|---|----------|---------|----------|
| 60% | high   | 2348 | 0.199257 | 0.201479 | +0.002223 |
| 60% | low    | 18780 | 0.238261 | 0.238468 | +0.000207 |
| 60% | all    | 21128 | 0.279150 | 0.279886 | +0.000736 |
| 70% | high   | 1244 | 0.230089 | 0.232765 | +0.002676 |
| 70% | low    | 19884 | 0.244036 | 0.244469 | +0.000433 |
| 70% | all    | 21128 | 0.279150 | 0.279886 | +0.000736 |
| 80% | high   | 466 | 0.158941 | 0.159913 | +0.000972 |
| 80% | low    | 20662 | 0.252312 | 0.253066 | +0.000754 |
| 80% | all    | 21128 | 0.279150 | 0.279886 | +0.000736 |
| 90% | high   | 176 | 0.034427 | 0.034476 | +0.000049 |
| 90% | low    | 20952 | 0.264760 | 0.265538 | +0.000778 |
| 90% | all    | 21128 | 0.279150 | 0.279886 | +0.000736 |

### pk_t+5

| Threshold | Regime | n | price_r2 | news_r2 | delta_r2 |
|-----------|--------|---|----------|---------|----------|
| 60% | high   | 2332 | 0.172596 | 0.173825 | +0.001230 |
| 60% | low    | 18556 | 0.133753 | 0.134469 | +0.000716 |
| 60% | all    | 20888 | 0.174816 | 0.175623 | +0.000807 |
| 70% | high   | 1238 | 0.154212 | 0.156590 | +0.002378 |
| 70% | low    | 19650 | 0.148566 | 0.149157 | +0.000591 |
| 70% | all    | 20888 | 0.174816 | 0.175623 | +0.000807 |
| 80% | high   | 466 | 0.040650 | 0.042476 | +0.001827 |
| 80% | low    | 20422 | 0.162411 | 0.163175 | +0.000764 |
| 80% | all    | 20888 | 0.174816 | 0.175623 | +0.000807 |
| 90% | high   | 176 | -0.095705 | -0.089206 | +0.006500 |
| 90% | low    | 20712 | 0.171492 | 0.172224 | +0.000733 |
| 90% | all    | 20888 | 0.174816 | 0.175623 | +0.000807 |

### pk_t+10

| Threshold | Regime | n | price_r2 | news_r2 | delta_r2 |
|-----------|--------|---|----------|---------|----------|
| 60% | high   | 2312 | 0.154247 | 0.154395 | +0.000148 |
| 60% | low    | 18276 | 0.092578 | 0.091631 | -0.000947 |
| 60% | all    | 20588 | 0.122894 | 0.122187 | -0.000707 |
| 70% | high   | 1234 | 0.138424 | 0.137517 | -0.000907 |
| 70% | low    | 19354 | 0.107876 | 0.107184 | -0.000692 |
| 70% | all    | 20588 | 0.122894 | 0.122187 | -0.000707 |
| 80% | high   | 466 | 0.038912 | 0.040020 | +0.001108 |
| 80% | low    | 20122 | 0.116262 | 0.115466 | -0.000796 |
| 80% | all    | 20588 | 0.122894 | 0.122187 | -0.000707 |
| 90% | high   | 176 | -0.149845 | -0.148516 | +0.001329 |
| 90% | low    | 20412 | 0.122414 | 0.121666 | -0.000748 |
| 90% | all    | 20588 | 0.122894 | 0.122187 | -0.000707 |

### pk_t+22

| Threshold | Regime | n | price_r2 | news_r2 | delta_r2 |
|-----------|--------|---|----------|---------|----------|
| 60% | high   | 2276 | 0.136720 | 0.137867 | +0.001147 |
| 60% | low    | 17592 | 0.028483 | 0.029373 | +0.000890 |
| 60% | all    | 19868 | 0.055000 | 0.055928 | +0.000928 |
| 70% | high   | 1222 | 0.120936 | 0.123211 | +0.002275 |
| 70% | low    | 18646 | 0.038014 | 0.038783 | +0.000768 |
| 70% | all    | 19868 | 0.055000 | 0.055928 | +0.000928 |
| 80% | high   | 466 | 0.055860 | 0.057031 | +0.001171 |
| 80% | low    | 19402 | 0.045092 | 0.046014 | +0.000921 |
| 80% | all    | 19868 | 0.055000 | 0.055928 | +0.000928 |
| 90% | high   | 176 | 0.024569 | 0.026467 | +0.001898 |
| 90% | low    | 19692 | 0.050786 | 0.051687 | +0.000901 |
| 90% | all    | 19868 | 0.055000 | 0.055928 | +0.000928 |

## 19-3: High-Vol Subset Model


### pk_t+1

- n_high=7630, train=7164, test=466
- Price-only R²=0.119184
- Price+news R²=0.072576
- ΔR²=-0.046608

### pk_t+5

- n_high=7630, train=7164, test=466
- Price-only R²=0.018484
- Price+news R²=-0.026629
- ΔR²=-0.045113

### pk_t+10

- n_high=7630, train=7164, test=466
- Price-only R²=0.062229
- Price+news R²=-0.051188
- ΔR²=-0.113417

### pk_t+22

- n_high=7630, train=7164, test=466
- Price-only R²=0.051521
- Price+news R²=-0.066733
- ΔR²=-0.118254

## 19-4: Sensitive × High-Vol Cross Analysis


### pk_t+1

- sensitive       / high   (n=184): ΔR²=+0.001150
- sensitive       / low    (n=2632): ΔR²=+0.000778
- non_sensitive   / high   (n=282): ΔR²=+0.000871
- non_sensitive   / low    (n=18030): ΔR²=+0.000753

### pk_t+5

- sensitive       / high   (n=184): ΔR²=+0.003269
- sensitive       / low    (n=2600): ΔR²=+0.000738
- non_sensitive   / high   (n=282): ΔR²=+0.000702
- non_sensitive   / low    (n=17822): ΔR²=+0.000774

### pk_t+10

- sensitive       / high   (n=184): ΔR²=+0.003449
- sensitive       / low    (n=2560): ΔR²=+0.001736
- non_sensitive   / high   (n=282): ΔR²=-0.001756
- non_sensitive   / low    (n=17562): ΔR²=-0.001400

### pk_t+22

- sensitive       / high   (n=184): ΔR²=+0.000152
- sensitive       / low    (n=2464): ΔR²=+0.003956
- non_sensitive   / high   (n=282): ΔR²=+0.002813
- non_sensitive   / low    (n=16938): ΔR²=+0.000253

## 19-5: Walk-Forward Validation


### pk_t+1

- Folds with positive ΔR²: 1/3
  Fold 0: ΔR²=-0.206420 (n_test_high=5788)
  Fold 1: ΔR²=-0.027317 (n_test_high=4216)
  Fold 2: ΔR²=+0.000979 (n_test_high=1470)

### pk_t+5

- Folds with positive ΔR²: 0/3
  Fold 0: ΔR²=-0.122058 (n_test_high=5792)
  Fold 1: ΔR²=-0.018722 (n_test_high=4216)
  Fold 2: ΔR²=-0.001593 (n_test_high=1470)

### pk_t+10

- Folds with positive ΔR²: 0/3
  Fold 0: ΔR²=-2.008998 (n_test_high=5796)
  Fold 1: ΔR²=-0.212965 (n_test_high=4216)
  Fold 2: ΔR²=-0.002742 (n_test_high=1470)

### pk_t+22

- Folds with positive ΔR²: 0/3
  Fold 0: ΔR²=-0.416171 (n_test_high=5804)
  Fold 1: ΔR²=-0.031244 (n_test_high=4216)
  Fold 2: ΔR²=-0.002399 (n_test_high=1474)

## 19-6: Statistical Tests + Placebos


### pk_t+1

- DM stat=1.1026, p=0.2702
- Block-bootstrap 95% CI: [-0.001643, 0.003061]
- n_high_vol_test=466

  Placebo tests:
  - placebo_block_shuffle: ΔR²=-0.000161
  - placebo_time_shift_-10d: ΔR²=-0.001111
  - placebo_time_shift_-5d: ΔR²=0.000734
  - placebo_time_shift_5d: ΔR²=0.000426
  - placebo_time_shift_10d: ΔR²=0.000477

### pk_t+5

- DM stat=1.6709, p=0.0947
- Block-bootstrap 95% CI: [-0.001253, 0.005503]
- n_high_vol_test=466

  Placebo tests:
  - placebo_block_shuffle: ΔR²=-0.000108
  - placebo_time_shift_-10d: ΔR²=-0.001856
  - placebo_time_shift_-5d: ΔR²=0.001348
  - placebo_time_shift_5d: ΔR²=0.00173
  - placebo_time_shift_10d: ΔR²=0.001346

### pk_t+10

- DM stat=0.8111, p=0.4173
- Block-bootstrap 95% CI: [-0.004215, 0.005806]
- n_high_vol_test=466

  Placebo tests:
  - placebo_block_shuffle: ΔR²=-0.000655
  - placebo_time_shift_-10d: ΔR²=-7e-05
  - placebo_time_shift_-5d: ΔR²=0.002809
  - placebo_time_shift_5d: ΔR²=0.004175
  - placebo_time_shift_10d: ΔR²=0.001325

### pk_t+22

- DM stat=1.0466, p=0.2953
- Block-bootstrap 95% CI: [-0.002631, 0.004685]
- n_high_vol_test=466

  Placebo tests:
  - placebo_block_shuffle: ΔR²=-0.001031
  - placebo_time_shift_-10d: ΔR²=0.002273
  - placebo_time_shift_-5d: ΔR²=-0.000666
  - placebo_time_shift_5d: ΔR²=0.000441
  - placebo_time_shift_10d: ΔR²=-0.00144

## 19-7: Keep/Drop Decision

- pk_t+1: **DROP** (score=0/6, ΔR²=0)
- pk_t+5: **DROP** (score=0/6, ΔR²=0)
- pk_t+10: **DROP** (score=0/6, ΔR²=0)
- pk_t+22: **DROP** (score=0/6, ΔR²=0)

### Decision rationale

- pk_t+1: 1/3 folds positive, DM test NOT significant, Bootstrap CI includes zero, Placebo better than real
- pk_t+5: DM test NOT significant, Bootstrap CI includes zero, Placebo better than real
- pk_t+10: DM test NOT significant, Bootstrap CI includes zero, Placebo better than real
- pk_t+22: DM test NOT significant, Bootstrap CI includes zero, Placebo better than real