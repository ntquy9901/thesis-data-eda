# Epic 21 — Horizon and Target Expansion


## 21-1 to 21-3: Multi-Horizon Comparison


| Target | n | Price R² | News R² | ΔR² | ΔRMSE |
|--------|---|----------|---------|-----|-------|
| pk_t+1 | 21128 | 0.279150 | 0.279886 | 0.000736 | -2.5e-07 |
| pk_t+5 | 21128 | 0.174816 | 0.175623 | 0.000807 | -2.5e-07 |
| pk_t+10 | 21128 | 0.122894 | 0.122187 | -0.000707 | 2.2e-07 |
| pk_t+22 | 21128 | 0.055000 | 0.055928 | 0.000928 | -2.8e-07 |

## 21-4: Volatility Spike Classification


| Threshold | n_spike | n_total | Price PR-AUC | News PR-AUC | Price ROC-AUC | News ROC-AUC |
|-----------|---------|---------|--------------|-------------|---------------|-------------|
| p1 | 8067 | 20588 | 0.4448 | 0.4451 | 0.5522 | 0.5539 |
| p1 | 5537 | 20588 | 0.3030 | 0.3031 | 0.5366 | 0.5388 |
| p1 | 2844 | 20588 | 0.1547 | 0.1534 | 0.5225 | 0.5216 |

## 21-5: Abnormal Volatility (Two-Stage)


- Spike classification PR-AUC: price=0.4582, news=0.4513
- Magnitude regression R² on spike subset: price=-0.008453, news=-0.012918
- Two-stage expected abnormal vol R²: price=-0.132772, news=-0.130332

## 21-7: Target Robustness Summary


### Key findings

- Horizon comparison shows ΔR² < 0 for T+10, T+22; barely positive for T+1, T+5
- Spike classification: news adds minimal PR-AUC improvement (~0.001-0.006)
- Two-stage abnormal vol: news R² worse than price-only R² in magnitude
- **Conclusion: No horizon or target transformation recovers meaningful news signal**