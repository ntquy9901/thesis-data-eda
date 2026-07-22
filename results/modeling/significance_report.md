# Statistical Significance — News Contribution


## Diebold-Mariano + bootstrap (price vs price+news_adv, Ridge)

- **pk_t+1**: DM p=0.0046 → significant; ΔR² 95% CI [0.000179, 0.000892]
- **pk_t+5**: DM p=0.0001 → significant; ΔR² 95% CI [0.000461, 0.001399]
- **pk_t+10**: DM p=0.0004 → significant; ΔR² 95% CI [0.0007, 0.002302]
- **pk_t+22**: DM p=0.0051 → significant; ΔR² 95% CI [0.000334, 0.001756]

## Per-ticker heterogeneity (ΔR² = +news_adv − price; >0 = news helps)

- **pk_t+1**: news helps in 2/30 tickers; ΔR² median=-0.0116, max=0.0043
- **pk_t+5**: news helps in 4/30 tickers; ΔR² median=-0.0249, max=0.0106
- **pk_t+10**: news helps in 4/30 tickers; ΔR² median=-0.0328, max=0.0244
- **pk_t+22**: news helps in 5/30 tickers; ΔR² median=-0.0382, max=0.0439

## Event abnormal-volatility t-test (mean ≠ 0?)

- horizon 1: mean abnormal vol=-5e-06, p=0.858 → not significant
- horizon 5: mean abnormal vol=1.7e-05, p=0.3498 → not significant
- horizon 10: mean abnormal vol=1.8e-05, p=0.2714 → not significant

## Per-family ablation (Level-1 guideline: sentiment / event-type, Ridge)


### price+sentiment5
- **pk_t+1**: DM p=0.0016 → significant; ΔR² 95% CI [3.7e-05, 0.000165]
- **pk_t+5**: DM p=0.8119 → NOT significant; ΔR² 95% CI [-0.000153, 0.000227]
- **pk_t+10**: DM p=0.9843 → NOT significant; ΔR² 95% CI [-0.000139, 0.000174]
- **pk_t+22**: DM p=0.0702 → NOT significant; ΔR² 95% CI [-0.000535, -1e-05]

### price+event_type
- **pk_t+1**: DM p=0.2198 → NOT significant; ΔR² 95% CI [-0.000199, 3.6e-05]
- **pk_t+5**: DM p=0.026 → significant; ΔR² 95% CI [-0.000739, -8e-05]
- **pk_t+10**: DM p=0.4713 → NOT significant; ΔR² 95% CI [-0.000117, 0.000316]
- **pk_t+22**: DM p=0.0384 → significant; ΔR² 95% CI [-0.000478, -2.9e-05]

### price+sentiment5+event_type
- **pk_t+1**: DM p=0.7766 → NOT significant; ΔR² 95% CI [-0.000127, 0.000146]
- **pk_t+5**: DM p=0.0081 → significant; ΔR² 95% CI [-0.000777, -0.000138]
- **pk_t+10**: DM p=0.3267 → NOT significant; ΔR² 95% CI [-0.000127, 0.000514]
- **pk_t+22**: DM p=0.0043 → significant; ΔR² 95% CI [-0.000721, -0.000162]

## Per-family ablation (Level-1 guideline: sentiment / event-type, GBM — nonlinear, checks multicollinearity hypothesis)


### price+sentiment5
- **pk_t+1**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+5**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+10**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+22**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]

### price+event_type
- **pk_t+1**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+5**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+10**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+22**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]

### price+sentiment5+event_type
- **pk_t+1**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+5**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+10**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]
- **pk_t+22**: DM p=1.0 → NOT significant; ΔR² 95% CI [0.0, 0.0]