# Modeling Comparison — News Contribution to Parkinson Vol

Split: train < 2025-01-01, test >= 2025-01-01. Models: ridge (linear HAR) + gbm (HistGradientBoosting). Feature sets: price / +news_basic / +news_adv. 30 tickers.

## Metrics (model × feature_set × target)

```
 target model                 feature_set  n_features     rmse      mae        r2        qlike  dir_acc
 pk_t+1 ridge                       price           6 0.000491 0.000279  0.279150 5.037380e-01   0.4801
 pk_t+1   gbm                       price           6 0.000523 0.000309  0.182471 5.959260e-01   0.4494
 pk_t+1 ridge            price+news_basic          11 0.000491 0.000280  0.279845 5.022670e-01   0.4767
 pk_t+1   gbm            price+news_basic          11 0.000523 0.000309  0.182471 5.959260e-01   0.4494
 pk_t+1 ridge              price+news_adv          50 0.000491 0.000280  0.279682 3.013509e+04   0.4763
 pk_t+1   gbm              price+news_adv          50 0.000523 0.000309  0.182471 5.959260e-01   0.4494
 pk_t+1 ridge         price+news_adv_dual          91 0.000491 0.000280  0.279886 5.263598e+04   0.4768
 pk_t+1   gbm         price+news_adv_dual          91 0.000523 0.000309  0.182638 5.958950e-01   0.4494
 pk_t+1 ridge  price+news_adv_dual_ewma30         157 0.000492 0.000272  0.276527 3.766138e+05   0.5142
 pk_t+1   gbm  price+news_adv_dual_ewma30         157 0.000523 0.000309  0.181188 5.959500e-01   0.4494
 pk_t+1 ridge         price+news_adv_full         493 0.000504 0.000278  0.239527 5.625813e+06   0.5255
 pk_t+1   gbm         price+news_adv_full         493 0.000524 0.000309  0.178147 5.965870e-01   0.4494
 pk_t+1 ridge      price+news_adv_novelty          97 0.000491 0.000280  0.279885 5.750564e+04   0.4770
 pk_t+1   gbm      price+news_adv_novelty          97 0.000523 0.000309  0.182638 5.958950e-01   0.4494
 pk_t+1 ridge   price+news_adv_multi_ewma         421 0.000504 0.000278  0.239858 5.473939e+06   0.5245
 pk_t+1   gbm   price+news_adv_multi_ewma         421 0.000524 0.000309  0.178147 5.965870e-01   0.4494
 pk_t+1 ridge            price+sentiment5          11 0.000491 0.000278  0.279254 5.051440e-01   0.4803
 pk_t+1   gbm            price+sentiment5          11 0.000523 0.000309  0.182471 5.959260e-01   0.4494
 pk_t+1 ridge            price+event_type          13 0.000491 0.000279  0.279075 5.257830e-01   0.4801
 pk_t+1   gbm            price+event_type          13 0.000523 0.000309  0.182471 5.959260e-01   0.4494
 pk_t+1 ridge price+sentiment5+event_type          18 0.000491 0.000279  0.279170 1.310147e+05   0.4804
 pk_t+1   gbm price+sentiment5+event_type          18 0.000523 0.000309  0.182471 5.959260e-01   0.4494
 pk_t+5 ridge                       price           6 0.000527 0.000304  0.174816 6.035540e-01   0.4684
 pk_t+5   gbm                       price           6 0.000547 0.000324  0.110725 6.503170e-01   0.4541
 pk_t+5 ridge            price+news_basic          11 0.000527 0.000306  0.175935 6.032930e-01   0.4666
 pk_t+5   gbm            price+news_basic          11 0.000548 0.000324  0.109532 6.507400e-01   0.4542
 pk_t+5 ridge              price+news_adv          50 0.000527 0.000305  0.175760 2.185139e+04   0.4670
 pk_t+5   gbm              price+news_adv          50 0.000548 0.000324  0.109532 6.507400e-01   0.4542
 pk_t+5 ridge         price+news_adv_dual          91 0.000527 0.000305  0.175623 2.486265e+04   0.4675
 pk_t+5   gbm         price+news_adv_dual          91 0.000548 0.000324  0.109517 6.507370e-01   0.4543
 pk_t+5 ridge  price+news_adv_dual_ewma30         157 0.000531 0.000293  0.163066 3.214265e+05   0.5064
 pk_t+5   gbm  price+news_adv_dual_ewma30         157 0.000549 0.000324  0.106889 6.525390e-01   0.4546
 pk_t+5 ridge         price+news_adv_full         493 0.000567 0.000310  0.047525 2.319271e+07   0.5403
 pk_t+5   gbm         price+news_adv_full         493 0.000549 0.000324  0.107047 6.524770e-01   0.4543
 pk_t+5 ridge      price+news_adv_novelty          97 0.000527 0.000305  0.175619 2.486265e+04   0.4674
 pk_t+5   gbm      price+news_adv_novelty          97 0.000548 0.000324  0.109517 6.507370e-01   0.4543
 pk_t+5 ridge   price+news_adv_multi_ewma         421 0.000566 0.000310  0.050548 2.271816e+07   0.5395
 pk_t+5   gbm   price+news_adv_multi_ewma         421 0.000549 0.000324  0.107047 6.524770e-01   0.4543
 pk_t+5 ridge            price+sentiment5          11 0.000527 0.000304  0.174839 6.035840e-01   0.4683
 pk_t+5   gbm            price+sentiment5          11 0.000547 0.000324  0.110725 6.503170e-01   0.4541
 pk_t+5 ridge            price+event_type          13 0.000527 0.000304  0.174462 6.041200e-01   0.4684
 pk_t+5   gbm            price+event_type          13 0.000547 0.000324  0.110725 6.503170e-01   0.4541
 pk_t+5 ridge price+sentiment5+event_type          18 0.000527 0.000304  0.174390 6.041470e-01   0.4682
 pk_t+5   gbm price+sentiment5+event_type          18 0.000547 0.000324  0.110725 6.503170e-01   0.4541
pk_t+10 ridge                       price           6 0.000546 0.000317  0.122894 6.493540e-01   0.4661
pk_t+10   gbm                       price           6 0.000559 0.000332  0.079850 6.765350e-01   0.4551
pk_t+10 ridge            price+news_basic          11 0.000546 0.000318  0.124390 6.469420e-01   0.4644
pk_t+10   gbm            price+news_basic          11 0.000559 0.000332  0.079850 6.765350e-01   0.4551
pk_t+10 ridge              price+news_adv          50 0.000546 0.000318  0.124326 6.469230e-01   0.4643
pk_t+10   gbm              price+news_adv          50 0.000559 0.000332  0.079800 6.766280e-01   0.4550
pk_t+10 ridge         price+news_adv_dual          91 0.000546 0.000318  0.122187 6.822230e-01   0.4648
pk_t+10   gbm         price+news_adv_dual          91 0.000559 0.000332  0.079923 6.764720e-01   0.4548
pk_t+10 ridge  price+news_adv_dual_ewma30         157 0.000552 0.000303  0.103299 7.026230e+05   0.5107
pk_t+10   gbm  price+news_adv_dual_ewma30         157 0.000560 0.000332  0.076435 6.783870e-01   0.4561
pk_t+10 ridge         price+news_adv_full         493 0.000613 0.000333 -0.104616 3.040378e+07   0.5468
pk_t+10   gbm         price+news_adv_full         493 0.000561 0.000332  0.074977 6.794580e-01   0.4568
pk_t+10 ridge      price+news_adv_novelty          97 0.000546 0.000318  0.122170 7.074480e-01   0.4648
pk_t+10   gbm      price+news_adv_novelty          97 0.000559 0.000332  0.079923 6.764720e-01   0.4548
pk_t+10 ridge   price+news_adv_multi_ewma         421 0.000612 0.000333 -0.100509 3.016982e+07   0.5463
pk_t+10   gbm   price+news_adv_multi_ewma         421 0.000561 0.000332  0.074977 6.794580e-01   0.4568
pk_t+10 ridge            price+sentiment5          11 0.000546 0.000317  0.122893 6.493130e-01   0.4662
pk_t+10   gbm            price+sentiment5          11 0.000559 0.000332  0.079850 6.765350e-01   0.4551
pk_t+10 ridge            price+event_type          13 0.000546 0.000317  0.122976 6.492790e-01   0.4663
pk_t+10   gbm            price+event_type          13 0.000559 0.000332  0.079850 6.765350e-01   0.4551
pk_t+10 ridge price+sentiment5+event_type          18 0.000546 0.000317  0.123053 6.491510e-01   0.4662
pk_t+10   gbm price+sentiment5+event_type          18 0.000559 0.000332  0.079850 6.765350e-01   0.4551
pk_t+22 ridge                       price           6 0.000574 0.000336  0.055000 7.185490e-01   0.4704
pk_t+22   gbm                       price           6 0.000580 0.000344  0.034531 7.154810e-01   0.4647
pk_t+22 ridge            price+news_basic          11 0.000573 0.000337  0.056333 7.163260e-01   0.4689
pk_t+22   gbm            price+news_basic          11 0.000580 0.000344  0.034672 7.153970e-01   0.4649
pk_t+22 ridge              price+news_adv          50 0.000574 0.000337  0.056052 7.195890e-01   0.4693
pk_t+22   gbm              price+news_adv          50 0.000580 0.000344  0.034671 7.153970e-01   0.4649
pk_t+22 ridge         price+news_adv_dual          91 0.000574 0.000337  0.055928 7.179550e-01   0.4682
pk_t+22   gbm         price+news_adv_dual          91 0.000580 0.000344  0.034671 7.154030e-01   0.4649
pk_t+22 ridge  price+news_adv_dual_ewma30         157 0.000585 0.000320  0.017436 3.151274e+06   0.5193
pk_t+22   gbm  price+news_adv_dual_ewma30         157 0.000582 0.000344  0.027364 7.206910e-01   0.4672
pk_t+22 ridge         price+news_adv_full         493 0.000694 0.000372 -0.381206 4.569560e+07   0.5548
pk_t+22   gbm         price+news_adv_full         493 0.000583 0.000344  0.024877 7.222770e-01   0.4676
pk_t+22 ridge      price+news_adv_novelty          97 0.000574 0.000337  0.055933 7.179880e-01   0.4680
pk_t+22   gbm      price+news_adv_novelty          97 0.000580 0.000344  0.034671 7.154030e-01   0.4649
pk_t+22 ridge   price+news_adv_multi_ewma         421 0.000692 0.000371 -0.374525 4.552033e+07   0.5544
pk_t+22   gbm   price+news_adv_multi_ewma         421 0.000583 0.000344  0.024875 7.222790e-01   0.4676
pk_t+22 ridge            price+sentiment5          11 0.000574 0.000336  0.054745 7.187730e-01   0.4701
pk_t+22   gbm            price+sentiment5          11 0.000580 0.000344  0.034531 7.154810e-01   0.4647
pk_t+22 ridge            price+event_type          13 0.000574 0.000336  0.054750 7.191230e-01   0.4703
pk_t+22   gbm            price+event_type          13 0.000580 0.000344  0.034531 7.154810e-01   0.4647
pk_t+22 ridge price+sentiment5+event_type          18 0.000574 0.000336  0.054572 7.192440e-01   0.4700
pk_t+22   gbm price+sentiment5+event_type          18 0.000580 0.000344  0.034531 7.154810e-01   0.4647
```


## News contribution (ΔR² vs price-only; >0 = news helps)


### ridge
- pk_t+1 [price+news_basic]: ΔR²=+0.0007  ΔRMSE=-0.000000 → HELPS
- pk_t+1 [price+news_adv]: ΔR²=+0.0005  ΔRMSE=-0.000000 → HELPS
- pk_t+5 [price+news_basic]: ΔR²=+0.0011  ΔRMSE=-0.000000 → HELPS
- pk_t+5 [price+news_adv]: ΔR²=+0.0009  ΔRMSE=-0.000000 → HELPS
- pk_t+10 [price+news_basic]: ΔR²=+0.0015  ΔRMSE=-0.000000 → HELPS
- pk_t+10 [price+news_adv]: ΔR²=+0.0014  ΔRMSE=-0.000000 → HELPS
- pk_t+22 [price+news_basic]: ΔR²=+0.0013  ΔRMSE=-0.000000 → HELPS
- pk_t+22 [price+news_adv]: ΔR²=+0.0011  ΔRMSE=-0.000000 → HELPS

### gbm
- pk_t+1 [price+news_basic]: ΔR²=+0.0000  ΔRMSE=+0.000000 → neutral
- pk_t+1 [price+news_adv]: ΔR²=+0.0000  ΔRMSE=+0.000000 → neutral
- pk_t+5 [price+news_basic]: ΔR²=-0.0012  ΔRMSE=+0.000000 → no effect
- pk_t+5 [price+news_adv]: ΔR²=-0.0012  ΔRMSE=+0.000000 → no effect
- pk_t+10 [price+news_basic]: ΔR²=+0.0000  ΔRMSE=+0.000000 → neutral
- pk_t+10 [price+news_adv]: ΔR²=-0.0001  ΔRMSE=+0.000000 → no effect
- pk_t+22 [price+news_basic]: ΔR²=+0.0001  ΔRMSE=-0.000000 → HELPS
- pk_t+22 [price+news_adv]: ΔR²=+0.0001  ΔRMSE=-0.000000 → HELPS