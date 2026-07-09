# Statistical Validation Report — Paper 1 resubmission

Data: 5 seeds x 4 CV folds per optimizer/dataset cell (paired by identical seed+fold, i.e., identical data partitions). QDPSO (slow reference) excluded: single seed, timing study only.

## 1. Test accuracy: mean ± CI95 (n = 20 seed-fold measurements)

| Dataset | QPSOo | PSO_bound | PSO | Adam(ep100) | Adam(ep1000) |
|---|---|---|---|---|---|
| circle | 0.9170 ± 0.0301 | 0.9060 ± 0.0361 | 0.7995 ± 0.0308 | 0.5410 ± 0.0257 | 0.9825 ± 0.0253 |
| iris | 0.9500 ± 0.0199 | 0.9433 ± 0.0209 | 0.8217 ± 0.0564 | 0.7183 ± 0.0682 | 0.9667 ± 0.0000 |
| wine | 0.8236 ± 0.0389 | 0.7514 ± 0.0412 | 0.6500 ± 0.0333 | 0.9514 ± 0.0168 | 0.9472 ± 0.0111 |
| breast_cancer | 0.9399 ± 0.0124 | 0.8930 ± 0.0131 | 0.8684 ± 0.0186 | 0.9636 ± 0.0033 | 0.9557 ± 0.0036 |

## 2. Friedman test per dataset (H0: all optimizers rank equally)

| Dataset | chi2 | p-value | Mean ranks (QPSOo, PSO_bound, PSO, Adam(ep100), Adam(ep1000)) |
|---|---|---|---|
| circle | 64.74 | 2.92e-13 | 2.42, 2.40, 3.80, 5.00, 1.38 |
| iris | 56.46 | 1.61e-11 | 2.20, 2.30, 4.00, 4.65, 1.85 |
| wine | 67.51 | 7.60e-14 | 3.17, 3.95, 4.75, 1.48, 1.65 |
| breast_cancer | 63.50 | 5.33e-13 | 2.75, 4.17, 4.62, 1.38, 2.08 |

## 3. Pairwise Wilcoxon signed-rank: QDPSOo vs each baseline

Positive mean diff = QDPSOo better. Holm-corrected p-values per dataset.

### circle (n = 20 pairs)

| vs | mean diff | W | p | p (Holm) | significant (α=0.05) |
|---|---|---|---|---|---|
| PSO_bound | +0.0110 | 88.0 | 7.78e-01 | 7.78e-01 | no |
| PSO | +0.1175 | 1.0 | 1.03e-04 | 3.53e-04 | **yes** |
| Adam(ep100) | +0.3760 | 0.0 | 8.82e-05 | 3.53e-04 | **yes** |
| Adam(ep1000) | -0.0655 | 29.5 | 1.47e-02 | 2.93e-02 | **yes** |

### iris (n = 20 pairs)

| vs | mean diff | W | p | p (Holm) | significant (α=0.05) |
|---|---|---|---|---|---|
| PSO_bound | +0.0067 | 48.0 | 7.75e-01 | 7.75e-01 | no |
| PSO | +0.1283 | 7.5 | 1.02e-03 | 3.07e-03 | **yes** |
| Adam(ep100) | +0.2317 | 0.0 | 1.85e-04 | 7.39e-04 | **yes** |
| Adam(ep1000) | -0.0167 | 14.0 | 8.48e-02 | 1.70e-01 | no |

### wine (n = 20 pairs)

| vs | mean diff | W | p | p (Holm) | significant (α=0.05) |
|---|---|---|---|---|---|
| PSO_bound | +0.0722 | 32.5 | 2.07e-02 | 2.07e-02 | **yes** |
| PSO | +0.1736 | 0.0 | 8.70e-05 | 3.48e-04 | **yes** |
| Adam(ep100) | -0.1278 | 2.5 | 1.27e-04 | 3.82e-04 | **yes** |
| Adam(ep1000) | -0.1236 | 1.5 | 1.64e-04 | 3.82e-04 | **yes** |

### breast_cancer (n = 20 pairs)

| vs | mean diff | W | p | p (Holm) | significant (α=0.05) |
|---|---|---|---|---|---|
| PSO_bound | +0.0469 | 5.0 | 4.48e-04 | 1.53e-03 | **yes** |
| PSO | +0.0715 | 10.0 | 3.83e-04 | 1.53e-03 | **yes** |
| Adam(ep100) | -0.0237 | 11.0 | 1.89e-03 | 3.79e-03 | **yes** |
| Adam(ep1000) | -0.0158 | 31.0 | 3.02e-02 | 3.02e-02 | **yes** |

## 4. Robustness check: per-seed means (n = 5 seeds)

With n = 5 the Wilcoxon test cannot reach significance (minimum attainable p = 0.0625), so this check reports *directional consistency*: in how many of the 5 independent seeds the mean difference has the same sign as the main analysis.

| Dataset | vs | mean diff | seeds agreeing | consistent |
|---|---|---|---|---|
| circle | PSO_bound | +0.0110 | 2/5 | no |
| circle | PSO | +0.1175 | 5/5 | **yes** |
| circle | Adam(ep100) | +0.3760 | 5/5 | **yes** |
| circle | Adam(ep1000) | -0.0655 | 5/5 | **yes** |
| iris | PSO_bound | +0.0067 | 3/5 | no |
| iris | PSO | +0.1283 | 5/5 | **yes** |
| iris | Adam(ep100) | +0.2317 | 5/5 | **yes** |
| iris | Adam(ep1000) | -0.0167 | 4/5 | **yes** |
| wine | PSO_bound | +0.0722 | 4/5 | **yes** |
| wine | PSO | +0.1736 | 5/5 | **yes** |
| wine | Adam(ep100) | -0.1278 | 5/5 | **yes** |
| wine | Adam(ep1000) | -0.1236 | 5/5 | **yes** |
| breast_cancer | PSO_bound | +0.0469 | 5/5 | **yes** |
| breast_cancer | PSO | +0.0715 | 5/5 | **yes** |
| breast_cancer | Adam(ep100) | -0.0237 | 5/5 | **yes** |
| breast_cancer | Adam(ep1000) | -0.0158 | 4/5 | **yes** |

## 5. MCW: reduction methods (QDPSOo, n = 20 seed-fold measurements)

| Config | test acc mean ± CI95 |
|---|---|
| ORG/84 | 0.7064 ± 0.0223 |
| isomap/42 | 0.6867 ± 0.0217 |
| pca/42 | 0.7027 ± 0.0211 |
| mds/42 | 0.1767 ± 0.0205 |
| isomap/21 | 0.7598 ± 0.0245 |
| pca/21 | 0.7522 ± 0.0228 |
| mds/21 | 0.2351 ± 0.0231 |
| isomap/14 | 0.7978 ± 0.0165 |
| pca/14 | 0.7980 ± 0.0192 |
| mds/14 | 0.2673 ± 0.0203 |
| isomap/7 | 0.8218 ± 0.0144 |
| pca/7 | 0.8129 ± 0.0111 |
| mds/7 | 0.3458 ± 0.0241 |

### Isomap vs PCA per component count (Wilcoxon, paired)

| Components | mean diff (iso−pca) | p | significant |
|---|---|---|---|
| 42 | -0.0160 | 2.12e-01 | no |
| 21 | +0.0076 | 6.29e-01 | no |
| 14 | -0.0002 | 8.78e-01 | no |
| 7 | +0.0089 | 3.24e-01 | no |

### Best config (isomap/7) vs original representation (ORG/84)

Mean diff = +0.1153, W = 0.0, p = 8.81e-05 (significant at α=0.05)
