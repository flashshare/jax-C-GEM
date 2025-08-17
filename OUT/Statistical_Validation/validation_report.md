# JAX C-GEM Model Validation Report
==================================================

## Executive Summary
- Variables validated: 4
- Average R²: -43.466
- Average Nash-Sutcliffe: -43.466

## Validation Results by Variable

### TIDAL_RANGE
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 4.8981 |
| MAE | 4.3010 |
| R² | -67.5111 |
| Nash-Sutcliffe | -67.5111 |
| Percent Bias | 156.70% |
| Kling-Gupta | -2.6796 |
| Willmott Index | 0.1401 |
| N Observations | 210 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0271
- residual_normality: p-value = 0.0003
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 4.1529
- Residual std: 2.5971
- Within 1σ: 26.2%
- Within 2σ: 63.3%

### SALINITY
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 11.5269 |
| MAE | 8.0918 |
| R² | -1.2070 |
| Nash-Sutcliffe | -1.2070 |
| Percent Bias | 0.24% |
| Kling-Gupta | -0.2221 |
| Willmott Index | 0.2146 |
| N Observations | 290 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0002
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 0.0116
- Residual std: 11.5269
- Within 1σ: 67.9%
- Within 2σ: 92.4%

### OXYGEN
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 169.1580 |
| MAE | 165.8201 |
| R² | -24.5918 |
| Nash-Sutcliffe | -24.5918 |
| Percent Bias | 204.09% |
| Kling-Gupta | -1.7303 |
| Willmott Index | 0.2507 |
| N Observations | 143 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0000
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 165.8201
- Residual std: 33.4383
- Within 1σ: 0.0%
- Within 2σ: 0.0%

### TIDAL_RANGE_SIHYMECC
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 5.0821 |
| MAE | 4.6082 |
| R² | -80.5524 |
| Nash-Sutcliffe | -80.5524 |
| Percent Bias | 159.42% |
| Kling-Gupta | -2.7707 |
| Willmott Index | 0.1382 |
| N Observations | 129 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.5487
- residual_normality: p-value = 0.0001
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 4.4323
- Residual std: 2.4865
- Within 1σ: 17.1%
- Within 2σ: 57.4%

## Interpretation and Recommendations

**Model Performance Assessment:**
- R² > 0.7: Excellent agreement
- R² 0.5-0.7: Good agreement
- R² 0.3-0.5: Moderate agreement
- R² < 0.3: Poor agreement

**Nash-Sutcliffe Efficiency:**
- NSE > 0.7: Very good model performance
- NSE 0.5-0.7: Good model performance
- NSE 0.2-0.5: Satisfactory model performance
- NSE < 0.2: Unsatisfactory model performance

**Recommendations for Model Improvement:**
- Focus calibration on variables with poor performance
- Investigate systematic biases in residuals
- Consider additional processes if unexplained variance is high
- Increase observation frequency for better validation
