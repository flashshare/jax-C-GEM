# JAX C-GEM Model Validation Report
==================================================

## Executive Summary
- Variables validated: 4
- Average R²: -11.107
- Average Nash-Sutcliffe: -11.107

## Validation Results by Variable

### TIDAL_RANGE
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 1.9648 |
| MAE | 1.5844 |
| R² | -10.0237 |
| Nash-Sutcliffe | -10.0237 |
| Percent Bias | -49.70% |
| Kling-Gupta | -0.6777 |
| Willmott Index | 0.2563 |
| N Observations | 210 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.9598
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: -1.3173
- Residual std: 1.4578
- Within 1σ: 51.0%
- Within 2σ: 82.4%

### SALINITY
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 12.0362 |
| MAE | 9.0300 |
| R² | -1.4063 |
| Nash-Sutcliffe | -1.4063 |
| Percent Bias | 29.20% |
| Kling-Gupta | -0.2872 |
| Willmott Index | 0.1678 |
| N Observations | 290 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0000
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 1.4278
- Residual std: 11.9512
- Within 1σ: 65.9%
- Within 2σ: 94.5%

### OXYGEN
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 156.3369 |
| MAE | 150.7829 |
| R² | -20.8594 |
| Nash-Sutcliffe | -20.8594 |
| Percent Bias | 184.61% |
| Kling-Gupta | -1.2055 |
| Willmott Index | 0.2578 |
| N Observations | 143 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0422
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 149.9962
- Residual std: 44.0720
- Within 1σ: 1.4%
- Within 2σ: 7.0%

### TIDAL_RANGE_SIHYMECC
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 2.0398 |
| MAE | 1.6116 |
| R² | -12.1376 |
| Nash-Sutcliffe | -12.1376 |
| Percent Bias | -51.82% |
| Kling-Gupta | -0.7685 |
| Willmott Index | 0.2568 |
| N Observations | 129 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.9844
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: -1.4407
- Residual std: 1.4440
- Within 1σ: 50.4%
- Within 2σ: 75.2%

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
