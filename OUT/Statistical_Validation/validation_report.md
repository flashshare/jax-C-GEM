# JAX C-GEM Model Validation Report
==================================================

## Executive Summary
- Variables validated: 4
- Average R²: -12.053
- Average Nash-Sutcliffe: -12.053

## Validation Results by Variable

### TIDAL_RANGE
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 2.6695 |
| MAE | 2.5769 |
| R² | -19.3509 |
| Nash-Sutcliffe | -19.3509 |
| Percent Bias | -97.02% |
| Kling-Gupta | -0.4655 |
| Willmott Index | 0.2633 |
| N Observations | 210 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.5642
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: -2.5713
- Residual std: 0.7174
- Within 1σ: 1.9%
- Within 2σ: 4.8%

### SALINITY
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 10.3389 |
| MAE | 6.1172 |
| R² | -0.7755 |
| Nash-Sutcliffe | -0.7755 |
| Percent Bias | -57.62% |
| Kling-Gupta | -0.2893 |
| Willmott Index | 0.3325 |
| N Observations | 290 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0547
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.6607

**Residual Analysis:**
- Mean residual: -2.8170
- Residual std: 9.9477
- Within 1σ: 73.1%
- Within 2σ: 89.7%

### OXYGEN
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 78.9022 |
| MAE | 70.5769 |
| R² | -4.5679 |
| Nash-Sutcliffe | -4.5679 |
| Percent Bias | -71.94% |
| Kling-Gupta | -0.1818 |
| Willmott Index | 0.4232 |
| N Observations | 143 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0871
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: -58.4524
- Residual std: 52.9988
- Within 1σ: 38.5%
- Within 2σ: 83.9%

### TIDAL_RANGE_SIHYMECC
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 2.7865 |
| MAE | 2.6961 |
| R² | -23.5171 |
| Nash-Sutcliffe | -23.5171 |
| Percent Bias | -96.97% |
| Kling-Gupta | -0.4687 |
| Willmott Index | 0.2506 |
| N Observations | 129 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.4965
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: -2.6959
- Residual std: 0.7049
- Within 1σ: 2.3%
- Within 2σ: 4.7%

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
