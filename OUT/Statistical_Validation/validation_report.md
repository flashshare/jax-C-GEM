# JAX C-GEM Model Validation Report
==================================================

## Executive Summary
- Variables validated: 4
- Average R²: -52.253
- Average Nash-Sutcliffe: -52.253

## Validation Results by Variable

### TIDAL_RANGE
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 5.3886 |
| MAE | 4.8570 |
| R² | -81.9203 |
| Nash-Sutcliffe | -81.9203 |
| Percent Bias | 178.36% |
| Kling-Gupta | -2.7743 |
| Willmott Index | 0.1346 |
| N Observations | 210 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.0624
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 4.7270
- Residual std: 2.5870
- Within 1σ: 19.5%
- Within 2σ: 51.0%

### SALINITY
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 21.0803 |
| MAE | 16.6739 |
| R² | -6.3812 |
| Nash-Sutcliffe | -6.3812 |
| Percent Bias | 257.87% |
| Kling-Gupta | -1.9193 |
| Willmott Index | 0.3112 |
| N Observations | 290 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.9881
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 12.6079
- Residual std: 16.8943
- Within 1σ: 52.4%
- Within 2σ: 83.1%

### OXYGEN
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 181.5737 |
| MAE | 155.8333 |
| R² | -28.4864 |
| Nash-Sutcliffe | -28.4864 |
| Percent Bias | 115.79% |
| Kling-Gupta | -2.8171 |
| Willmott Index | 0.1994 |
| N Observations | 143 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.5151
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0057

**Residual Analysis:**
- Mean residual: 94.0775
- Residual std: 155.3011
- Within 1σ: 53.1%
- Within 2σ: 98.6%

### TIDAL_RANGE_SIHYMECC
**Performance Metrics:**
| Metric | Value |
|--------|-------|
| RMSE | 5.4336 |
| MAE | 4.9699 |
| R² | -92.2222 |
| Nash-Sutcliffe | -92.2222 |
| Percent Bias | 174.85% |
| Kling-Gupta | -2.7247 |
| Willmott Index | 0.1296 |
| N Observations | 129 |

**Statistical Tests:**
- pearson_correlation: p-value = 0.2783
- residual_normality: p-value = 0.0000
- ks_test: p-value = 0.0000
- mann_whitney: p-value = 0.0000

**Residual Analysis:**
- Mean residual: 4.8612
- Residual std: 2.4274
- Within 1σ: 15.5%
- Within 2σ: 48.8%

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
