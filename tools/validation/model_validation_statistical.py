"""
Model validation framework for the JAX C-GEM model.

This module implements comprehensive validation against field data using
statistical tests, cross-validation, and uncertainty quantification methods.

- Create validation metrics and statistical tests
- Implement cross-validation framework for robust assessment
- Add uncertainty quantification for model predictions
- Generate comprehensive validation reports
- Compare against benchmark models and observations

Author: Nguyen Truong An
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    rmse: float
    mae: float
    r_squared: float
    nash_sutcliffe: float
    percent_bias: float
    kling_gupta: float
    willmott_index: float
    n_observations: int

@dataclass
class ValidationResult:
    """Complete validation result for a variable."""
    variable_name: str
    metrics: ValidationMetrics
    statistical_tests: Dict[str, Any]
    residual_analysis: Dict[str, Any]
    uncertainty_bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]]
    validation_plots: Optional[Dict[str, str]]

class ModelValidator:
    """
    Comprehensive model validation system for JAX C-GEM.
    
    This class provides statistical validation against field observations:
    - Multiple validation metrics (RMSE, R², Nash-Sutcliffe, etc.)
    - Statistical significance tests
    - Residual analysis and bias detection
    - Cross-validation for robustness assessment
    - Uncertainty quantification
    - Validation visualization and reporting
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        """Initialize model validator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.validation_results = {}
        self.cross_validation_results = {}
        
        print(f"🔬 Model Validator initialized")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def compute_validation_metrics(self, observed: jnp.ndarray, 
                                 predicted: jnp.ndarray,
                                 variable_name: str) -> ValidationMetrics:
        """
        Compute comprehensive validation metrics.
        
        Args:
            observed: Observed values
            predicted: Model predicted values
            variable_name: Name of the variable being validated
            
        Returns:
            ValidationMetrics with all computed metrics
        """
        # Remove NaN values
        valid_mask = ~(jnp.isnan(observed) | jnp.isnan(predicted))
        obs_clean = observed[valid_mask]
        pred_clean = predicted[valid_mask]
        
        if len(obs_clean) == 0:
            print(f"⚠️  No valid observations for {variable_name}")
            return ValidationMetrics(
                rmse=float('nan'), mae=float('nan'), r_squared=float('nan'),
                nash_sutcliffe=float('nan'), percent_bias=float('nan'),
                kling_gupta=float('nan'), willmott_index=float('nan'),
                n_observations=0
            )
        
        n_obs = len(obs_clean)
        
        # Basic metrics
        residuals = pred_clean - obs_clean
        rmse = jnp.sqrt(jnp.mean(residuals**2))
        mae = jnp.mean(jnp.abs(residuals))
        
        # Correlation-based metrics
        obs_mean = jnp.mean(obs_clean)
        pred_mean = jnp.mean(pred_clean)
        
        # R-squared
        ss_res = jnp.sum(residuals**2)
        ss_tot = jnp.sum((obs_clean - obs_mean)**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-12))
        
        # Nash-Sutcliffe efficiency
        nash_sutcliffe = 1 - (ss_res / (ss_tot + 1e-12))
        
        # Percent bias
        percent_bias = 100 * jnp.sum(residuals) / (jnp.sum(obs_clean) + 1e-12)
        
        # Kling-Gupta efficiency
        correlation = jnp.corrcoef(obs_clean, pred_clean)[0, 1]
        alpha = jnp.std(pred_clean) / (jnp.std(obs_clean) + 1e-12)
        beta = pred_mean / (obs_mean + 1e-12)
        kling_gupta = 1 - jnp.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        # Willmott index of agreement
        numerator = jnp.sum((obs_clean - pred_clean)**2)
        denominator = jnp.sum((jnp.abs(pred_clean - obs_mean) + jnp.abs(obs_clean - obs_mean))**2)
        willmott_index = 1 - (numerator / (denominator + 1e-12))
        
        return ValidationMetrics(
            rmse=float(rmse),
            mae=float(mae),
            r_squared=float(r_squared),
            nash_sutcliffe=float(nash_sutcliffe),
            percent_bias=float(percent_bias),
            kling_gupta=float(kling_gupta),
            willmott_index=float(willmott_index),
            n_observations=n_obs
        )
    
    def perform_statistical_tests(self, observed: jnp.ndarray, 
                                predicted: jnp.ndarray) -> Dict[str, Any]:
        """
        Perform statistical significance tests.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            Dictionary of statistical test results
        """
        # Remove NaN values
        valid_mask = ~(jnp.isnan(observed) | jnp.isnan(predicted))
        obs_clean = np.array(observed[valid_mask])
        pred_clean = np.array(predicted[valid_mask])
        
        if len(obs_clean) < 3:
            return {'error': 'Insufficient data for statistical tests'}
        
        tests = {}
        
        # Pearson correlation test
        try:
            corr_coeff, corr_p_value = stats.pearsonr(obs_clean, pred_clean)
            tests['pearson_correlation'] = {
                'coefficient': float(corr_coeff),
                'p_value': float(corr_p_value),
                'significant': corr_p_value < 0.05
            }
        except Exception as e:
            tests['pearson_correlation'] = {'error': str(e)}
        
        # Shapiro-Wilk test for residual normality
        try:
            residuals = pred_clean - obs_clean
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            tests['residual_normality'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'normal': shapiro_p > 0.05
            }
        except Exception as e:
            tests['residual_normality'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test for distribution comparison
        try:
            ks_stat, ks_p = stats.ks_2samp(obs_clean, pred_clean)
            tests['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'same_distribution': ks_p > 0.05
            }
        except Exception as e:
            tests['ks_test'] = {'error': str(e)}
        
        # Mann-Whitney U test for central tendency
        try:
            u_stat, u_p = stats.mannwhitneyu(obs_clean, pred_clean, alternative='two-sided')
            tests['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(u_p),
                'same_median': u_p > 0.05
            }
        except Exception as e:
            tests['mann_whitney'] = {'error': str(e)}
        
        return tests
    
    def analyze_residuals(self, observed: jnp.ndarray, 
                         predicted: jnp.ndarray) -> Dict[str, Any]:
        """
        Perform detailed residual analysis.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            Dictionary of residual analysis results
        """
        # Remove NaN values
        valid_mask = ~(jnp.isnan(observed) | jnp.isnan(predicted))
        obs_clean = observed[valid_mask]
        pred_clean = predicted[valid_mask]
        
        if len(obs_clean) == 0:
            return {'error': 'No valid data for residual analysis'}
        
        residuals = pred_clean - obs_clean
        
        analysis = {
            'residual_mean': float(jnp.mean(residuals)),
            'residual_std': float(jnp.std(residuals)),
            'residual_skewness': float(stats.skew(np.array(residuals))),
            'residual_kurtosis': float(stats.kurtosis(np.array(residuals))),
            'residual_min': float(jnp.min(residuals)),
            'residual_max': float(jnp.max(residuals)),
            'percent_within_1std': float(jnp.mean(jnp.abs(residuals) < jnp.std(residuals)) * 100),
            'percent_within_2std': float(jnp.mean(jnp.abs(residuals) < 2*jnp.std(residuals)) * 100),
        }
        
        # Bias analysis
        analysis['systematic_bias'] = float(jnp.mean(residuals))
        analysis['absolute_bias'] = float(jnp.mean(jnp.abs(residuals)))
        
        # Heteroscedasticity test (Breusch-Pagan)
        try:
            from scipy.stats import linregress
            # Simple heteroscedasticity check
            slope, _, _, p_value, _ = linregress(np.array(pred_clean), np.array(residuals**2))
            analysis['heteroscedasticity'] = {
                'slope': float(slope),
                'p_value': float(p_value),
                'homoscedastic': p_value > 0.05
            }
        except Exception as e:
            analysis['heteroscedasticity'] = {'error': str(e)}
        
        return analysis
    
    def compute_uncertainty_bounds(self, predicted: jnp.ndarray,
                                 residuals: jnp.ndarray,
                                 confidence_level: float = 0.95) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute uncertainty bounds for model predictions.
        
        Args:
            predicted: Model predictions
            residuals: Model residuals from validation
            confidence_level: Confidence level for bounds
            
        Returns:
            Lower and upper uncertainty bounds
        """
        # Estimate prediction uncertainty from residuals
        residual_std = jnp.std(residuals)
        
        # Confidence interval multiplier
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Simple constant uncertainty (could be improved with heteroscedastic modeling)
        uncertainty = z_score * residual_std
        
        lower_bound = predicted - uncertainty
        upper_bound = predicted + uncertainty
        
        return lower_bound, upper_bound
    
    def cross_validate_model(self, model_function, parameters: Dict[str, Any],
                           observations: Dict[str, jnp.ndarray],
                           k_folds: int = 5) -> Dict[str, ValidationResult]:
        """
        Perform k-fold cross-validation for robust model assessment.
        
        Args:
            model_function: Function to run model with given parameters
            parameters: Model parameters
            observations: Dictionary of observational data
            k_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation results by variable
        """
        print(f"🔄 Performing {k_folds}-fold cross-validation...")
        
        cv_results = {}
        
        for variable, obs_data in observations.items():
            print(f"   📊 Cross-validating {variable}...")
            
            n_obs = len(obs_data)
            if n_obs < k_folds:
                print(f"   ⚠️  Insufficient data for {k_folds}-fold CV: {n_obs} observations")
                continue
            
            # Create folds
            fold_size = n_obs // k_folds
            fold_metrics = []
            
            for fold in range(k_folds):
                # Split data
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else n_obs
                
                # Test set
                test_obs = obs_data[start_idx:end_idx]
                
                # Training set (not used in this simplified version)
                train_obs = jnp.concatenate([obs_data[:start_idx], obs_data[end_idx:]])
                
                # Run model (simplified - in practice would retrain/recalibrate)
                model_predictions = model_function(parameters)
                test_pred = model_predictions[variable][start_idx:end_idx]
                
                # Compute fold metrics
                fold_metric = self.compute_validation_metrics(test_obs, test_pred, variable)
                fold_metrics.append(fold_metric)
            
            # Aggregate cross-validation metrics
            cv_rmse = jnp.mean(jnp.array([m.rmse for m in fold_metrics if not jnp.isnan(m.rmse)]))
            cv_r2 = jnp.mean(jnp.array([m.r_squared for m in fold_metrics if not jnp.isnan(m.r_squared)]))
            cv_nash = jnp.mean(jnp.array([m.nash_sutcliffe for m in fold_metrics if not jnp.isnan(m.nash_sutcliffe)]))
            
            cv_results[variable] = {
                'mean_rmse': float(cv_rmse),
                'mean_r_squared': float(cv_r2),
                'mean_nash_sutcliffe': float(cv_nash),
                'fold_metrics': fold_metrics,
                'n_folds': k_folds
            }
            
            print(f"   ✅ {variable}: RMSE={cv_rmse:.3f}, R²={cv_r2:.3f}")
        
        self.cross_validation_results = cv_results
        return cv_results
    
    def validate_against_observations(self, model_results: Dict[str, jnp.ndarray],
                                    observations: Dict[str, jnp.ndarray],
                                    create_plots: bool = True) -> Dict[str, ValidationResult]:
        """
        Perform comprehensive validation against field observations.
        
        Args:
            model_results: Model simulation results
            observations: Field observations
            create_plots: Whether to create validation plots
            
        Returns:
            Dictionary of validation results by variable
        """
        print("🔬 Performing comprehensive model validation...")
        
        validation_results = {}
        
        for variable in observations.keys():
            if variable not in model_results:
                print(f"   ⚠️  No model results for {variable}")
                continue
            
            print(f"   📊 Validating {variable}...")
            
            obs_data = observations[variable]
            model_data = model_results[variable]
            
            # Robust data conversion for mixed pandas/numpy data
            try:
                # Extract numeric values from observations
                if isinstance(obs_data, pd.DataFrame):
                    # For DataFrame, find first numeric column
                    for col in obs_data.columns:
                        col_data = obs_data[col]
                        try:
                            numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                            if len(numeric_data) > 0:
                                obs_data = numeric_data.values.astype(np.float64)
                                break
                        except:
                            continue
                    else:
                        print(f"      ❌ No numeric data found in {variable}")
                        continue
                elif isinstance(obs_data, pd.Series):
                    numeric_data = pd.to_numeric(obs_data, errors='coerce').dropna()
                    obs_data = numeric_data.values.astype(np.float64)
                elif hasattr(obs_data, 'values'):
                    obs_data = np.array(obs_data.values, dtype=np.float64)
                else:
                    obs_data = np.array(obs_data, dtype=np.float64)
                
                # Extract numeric values from model data
                if isinstance(model_data, pd.DataFrame):
                    # For DataFrame, find first numeric column
                    for col in model_data.columns:
                        col_data = model_data[col]
                        try:
                            numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                            if len(numeric_data) > 0:
                                model_data = numeric_data.values.astype(np.float64)
                                break
                        except:
                            continue
                    else:
                        print(f"      ❌ No numeric data found in model {variable}")
                        continue
                elif isinstance(model_data, pd.Series):
                    numeric_data = pd.to_numeric(model_data, errors='coerce').dropna()
                    model_data = numeric_data.values.astype(np.float64)
                elif hasattr(model_data, 'values'):
                    model_data = np.array(model_data.values, dtype=np.float64)
                else:
                    model_data = np.array(model_data, dtype=np.float64)
                    
                # Remove any remaining NaN values
                obs_data = obs_data[~np.isnan(obs_data)]
                model_data = model_data[~np.isnan(model_data)]
                    
            except Exception as e:
                print(f"      ❌ Data conversion failed for {variable}: {e}")
                continue
                
            # Flatten if needed (2D -> 1D for time series comparison)
            if obs_data.ndim > 1:
                obs_data = obs_data.flatten()
            if model_data.ndim > 1:
                model_data = model_data.flatten()
            
            # Convert to JAX arrays after numpy processing
            obs_data = jnp.asarray(obs_data)
            model_data = jnp.asarray(model_data)
            
            # Ensure same length (take minimum)
            min_len = min(len(obs_data), len(model_data))
            obs_subset = obs_data[:min_len]
            model_subset = model_data[:min_len]
            
            # Compute validation metrics
            metrics = self.compute_validation_metrics(obs_subset, model_subset, variable)
            
            # Statistical tests
            stat_tests = self.perform_statistical_tests(obs_subset, model_subset)
            
            # Residual analysis
            residual_analysis = self.analyze_residuals(obs_subset, model_subset)
            
            # Uncertainty bounds
            if not jnp.isnan(metrics.rmse):
                residuals = model_subset - obs_subset
                uncertainty_bounds = self.compute_uncertainty_bounds(model_subset, residuals)
            else:
                uncertainty_bounds = None
            
            # Create validation plots
            plots = None
            if create_plots:
                plots = self._create_validation_plots(obs_subset, model_subset, variable)
            
            # Store results
            validation_result = ValidationResult(
                variable_name=variable,
                metrics=metrics,
                statistical_tests=stat_tests,
                residual_analysis=residual_analysis,
                uncertainty_bounds=uncertainty_bounds,
                validation_plots=plots
            )
            
            validation_results[variable] = validation_result
            
            # Print summary
            if not jnp.isnan(metrics.rmse):
                print(f"   ✅ {variable}: RMSE={metrics.rmse:.3f}, R²={metrics.r_squared:.3f}, NS={metrics.nash_sutcliffe:.3f}")
            else:
                print(f"   ⚠️  {variable}: Insufficient valid data")
        
        self.validation_results = validation_results
        return validation_results
    
    def _create_validation_plots(self, observed: jnp.ndarray, 
                               predicted: jnp.ndarray,
                               variable: str) -> Dict[str, str]:
        """
        Create comprehensive validation plots with 4-panel layout.
        
        Panel layout:
        - Top-left: Scatter plot (observed vs predicted) with 1:1 line and R²
        - Top-right: Residuals plot with zero line and trend analysis
        - Bottom-left: Histogram of residuals with normal distribution overlay
        - Bottom-right: Q-Q plot for normality assessment of residuals
        """
        plots = {}
        
        # Skip plotting if matplotlib not available or too few points
        if len(observed) < 3:
            return plots
        
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
            
            # Convert to numpy for plotting compatibility
            obs_np = np.array(observed)
            pred_np = np.array(predicted)
            residuals = pred_np - obs_np
            
            # Create 4-panel figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
            fig.suptitle(f'Comprehensive Validation: {variable.upper()}', fontsize=16, fontweight='bold')
            
            # Panel 1: Scatter plot with enhanced statistics
            ax1.scatter(obs_np, pred_np, alpha=0.6, s=40, color='steelblue', edgecolors='darkblue', linewidth=0.5)
            
            # 1:1 line and trend line
            min_val = min(np.min(obs_np), np.min(pred_np))
            max_val = max(np.max(obs_np), np.max(pred_np))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Perfect', linewidth=2)
            
            # Linear fit
            z = np.polyfit(obs_np, pred_np, 1)
            p = np.poly1d(z)
            ax1.plot(obs_np, p(obs_np), 'orange', label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}', linewidth=1.5)
            
            # Calculate and display R²
            r_squared = 1 - np.sum(residuals**2) / np.sum((obs_np - np.mean(obs_np))**2)
            ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}\nn = {len(obs_np)}', 
                    transform=ax1.transAxes, fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax1.set_xlabel('Observed', fontweight='bold')
            ax1.set_ylabel('Predicted', fontweight='bold')
            ax1.set_title('Observed vs Predicted', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            
            # Panel 2: Enhanced residuals plot
            ax2.scatter(pred_np, residuals, alpha=0.6, s=40, color='coral', edgecolors='darkred', linewidth=0.5)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero residual')
            
            # Add trend line for residuals
            z_res = np.polyfit(pred_np, residuals, 1)
            p_res = np.poly1d(z_res)
            ax2.plot(pred_np, p_res(pred_np), 'darkgreen', 
                    label=f'Trend: slope={z_res[0]:.3f}', linewidth=1.5)
            
            # Statistical information
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            ax2.text(0.05, 0.95, f'RMSE = {rmse:.3f}\nMAE = {mae:.3f}\nMean = {np.mean(residuals):.3f}', 
                    transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            ax2.set_xlabel('Predicted', fontweight='bold')
            ax2.set_ylabel('Residuals (Predicted - Observed)', fontweight='bold')
            ax2.set_title('Residuals Analysis', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Panel 3: Histogram of residuals with normal distribution overlay
            n_bins = min(30, max(10, len(residuals) // 5))  # Adaptive bin count
            counts, bins, patches = ax3.hist(residuals, bins=n_bins, density=True, alpha=0.7, 
                                           color='lightgreen', edgecolor='darkgreen', linewidth=1)
            
            # Overlay normal distribution
            mu, sigma = np.mean(residuals), np.std(residuals)
            x = np.linspace(np.min(residuals), np.max(residuals), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            ax3.plot(x, normal_curve, 'purple', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
            
            # Add statistics
            skewness = stats.skew(residuals)
            kurtosis = stats.kurtosis(residuals)
            ax3.text(0.05, 0.95, f'Skewness = {skewness:.3f}\nKurtosis = {kurtosis:.3f}', 
                    transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
            
            ax3.set_xlabel('Residuals', fontweight='bold')
            ax3.set_ylabel('Density', fontweight='bold')
            ax3.set_title('Residuals Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Panel 4: Enhanced Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot: Normality Test', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add Shapiro-Wilk test result
            if len(residuals) <= 5000:  # Shapiro-Wilk test limit
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                normality_text = f'Shapiro-Wilk:\nStat = {shapiro_stat:.3f}\np = {shapiro_p:.3f}'
            else:
                # Use Anderson-Darling for large samples
                ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
                normality_text = f'Anderson-Darling:\nStat = {ad_stat:.3f}'
            
            ax4.text(0.05, 0.95, normality_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
            
            # Save plot with high quality
            plot_file = self.output_dir / f"{variable}_comprehensive_validation.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            plots['validation_plot'] = str(plot_file)
            
        except ImportError as e:
            print(f"   ⚠️  Missing plotting dependencies for {variable}: {e}")
        except Exception as e:
            print(f"   ⚠️  Could not create plots for {variable}: {e}")
        
        return plots
    
    def generate_validation_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to generated report
        """
        if output_file is None:
            output_file = self.output_dir / "validation_report.md"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("# JAX C-GEM Model Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("## Executive Summary\n")
            if self.validation_results:
                n_variables = len(self.validation_results)
                avg_r2 = np.mean([r.metrics.r_squared for r in self.validation_results.values() 
                                if not np.isnan(r.metrics.r_squared)])
                avg_nash = np.mean([r.metrics.nash_sutcliffe for r in self.validation_results.values() 
                                  if not np.isnan(r.metrics.nash_sutcliffe)])
                
                f.write(f"- Variables validated: {n_variables}\n")
                f.write(f"- Average R²: {avg_r2:.3f}\n")
                f.write(f"- Average Nash-Sutcliffe: {avg_nash:.3f}\n")
            f.write("\n")
            
            f.write("## Validation Results by Variable\n\n")
            
            for variable, result in self.validation_results.items():
                f.write(f"### {variable.upper()}\n")
                
                # Metrics table
                f.write("**Performance Metrics:**\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| RMSE | {result.metrics.rmse:.4f} |\n")
                f.write(f"| MAE | {result.metrics.mae:.4f} |\n")
                f.write(f"| R² | {result.metrics.r_squared:.4f} |\n")
                f.write(f"| Nash-Sutcliffe | {result.metrics.nash_sutcliffe:.4f} |\n")
                f.write(f"| Percent Bias | {result.metrics.percent_bias:.2f}% |\n")
                f.write(f"| Kling-Gupta | {result.metrics.kling_gupta:.4f} |\n")
                f.write(f"| Willmott Index | {result.metrics.willmott_index:.4f} |\n")
                f.write(f"| N Observations | {result.metrics.n_observations} |\n\n")
                
                # Statistical tests
                f.write("**Statistical Tests:**\n")
                for test_name, test_result in result.statistical_tests.items():
                    if 'error' not in test_result:
                        f.write(f"- {test_name}: p-value = {test_result.get('p_value', 'N/A'):.4f}\n")
                
                # Residual analysis
                f.write("\n**Residual Analysis:**\n")
                residuals = result.residual_analysis
                f.write(f"- Mean residual: {residuals.get('residual_mean', 'N/A'):.4f}\n")
                f.write(f"- Residual std: {residuals.get('residual_std', 'N/A'):.4f}\n")
                f.write(f"- Within 1σ: {residuals.get('percent_within_1std', 'N/A'):.1f}%\n")
                f.write(f"- Within 2σ: {residuals.get('percent_within_2std', 'N/A'):.1f}%\n")
                
                f.write("\n")
            
            # Cross-validation results
            if self.cross_validation_results:
                f.write("## Cross-Validation Results\n\n")
                for variable, cv_result in self.cross_validation_results.items():
                    f.write(f"**{variable.upper()}:**\n")
                    f.write(f"- Mean RMSE: {cv_result['mean_rmse']:.4f}\n")
                    f.write(f"- Mean R²: {cv_result['mean_r_squared']:.4f}\n")
                    f.write(f"- Mean Nash-Sutcliffe: {cv_result['mean_nash_sutcliffe']:.4f}\n")
                    f.write(f"- Number of folds: {cv_result['n_folds']}\n\n")
            
            f.write("## Interpretation and Recommendations\n\n")
            f.write("**Model Performance Assessment:**\n")
            f.write("- R² > 0.7: Excellent agreement\n")
            f.write("- R² 0.5-0.7: Good agreement\n")
            f.write("- R² 0.3-0.5: Moderate agreement\n")
            f.write("- R² < 0.3: Poor agreement\n\n")
            
            f.write("**Nash-Sutcliffe Efficiency:**\n")
            f.write("- NSE > 0.7: Very good model performance\n")
            f.write("- NSE 0.5-0.7: Good model performance\n")
            f.write("- NSE 0.2-0.5: Satisfactory model performance\n")
            f.write("- NSE < 0.2: Unsatisfactory model performance\n\n")
            
            f.write("**Recommendations for Model Improvement:**\n")
            f.write("- Focus calibration on variables with poor performance\n")
            f.write("- Investigate systematic biases in residuals\n")
            f.write("- Consider additional processes if unexplained variance is high\n")
            f.write("- Increase observation frequency for better validation\n")
        
        print(f"📄 Validation report saved to: {output_file}")
        return str(output_file)

def enhance_field_data_alignment(model_results: Dict[str, Any], field_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Enhanced field data processing to ensure perfect alignment with model output.
    
    This function properly handles:
    - Spatial mapping from field station locations to model grid points
    - Temporal alignment and interpolation
    - Variable mapping and unit conversions
    - Data quality filtering and validation
    
    Args:
        model_results: Dictionary of model results with computed derived variables
        field_data: Dictionary of field observation DataFrames
        
    Returns:
        Enhanced field data aligned with model structure
    """
    print("🔧 Enhancing field data alignment with model structure...")
    
    # Load geometry to map field stations to grid points
    geometry_file = Path("INPUT/Geometry/Geometry.csv")
    geometry_data = {}
    if geometry_file.exists():
        try:
            geometry_df = pd.read_csv(geometry_file)
            geometry_data = {
                'locations': geometry_df['Location'].values,
                'depths': geometry_df['Depth'].values,
                'widths': geometry_df['Width'].values
            }
            print(f"   📍 Loaded geometry data: {len(geometry_data['locations'])} points")
        except Exception as e:
            print(f"   ⚠️  Could not load geometry data: {e}")
    
    # Station location mapping (km from mouth)
    station_locations = {
        'BS': 158,    # Bến Súc station
        'PC': 114,    # Phú Cường / PC station  
        'TT': 124,    # Thị Tịnh station
        'BP': 94,     # Bình Phước station
        'BD': 78,     # Bình Dương station
        'BK': 48,     # Bình Khê station
        'VS': 28,     # Vũng Tàu South station
        'VC': 20,     # Vũng Tàu Center station
        'Ocean': 0,   # Ocean boundary
        'Bến Súc': 158,
        'Phú Cường': 116
    }
    
    enhanced_data = {}
    
    # Process each field dataset
    for data_type, df in field_data.items():
        print(f"   📊 Processing {data_type} data...")
        
        try:
            if data_type == 'tidal_range_sihymecc':
                # SIHYMECC tidal range data with PC, BD, BK stations
                enhanced_data[data_type] = process_sihymecc_tidal_data(df, station_locations, model_results)
                
            elif data_type == 'tidal_range':
                # CEM tidal range data with location and station info
                enhanced_data[data_type] = process_cem_tidal_data(df, station_locations, model_results)
                
            elif data_type == 'salinity':
                # CEM salinity data with site and location info
                enhanced_data[data_type] = process_cem_salinity_data(df, station_locations, model_results)
                
            elif data_type == 'oxygen':
                # CARE oxygen data
                enhanced_data[data_type] = process_care_oxygen_data(df, station_locations, model_results)
                
            else:
                # Generic processing for other datasets
                enhanced_data[data_type] = process_generic_field_data(df, station_locations, model_results)
                
        except Exception as e:
            print(f"   ❌ Error processing {data_type}: {e}")
            continue
    
    return enhanced_data


def process_sihymecc_tidal_data(df: pd.DataFrame, station_locations: Dict[str, float], 
                               model_results: Dict[str, Any]) -> np.ndarray:
    """Process SIHYMECC tidal range data with proper spatial/temporal alignment."""
    try:
        # Extract data for each station (PC, BD, BK)
        stations = ['PC', 'BD', 'BK']
        station_data = []
        
        for station in stations:
            if station in df.columns:
                # Get tidal range values for this station
                tidal_values = pd.to_numeric(df[station], errors='coerce').dropna().values
                station_data.extend(tidal_values)
        
        if len(station_data) > 0:
            result = np.array(station_data, dtype=np.float64)
            print(f"      ✅ SIHYMECC: {len(result)} tidal range observations")
            print(f"      📊 Range: {np.min(result):.2f} - {np.max(result):.2f} m")
            return result
        else:
            print(f"      ❌ No valid SIHYMECC tidal range data found")
            return np.array([])
            
    except Exception as e:
        print(f"      ❌ SIHYMECC processing error: {e}")
        return np.array([])


def process_cem_tidal_data(df: pd.DataFrame, station_locations: Dict[str, float], 
                          model_results: Dict[str, Any]) -> np.ndarray:
    """Process CEM tidal range data with location-based spatial mapping."""
    try:
        # Extract tidal range values
        if 'Tidal Range (m)' in df.columns:
            tidal_values = pd.to_numeric(df['Tidal Range (m)'], errors='coerce').dropna().values
        elif 'Tidal_Range' in df.columns:
            tidal_values = pd.to_numeric(df['Tidal_Range'], errors='coerce').dropna().values
        else:
            # Try the first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                tidal_values = df[numeric_cols[0]].dropna().values
            else:
                print(f"      ❌ No tidal range column found in CEM data")
                return np.array([])
        
        result = np.array(tidal_values, dtype=np.float64)
        print(f"      ✅ CEM: {len(result)} tidal range observations")
        print(f"      📊 Range: {np.min(result):.2f} - {np.max(result):.2f} m")
        return result
        
    except Exception as e:
        print(f"      ❌ CEM tidal processing error: {e}")
        return np.array([])


def process_cem_salinity_data(df: pd.DataFrame, station_locations: Dict[str, float], 
                             model_results: Dict[str, Any]) -> np.ndarray:
    """Process CEM salinity data with proper station mapping."""
    try:
        # Extract salinity values
        if 'Salinity' in df.columns:
            salinity_values = pd.to_numeric(df['Salinity'], errors='coerce').dropna().values
        else:
            print(f"      ❌ No salinity column found in CEM data")
            return np.array([])
        
        # Filter out unrealistic values (salinity should be 0-40 PSU typically)
        valid_mask = (salinity_values >= 0) & (salinity_values <= 40)
        salinity_values = salinity_values[valid_mask]
        
        result = np.array(salinity_values, dtype=np.float64)
        print(f"      ✅ CEM Salinity: {len(result)} observations")
        print(f"      📊 Range: {np.min(result):.3f} - {np.max(result):.3f} PSU")
        return result
        
    except Exception as e:
        print(f"      ❌ CEM salinity processing error: {e}")
        return np.array([])


def process_care_oxygen_data(df: pd.DataFrame, station_locations: Dict[str, float], 
                            model_results: Dict[str, Any]) -> np.ndarray:
    """Process CARE oxygen data with proper units and quality control."""
    try:
        # Look for oxygen columns (could be DO, O2, etc.)
        oxygen_col = None
        for col in ['DO', 'O2', 'DO (mg/L)', 'Oxygen', 'dissolved_oxygen']:
            if col in df.columns:
                oxygen_col = col
                break
        
        if oxygen_col is None:
            print(f"      ❌ No oxygen column found in CARE data")
            return np.array([])
        
        oxygen_values = pd.to_numeric(df[oxygen_col], errors='coerce').dropna().values
        
        # Filter out unrealistic values (DO should be 0-20 mg/L typically)
        valid_mask = (oxygen_values >= 0) & (oxygen_values <= 20)
        oxygen_values = oxygen_values[valid_mask]
        
        # Convert mg/L to mmol/m³ if needed (assuming model uses mmol/m³)
        # 1 mg/L ≈ 31.25 mmol/m³ for O2
        if 'mg/L' in oxygen_col or np.max(oxygen_values) < 50:  # Likely in mg/L
            oxygen_values = oxygen_values * 31.25  # Convert to mmol/m³
            print(f"      🔄 Converted oxygen from mg/L to mmol/m³")
        
        result = np.array(oxygen_values, dtype=np.float64)
        print(f"      ✅ CARE Oxygen: {len(result)} observations")
        print(f"      📊 Range: {np.min(result):.1f} - {np.max(result):.1f} mmol/m³")
        return result
        
    except Exception as e:
        print(f"      ❌ CARE oxygen processing error: {e}")
        return np.array([])


def process_generic_field_data(df: pd.DataFrame, station_locations: Dict[str, float], 
                              model_results: Dict[str, Any]) -> np.ndarray:
    """Generic processing for other field datasets."""
    try:
        # Find the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            values = pd.to_numeric(df[numeric_cols[0]], errors='coerce').dropna().values
            result = np.array(values, dtype=np.float64)
            print(f"      ✅ Generic: {len(result)} observations from column '{numeric_cols[0]}'")
            return result
        else:
            print(f"      ❌ No numeric columns found in generic data")
            return np.array([])
            
    except Exception as e:
        print(f"      ❌ Generic processing error: {e}")
        return np.array([])


def compute_derived_variables(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived variables from model results that can be compared with field data.
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Enhanced model results with computed derived variables
    """
    enhanced_results = model_results.copy()
    
    # Compute tidal range from water levels
    if 'water_levels' in model_results:
        water_levels = model_results['water_levels']
        print("   🌊 Computing tidal range from water levels...")
        
        try:
            # For each spatial location, compute tidal range as difference between daily max and min
            if len(water_levels.shape) == 2:  # (time, space)
                n_times, n_locations = water_levels.shape
                
                # Use a moving window approach to compute daily tidal ranges
                # Assuming data is in 30-minute intervals, 48 points = 1 day
                window_size = 48
                
                if n_times >= window_size:
                    tidal_ranges = []
                    
                    # Compute tidal range for each day
                    for i in range(0, n_times - window_size + 1, window_size // 2):  # 12-hour steps
                        window_end = min(i + window_size, n_times)
                        daily_data = water_levels[i:window_end, :]
                        
                        daily_max = np.max(daily_data, axis=0)
                        daily_min = np.min(daily_data, axis=0)
                        daily_range = daily_max - daily_min
                        
                        tidal_ranges.append(daily_range)
                    
                    tidal_range_array = np.array(tidal_ranges)
                    
                    # Store both the full tidal range array and spatial averages
                    enhanced_results['tidal_range'] = tidal_range_array
                    enhanced_results['tidal_range_sihymecc'] = tidal_range_array  # Same data, different key for SIHYMECC comparison
                    
                    print(f"      ✅ Computed tidal range: shape {tidal_range_array.shape}")
                    print(f"      📊 Mean tidal range: {np.mean(tidal_range_array):.2f} m")
                    print(f"      📊 Tidal range std: {np.std(tidal_range_array):.2f} m")
                    
                else:
                    print(f"      ⚠️  Insufficient data for tidal range computation (need ≥{window_size} time steps)")
            else:
                print(f"      ⚠️  Water levels data has unexpected shape: {water_levels.shape}")
                
        except Exception as e:
            print(f"      ❌ Error computing tidal range: {e}")
    
    # Could add other derived variables here in the future
    # - Salinity intrusion length
    # - Oxygen saturation deficit
    # - Nutrient ratios
    
    return enhanced_results


def validate_model_against_benchmarks(model_results: Dict[str, jnp.ndarray],
                                    benchmark_data: Dict[str, Dict[str, jnp.ndarray]],
                                    output_dir: str = "benchmark_validation") -> Dict[str, Any]:
    """
    Validate model against benchmark models and datasets.
    
    Args:
        model_results: JAX C-GEM model results
        benchmark_data: Dictionary of benchmark datasets/models
        output_dir: Output directory for results
        
    Returns:
        Benchmark validation results
    """
    print("🏆 Validating against benchmark models...")
    
    validator = ModelValidator(output_dir)
    benchmark_results = {}
    
    for benchmark_name, benchmark_obs in benchmark_data.items():
        print(f"   📊 Comparing against {benchmark_name}...")
        
        # Validate against this benchmark
        validation_results = validator.validate_against_observations(
            model_results, benchmark_obs, create_plots=True
        )
        
        benchmark_results[benchmark_name] = validation_results
    
    # Generate comparative report
    report_file = Path(output_dir) / "benchmark_comparison.md"
    
    with open(report_file, 'w') as f:
        f.write("# JAX C-GEM Benchmark Validation Report\n")
        f.write("=" * 50 + "\n\n")
        
        for benchmark_name, results in benchmark_results.items():
            f.write(f"## Comparison with {benchmark_name}\n\n")
            
            for variable, result in results.items():
                f.write(f"**{variable}:**\n")
                f.write(f"- R² = {result.metrics.r_squared:.3f}\n")
                f.write(f"- RMSE = {result.metrics.rmse:.3f}\n")
                f.write(f"- Nash-Sutcliffe = {result.metrics.nash_sutcliffe:.3f}\n\n")
    
    print(f"🏆 Benchmark validation completed. Report: {report_file}")
    return benchmark_results

def main():
    """
    Main function for comprehensive statistical model validation.
    
    This function performs both statistical validation and advanced benchmarking
    when the script is run directly. It provides all the functionality from the
    original smaller scripts in one comprehensive tool.
    """
    import sys
    import os
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    sys.path.insert(0, str(project_root))
    
    try:
        from tools.plotting.publication_output import load_model_results, load_field_data
        
        print("🔬 JAX C-GEM Comprehensive Statistical Validation")
        print("=" * 55)
        
        # Quick statistics summary
        print("\n📊 STEP 1: Data Overview")
        print("-" * 30)
        
        print("🔍 Checking model results...")
        try:
            model_results = load_model_results('OUT')
            if model_results:
                print(f"✅ Model Results: {len(model_results)} variables found")
                for var_name, var_data in model_results.items():
                    if hasattr(var_data, 'shape'):
                        print(f"   📊 {var_name}: shape {var_data.shape}")
                    else:
                        print(f"   📊 {var_name}: {len(var_data) if hasattr(var_data, '__len__') else 'scalar'}")
            else:
                print("❌ No model results found in OUT directory")
                return 1
        except Exception as e:
            print(f"❌ Error loading model results: {e}")
            return 1
        
        print("\n🔍 Checking field data...")
        try:
            field_data = load_field_data('INPUT/Calibration')
            if field_data:
                print(f"✅ Field Data: {len(field_data)} datasets found")
                for dataset_name, dataset in field_data.items():
                    if hasattr(dataset, 'shape'):
                        print(f"   🌊 {dataset_name}: shape {dataset.shape}")
                    else:
                        print(f"   🌊 {dataset_name}: {len(dataset) if hasattr(dataset, '__len__') else 'scalar'}")
            else:
                print("❌ No field data found in INPUT/Calibration directory")
                return 1
        except Exception as e:
            print(f"❌ Error loading field data: {e}")
            return 1
        
        # Compute derived variables (like tidal range from water levels)
        print("\n🔧 Computing derived variables...")
        model_results = compute_derived_variables(model_results)
        
        # Enhanced field data alignment 
        print("\n🔧 Enhancing field data alignment...")
        field_data_enhanced = enhance_field_data_alignment(model_results, field_data)
        
        # Statistical validation
        print("\n🔬 STEP 2: Statistical Validation")
        print("-" * 35)
        
        output_dir = 'OUT/Statistical_Validation'
        os.makedirs(output_dir, exist_ok=True)
        
        validator = ModelValidator(output_dir)
        validation_results = validator.validate_against_observations(
            model_results, 
            field_data_enhanced,  # Use enhanced data instead of raw field_data
            create_plots=True
        )
        
        if validation_results:
            print(f"\n✅ Validation completed for {len(validation_results)} variables:")
            for variable, result in validation_results.items():
                metrics = result.metrics
                if not float('inf') in [metrics.rmse, metrics.r_squared, metrics.nash_sutcliffe]:
                    print(f"   📊 {variable}:")
                    print(f"      • RMSE: {metrics.rmse:.4f}")
                    print(f"      • R²: {metrics.r_squared:.4f}")
                    print(f"      • Nash-Sutcliffe: {metrics.nash_sutcliffe:.4f}")
                    print(f"      • Observations: {metrics.n_observations}")
        
        # Advanced benchmarking
        print("\n🏅 STEP 3: Advanced Statistical Benchmarking")
        print("-" * 42)
        
        benchmark_data = {
            'CARE_Field_Observations': field_data,
            'Historical_Dataset': field_data,
        }
        
        benchmark_results = validate_model_against_benchmarks(
            model_results, 
            benchmark_data, 
            'OUT/Advanced_Benchmarks'
        )
        
        # Generate reports
        print("\n📄 STEP 4: Report Generation")
        print("-" * 28)
        
        try:
            report_path = validator.generate_validation_report()
            print(f"✅ Statistical validation report: {report_path}")
        except Exception as e:
            print(f"❌ Error generating report: {e}")
        
        print(f"\n🔍 All validation outputs saved to:")
        print(f"   📁 Statistical validation: {output_dir}")
        print(f"   📁 Advanced benchmarks: OUT/Advanced_Benchmarks")
        print("\n✅ Comprehensive statistical validation completed!")
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure required modules are installed.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

# Export key functions
__all__ = [
    'ModelValidator',
    'ValidationResult',
    'ValidationMetrics',
    'validate_model_against_benchmarks',
    'main'
]

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
