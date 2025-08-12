"""
Sensitivity Analysis Tool for JAX C-GEM

This module provides comprehensive sensitivity analysis capabilities for the JAX C-GEM model,
allowing for systematic evaluation of parameter impacts and model response to varying inputs.

Features:
- Local and global sensitivity analysis
- Automated parameter perturbation
- Visualization of sensitivity results
- Statistical summary of parameter importance

Author: JAX C-GEM Team
"""

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import time

# Add src directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from config_parser import parse_model_config
from simulation_engine import run_simulation

class SensitivityAnalyzer:
    """Sensitivity analysis tools for JAX C-GEM."""
    
    def __init__(self, model_config_path: str = "config/model_config.txt"):
        """
        Initialize sensitivity analyzer with model configuration.
        
        Args:
            model_config_path: Path to model configuration file
        """
        self.config_path = model_config_path
        self.model_config = parse_model_config(model_config_path)
        
        # Parameters to analyze (names and bounds)
        self.parameters = {}
        self.extract_parameters()
        
        # Output metrics to track
        self.metrics = {
            'salinity_intrusion': self._calculate_salinity_intrusion,
            'tidal_damping': self._calculate_tidal_damping,
            'oxygen_minimum': self._calculate_oxygen_minimum,
            'residence_time': self._calculate_residence_time
        }
    
    def extract_parameters(self):
        """Extract parameters with bounds from model config."""
        # Look for parameters with _min and _max suffixes
        param_names = set()
        
        for key in self.model_config:
            # Strip _min and _max suffixes to get base parameter names
            base_name = None
            if key.endswith('_min'):
                base_name = key[:-4]
            elif key.endswith('_max'):
                base_name = key[:-4]
            
            if base_name and base_name in self.model_config:
                param_names.add(base_name)
        
        # Create parameter dictionary with bounds
        for param in param_names:
            min_key = f"{param}_min"
            max_key = f"{param}_max"
            
            if min_key in self.model_config and max_key in self.model_config:
                self.parameters[param] = {
                    'value': self.model_config[param],
                    'min': self.model_config[min_key],
                    'max': self.model_config[max_key],
                    'type': self._guess_parameter_type(param)
                }
    
    def _guess_parameter_type(self, param_name: str) -> str:
        """
        Guess parameter type from its name.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Parameter type string
        """
        if param_name.startswith('LC'):
            return 'hydrodynamic'
        elif param_name.startswith('Chezy'):
            return 'hydrodynamic'
        elif param_name.startswith('Rs'):
            return 'hydrodynamic'
        elif param_name.startswith('tau'):
            return 'sediment'
        elif param_name.startswith('Mero'):
            return 'sediment'
        elif param_name.startswith('K'):
            return 'biogeochemical'
        else:
            return 'other'
    
    def select_parameters(self, param_list: List[str] = None) -> Dict[str, Dict]:
        """
        Select subset of parameters to analyze.
        
        Args:
            param_list: List of parameter names (None for all)
            
        Returns:
            Dictionary of selected parameters
        """
        if param_list is None:
            return self.parameters
        
        selected = {}
        for name in param_list:
            if name in self.parameters:
                selected[name] = self.parameters[name]
        
        return selected
    
    def _calculate_salinity_intrusion(self, results: Dict[str, Any]) -> float:
        """
        Calculate salinity intrusion length.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Salinity intrusion length [km]
        """
        try:
            # Get last time step salinity profile
            if 'conc_save' in results:
                salinity = np.array(results['conc_save'])[-1, 9, :]  # Assuming salinity is index 9
            elif 'salinity' in results:
                salinity = np.array(results['salinity'])[-1, :]
            else:
                return 0.0
            
            # Get grid coordinates
            if 'x_grid' in results:
                grid = np.array(results['x_grid'])
            else:
                grid = np.arange(len(salinity)) * self.model_config.get('DELXI', 1000)
            
            # Find where salinity drops below 1 PSU threshold
            intrusion_idx = np.where(salinity < 1.0)[0]
            if len(intrusion_idx) > 0:
                # First point where salinity < 1 PSU
                return grid[intrusion_idx[0]] / 1000  # Convert to km
            else:
                # Salt intrudes entire domain
                return grid[-1] / 1000
        except Exception:
            return 0.0
    
    def _calculate_tidal_damping(self, results: Dict[str, Any]) -> float:
        """
        Calculate tidal damping rate.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Tidal damping coefficient
        """
        try:
            # Get water levels
            if 'h_save' in results:
                water_levels = np.array(results['h_save'])
            else:
                return 0.0
            
            # Calculate tidal range at mouth and 10km upstream
            ranges_at_points = np.max(water_levels, axis=0) - np.min(water_levels, axis=0)
            mouth_range = ranges_at_points[0]
            
            # Find point approximately 10km from mouth
            dx = self.model_config.get('DELXI', 1000)
            point_10km = int(10000 / dx)
            if point_10km >= len(ranges_at_points):
                point_10km = len(ranges_at_points) // 2
            
            upstream_range = ranges_at_points[point_10km]
            
            # Calculate damping
            if mouth_range > 0:
                damping = 1 - (upstream_range / mouth_range)
                return damping
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _calculate_oxygen_minimum(self, results: Dict[str, Any]) -> float:
        """
        Calculate minimum oxygen concentration.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Minimum oxygen concentration
        """
        try:
            # Get oxygen concentrations
            if 'conc_save' in results:
                oxygen = np.array(results['conc_save'])[:, 7, :]  # Assuming oxygen is index 7
            elif 'oxygen' in results:
                oxygen = np.array(results['oxygen'])
            else:
                return 0.0
            
            # Calculate minimum value (over space and time)
            return np.min(oxygen)
        except Exception:
            return 0.0
    
    def _calculate_residence_time(self, results: Dict[str, Any]) -> float:
        """
        Estimate residence time from velocity field.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Estimated residence time [days]
        """
        try:
            # Get velocities
            if 'u_save' in results:
                velocities = np.array(results['u_save'])
            else:
                return 0.0
            
            # Calculate mean velocity (absolute value)
            mean_velocity = np.mean(np.abs(velocities))
            if mean_velocity > 0:
                # Rough estimate: domain length divided by mean velocity
                domain_length = self.model_config.get('EL', 100000)  # m
                residence_seconds = domain_length / mean_velocity
                return residence_seconds / 86400  # Convert to days
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def run_model_with_params(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Run model with modified parameters.
        
        Args:
            param_dict: Dictionary of parameter values
            
        Returns:
            Model results dictionary
        """
        # Create a copy of the model config
        config = dict(self.model_config)
        
        # Update with new parameter values
        for name, value in param_dict.items():
            if name in config:
                config[name] = value
        
        # Run reduced simulation for sensitivity analysis
        # Shorter duration to speed up analysis
        config['MAXT'] = config.get('MAXT', 10) / 2  # Half the simulation time
        
        # Run model simulation
        results = run_simulation(config)
        
        return results
    
    def one_at_a_time_sensitivity(self, 
                                param_list: List[str] = None,
                                n_samples: int = 5) -> Dict[str, Dict[str, List]]:
        """
        Perform one-at-a-time sensitivity analysis.
        
        Args:
            param_list: List of parameters to analyze (None for all)
            n_samples: Number of samples per parameter
            
        Returns:
            Dictionary of results for each parameter and metric
        """
        selected_params = self.select_parameters(param_list)
        if not selected_params:
            print("No parameters selected for analysis")
            return {}
        
        results = {metric: {param: [] for param in selected_params} 
                 for metric in self.metrics}
        param_values = {param: [] for param in selected_params}
        
        print(f"\n🔍 Running one-at-a-time sensitivity analysis...")
        print(f"   Parameters: {', '.join(selected_params.keys())}")
        print(f"   Metrics: {', '.join(self.metrics.keys())}")
        
        # Run baseline model
        print("\n📊 Running baseline model...")
        baseline_params = {name: info['value'] for name, info in selected_params.items()}
        baseline_results = self.run_model_with_params(baseline_params)
        
        # Calculate baseline metric values
        baseline_metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            baseline_metrics[metric_name] = metric_fn(baseline_results)
            print(f"   Baseline {metric_name}: {baseline_metrics[metric_name]:.4f}")
        
        # One-at-a-time parameter variations
        total_runs = len(selected_params) * n_samples
        run_count = 0
        
        for param_name, param_info in selected_params.items():
            print(f"\n📊 Analyzing parameter: {param_name}")
            
            # Generate parameter samples between min and max
            p_min = param_info['min']
            p_max = param_info['max']
            samples = np.linspace(p_min, p_max, n_samples)
            
            for i, value in enumerate(samples):
                run_count += 1
                print(f"   Run {run_count}/{total_runs}: {param_name} = {value:.4f}")
                
                # Set up parameter dictionary with this value
                param_dict = dict(baseline_params)
                param_dict[param_name] = value
                
                # Run model with this parameter value
                run_results = self.run_model_with_params(param_dict)
                
                # Store parameter value
                param_values[param_name].append(value)
                
                # Calculate and store metrics
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(run_results)
                    results[metric_name][param_name].append(metric_value)
                    print(f"     {metric_name}: {metric_value:.4f}")
        
        # Format results
        sensitivity_results = {
            'parameters': param_values,
            'metrics': results,
            'baseline': baseline_metrics
        }
        
        return sensitivity_results
    
    def global_sensitivity_analysis(self, 
                                  param_list: List[str] = None,
                                  n_samples: int = 100) -> Dict[str, Any]:
        """
        Perform global sensitivity analysis using Sobol method.
        
        Args:
            param_list: List of parameters to analyze (None for all)
            n_samples: Number of samples for analysis
            
        Returns:
            Dictionary of Sobol sensitivity indices
        """
        selected_params = self.select_parameters(param_list)
        if not selected_params:
            print("No parameters selected for analysis")
            return {}
        
        print(f"\n🔍 Running global sensitivity analysis (Sobol method)...")
        print(f"   Parameters: {', '.join(selected_params.keys())}")
        print(f"   Metrics: {', '.join(self.metrics.keys())}")
        print(f"   Samples: {n_samples} (total model runs: {n_samples * (2 * len(selected_params) + 2)})")
        
        # Define parameter problem for SALib
        problem = {
            'num_vars': len(selected_params),
            'names': list(selected_params.keys()),
            'bounds': [[selected_params[p]['min'], selected_params[p]['max']] 
                     for p in selected_params]
        }
        
        # Generate samples using Saltelli's extension of Sobol sequence
        param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
        
        # Run models for all parameter sets
        total_runs = len(param_values)
        print(f"\n📊 Running {total_runs} model evaluations...")
        
        metric_values = {metric: [] for metric in self.metrics}
        start_time = time.time()
        
        for i, X in enumerate(param_values):
            # Create parameter dictionary
            param_dict = {name: X[j] for j, name in enumerate(selected_params.keys())}
            
            # Print progress
            if (i+1) % 10 == 0 or i == 0 or i == total_runs-1:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i+1)) * (total_runs - (i+1)) if i > 0 else 0
                print(f"   Run {i+1}/{total_runs} - Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
            
            # Run model
            results = self.run_model_with_params(param_dict)
            
            # Calculate metrics
            for metric_name, metric_fn in self.metrics.items():
                metric_values[metric_name].append(metric_fn(results))
        
        # Calculate Sobol sensitivity indices
        sensitivity_indices = {}
        
        for metric_name, Y in metric_values.items():
            print(f"\n📊 Calculating sensitivity indices for {metric_name}...")
            
            try:
                Si = sobol.analyze(problem, np.array(Y), calc_second_order=False, print_to_console=False)
                
                sensitivity_indices[metric_name] = {
                    'S1': {p: s for p, s in zip(problem['names'], Si['S1'])},
                    'ST': {p: s for p, s in zip(problem['names'], Si['ST'])},
                    'S1_conf': {p: s for p, s in zip(problem['names'], Si['S1_conf'])},
                    'ST_conf': {p: s for p, s in zip(problem['names'], Si['ST_conf'])}
                }
                
                # Print first-order indices
                print(f"   First-order indices (direct effect):")
                for p, s in sensitivity_indices[metric_name]['S1'].items():
                    print(f"     {p}: {s:.4f} ± {sensitivity_indices[metric_name]['S1_conf'][p]:.4f}")
                    
                # Print total-effect indices
                print(f"   Total-effect indices (including interactions):")
                for p, s in sensitivity_indices[metric_name]['ST'].items():
                    print(f"     {p}: {s:.4f} ± {sensitivity_indices[metric_name]['ST_conf'][p]:.4f}")
                    
            except Exception as e:
                print(f"Error calculating sensitivity indices for {metric_name}: {str(e)}")
                sensitivity_indices[metric_name] = {}
        
        return {
            'indices': sensitivity_indices,
            'problem': problem,
            'metrics': metric_values
        }
    
    def plot_one_at_a_time_results(self, results: Dict[str, Any], 
                                 output_dir: str = "OUT/Sensitivity") -> List[str]:
        """
        Create plots for one-at-a-time sensitivity analysis.
        
        Args:
            results: Results from one_at_a_time_sensitivity()
            output_dir: Directory to save plots
            
        Returns:
            List of saved plot filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        # Extract data
        param_values = results['parameters']
        metrics = results['metrics']
        baseline = results['baseline']
        
        # Plot each metric
        for metric_name in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for param_name, values in metrics[metric_name].items():
                # Normalize parameter values to [0, 1] for comparison
                param_min = min(param_values[param_name])
                param_max = max(param_values[param_name])
                norm_values = [(v - param_min) / (param_max - param_min) if param_max > param_min else 0.5 
                             for v in param_values[param_name]]
                
                ax.plot(norm_values, values, 'o-', label=param_name)
            
            # Add baseline value as horizontal line
            if metric_name in baseline:
                ax.axhline(y=baseline[metric_name], color='k', linestyle='--', 
                         label=f'Baseline ({baseline[metric_name]:.4f})')
            
            ax.set_xlabel('Normalized Parameter Value (0=min, 1=max)')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'One-at-a-Time Sensitivity: {metric_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(output_dir, f'sensitivity_oat_{metric_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(plot_path)
        
        return saved_files
    
    def plot_sobol_indices(self, results: Dict[str, Any], 
                         output_dir: str = "OUT/Sensitivity") -> List[str]:
        """
        Create bar plots for Sobol sensitivity indices.
        
        Args:
            results: Results from global_sensitivity_analysis()
            output_dir: Directory to save plots
            
        Returns:
            List of saved plot filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        # Extract indices
        indices = results['indices']
        
        # Plot each metric
        for metric_name, indices_dict in indices.items():
            if 'S1' not in indices_dict or 'ST' not in indices_dict:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get parameters and indices
            params = list(indices_dict['S1'].keys())
            s1_values = [indices_dict['S1'][p] for p in params]
            st_values = [indices_dict['ST'][p] for p in params]
            
            # Sort by total effect
            sorted_indices = np.argsort(st_values)[::-1]  # Descending
            sorted_params = [params[i] for i in sorted_indices]
            sorted_s1 = [s1_values[i] for i in sorted_indices]
            sorted_st = [st_values[i] for i in sorted_indices]
            
            # Plot bars
            x = np.arange(len(sorted_params))
            width = 0.35
            
            ax.bar(x - width/2, sorted_s1, width, label='First-Order', color='#3498db')
            ax.bar(x + width/2, sorted_st, width, label='Total-Effect', color='#e74c3c')
            
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Sensitivity Index')
            ax.set_title(f'Sobol Sensitivity Indices: {metric_name.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_params, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Ensure y axis starts at 0 and includes all values
            ax.set_ylim(0, max(1.0, max(sorted_st) * 1.1))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f'sensitivity_sobol_{metric_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(plot_path)
        
        return saved_files

def main():
    """Run sensitivity analysis when script is executed directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="JAX C-GEM Sensitivity Analysis")
    parser.add_argument("--config", default="config/model_config.txt",
                      help="Path to model configuration file")
    parser.add_argument("--mode", choices=['oat', 'sobol'], default='oat',
                      help="Analysis mode: one-at-a-time (oat) or global Sobol (sobol)")
    parser.add_argument("--parameters", nargs='+',
                      help="Parameters to analyze (default: all)")
    parser.add_argument("--samples", type=int, default=5,
                      help="Number of samples per parameter")
    parser.add_argument("--output-dir", default="OUT/Sensitivity",
                      help="Directory for output plots")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SensitivityAnalyzer(args.config)
    
    if args.mode == 'oat':
        # Run one-at-a-time analysis
        results = analyzer.one_at_a_time_sensitivity(
            param_list=args.parameters,
            n_samples=args.samples
        )
        
        # Plot results
        saved_plots = analyzer.plot_one_at_a_time_results(
            results, 
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")
        print(f"   Created {len(saved_plots)} plots:")
        for plot in saved_plots:
            print(f"   - {os.path.basename(plot)}")
        
    elif args.mode == 'sobol':
        # Run global sensitivity analysis
        results = analyzer.global_sensitivity_analysis(
            param_list=args.parameters,
            n_samples=args.samples
        )
        
        # Plot results
        saved_plots = analyzer.plot_sobol_indices(
            results, 
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")
        print(f"   Created {len(saved_plots)} plots:")
        for plot in saved_plots:
            print(f"   - {os.path.basename(plot)}")
        
if __name__ == "__main__":
    main()
