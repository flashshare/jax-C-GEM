"""
Gradient-based parameter calibration module for JAX C-GEM model.

This module implements sophisticated calibration methods using JAX's automatic
differentiation and advanced optimization techniques to fit the model
against observed data with a focus on system-level patterns.

Features:
- Multi-objective calibration against statistical aggregates
- Gradient-based optimization using JAX AD
- Support for complex weighted objectives
- Sparse data handling capabilities

Author: Nguyen Truong An
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import optax
import optimistix as optx
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable, Optional, Union

# Import needed model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "core"))
from config_parser import parse_model_config
from simulation_engine import run_simulation
import model_config as mc

class JAXCalibrator:
    """JAX-based gradient optimization for model calibration."""
    
    def __init__(self, model_config_path: str = "config/model_config.txt"):
        """
        Initialize calibrator with model configuration.
        
        Args:
            model_config_path: Path to model configuration file
        """
        self.config_path = model_config_path
        self.model_config = parse_model_config(model_config_path)
        
        # Parameters to calibrate (names and bounds)
        self.parameters_to_calibrate = {}
        self.extract_calibration_parameters()
        
        # Observed data
        self.observations = {
            'longitudinal': {},  # Mean profiles
            'seasonal': {},      # Monthly means
            'variability': {}    # Standard deviations
        }
        
        # Weights for different objective components
        self.weights = {
            'longitudinal': 1.0,
            'seasonal': 1.0,
            'variability': 1.0
        }
    
    def extract_calibration_parameters(self):
        """Extract parameters to calibrate from model config."""
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
                self.parameters_to_calibrate[param] = {
                    'value': self.model_config[param],
                    'min': self.model_config[min_key],
                    'max': self.model_config[max_key]
                }
    
    def load_observations(self, obs_files: Dict[str, str]):
        """
        Load observations from data files.
        
        Args:
            obs_files: Dictionary of observation file paths
                - 'longitudinal': Path to longitudinal profile data
                - 'seasonal': Path to seasonal cycle data
                - 'variability': Path to variability data
        """
        if 'longitudinal' in obs_files:
            try:
                self.observations['longitudinal'] = pd.read_csv(obs_files['longitudinal'])
                print(f"✅ Loaded longitudinal profile data")
            except Exception as e:
                print(f"❌ Error loading longitudinal data: {e}")
        
        if 'seasonal' in obs_files:
            try:
                self.observations['seasonal'] = pd.read_csv(obs_files['seasonal'])
                print(f"✅ Loaded seasonal cycle data")
            except Exception as e:
                print(f"❌ Error loading seasonal data: {e}")
        
        if 'variability' in obs_files:
            try:
                self.observations['variability'] = pd.read_csv(obs_files['variability'])
                print(f"✅ Loaded variability data")
            except Exception as e:
                print(f"❌ Error loading variability data: {e}")
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set weights for different objective components.
        
        Args:
            weights: Dictionary of weights for each component
                - 'longitudinal': Weight for longitudinal profile errors
                - 'seasonal': Weight for seasonal cycle errors
                - 'variability': Weight for variability errors
        """
        for key, value in weights.items():
            if key in self.weights:
                self.weights[key] = value
    
    def parameters_to_vector(self, params: Dict[str, Any] = None) -> jnp.ndarray:
        """
        Convert parameter dictionary to vector for optimization.
        
        Args:
            params: Parameter dictionary (uses current parameters if None)
            
        Returns:
            Parameter vector scaled to [0, 1] range
        """
        if params is None:
            params = {k: v['value'] for k, v in self.parameters_to_calibrate.items()}
        
        param_vector = []
        for name in self.parameters_to_calibrate:
            if name in params:
                # Scale to [0, 1] range
                p_min = self.parameters_to_calibrate[name]['min']
                p_max = self.parameters_to_calibrate[name]['max']
                scaled_value = (params[name] - p_min) / (p_max - p_min)
                param_vector.append(scaled_value)
            else:
                # Use middle of range if parameter not provided
                param_vector.append(0.5)
        
        return jnp.array(param_vector)
    
    def vector_to_parameters(self, param_vector: jnp.ndarray) -> Dict[str, Any]:
        """
        Convert parameter vector to dictionary for model execution.
        
        Args:
            param_vector: Parameter vector scaled to [0, 1] range
            
        Returns:
            Parameter dictionary with actual values
        """
        param_dict = {}
        param_names = list(self.parameters_to_calibrate.keys())
        
        for i, name in enumerate(param_names):
            if i < len(param_vector):
                # Convert from [0, 1] to actual range
                p_min = self.parameters_to_calibrate[name]['min']
                p_max = self.parameters_to_calibrate[name]['max']
                value = p_min + param_vector[i] * (p_max - p_min)
                param_dict[name] = value
        
        return param_dict
    
    def update_model_config(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update model configuration with new parameters.
        
        Args:
            param_dict: Dictionary of parameter values
            
        Returns:
            Updated model configuration dictionary
        """
        # Create a copy of the model config
        updated_config = dict(self.model_config)
        
        # Update with new parameter values
        for name, value in param_dict.items():
            if name in updated_config:
                updated_config[name] = value
        
        return updated_config
    
    def run_model_with_params(self, param_vector: jnp.ndarray) -> Dict[str, Any]:
        """
        Run model with given parameters.
        
        Args:
            param_vector: Parameter vector scaled to [0, 1] range
            
        Returns:
            Model results dictionary
        """
        # Convert parameter vector to dictionary
        param_dict = self.vector_to_parameters(param_vector)
        
        # Update model configuration
        updated_config = self.update_model_config(param_dict)
        
        # Run model simulation
        results = run_simulation(updated_config)
        
        return results
    
    def calculate_longitudinal_error(self, results: Dict[str, Any]) -> float:
        """
        Calculate error between model and observed longitudinal profiles.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Weighted RMSE for longitudinal profiles
        """
        if not self.observations['longitudinal']:
            return 0.0
        
        # Extract observed data
        obs_data = self.observations['longitudinal']
        
        # Extract model results - averaging over time for steady profiles
        # This implementation depends on your specific data structure
        error = 0.0
        count = 0
        
        # For each variable in observations, calculate error
        for var in obs_data['variable'].unique():
            if var == 'salinity' and 'salinity' in results:
                model_profile = jnp.mean(results['salinity'], axis=0)
                obs_points = obs_data[obs_data['variable'] == var]
                
                # Interpolate model to observation points
                # (Simplified - in practice, need proper distance-based interpolation)
                for _, row in obs_points.iterrows():
                    dist = row['distance_km']
                    value = row['value']
                    
                    # Find nearest point in model grid
                    grid_km = results['grid'] / 1000  # Convert m to km
                    idx = jnp.argmin(jnp.abs(grid_km - dist))
                    
                    # Calculate squared error
                    sq_error = (model_profile[idx] - value)**2
                    error += sq_error
                    count += 1
        
        # Calculate RMSE if we have observations
        if count > 0:
            rmse = jnp.sqrt(error / count)
            return float(rmse) * self.weights['longitudinal']
        
        return 0.0
    
    def calculate_seasonal_error(self, results: Dict[str, Any]) -> float:
        """
        Calculate error between model and observed seasonal cycles.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Weighted RMSE for seasonal cycles
        """
        if not self.observations['seasonal']:
            return 0.0
        
        # Extract observed data
        obs_data = self.observations['seasonal']
        
        # This implementation depends on your specific data structure
        error = 0.0
        count = 0
        
        # Calculate error for monthly means at specific stations
        # (Simplified - actual implementation would group by month and station)
        
        return float(error) * self.weights['seasonal']
    
    def calculate_variability_error(self, results: Dict[str, Any]) -> float:
        """
        Calculate error between model and observed variability.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Weighted RMSE for variability (standard deviations)
        """
        if not self.observations['variability']:
            return 0.0
        
        # Extract observed data
        obs_data = self.observations['variability']
        
        # This implementation depends on your specific data structure
        error = 0.0
        count = 0
        
        # Calculate error for variability (std dev) at specific stations
        # (Simplified - actual implementation would compare standard deviations)
        
        return float(error) * self.weights['variability']
    
    def objective_function(self, param_vector: jnp.ndarray) -> float:
        """
        Multi-objective function that calculates weighted sum of errors.
        
        Args:
            param_vector: Parameter vector scaled to [0, 1] range
            
        Returns:
            Total weighted error (lower is better)
        """
        # Run model with parameters
        results = self.run_model_with_params(param_vector)
        
        # Calculate individual error components
        long_error = self.calculate_longitudinal_error(results)
        seasonal_error = self.calculate_seasonal_error(results)
        variability_error = self.calculate_variability_error(results)
        
        # Compute total weighted error
        total_error = long_error + seasonal_error + variability_error
        
        return total_error
    
    def optimize_parameters(self, 
                          max_iterations: int = 50, 
                          learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Optimize parameters using gradient-based optimization.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with optimized parameters and convergence info
        """
        # Create JAX-transformed objective function with gradient
        jit_obj_fn = jax.jit(self.objective_function)
        grad_fn = jax.grad(jit_obj_fn)
        
        # Initial parameter vector
        initial_params = self.parameters_to_vector()
        
        # Set up optimizer (Adam with learning rate schedule)
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=max_iterations//10,
            decay_rate=0.9
        )
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(initial_params)
        
        # Optimization loop
        params = initial_params
        best_params = params
        best_error = float('inf')
        errors = []
        
        print(f"\n🎯 Starting parameter optimization ({max_iterations} iterations)...")
        
        for i in range(max_iterations):
            # Calculate objective value and gradient
            error = jit_obj_fn(params)
            grads = grad_fn(params)
            
            # Update parameters using optimizer
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Clip parameters to [0, 1] range
            params = jnp.clip(params, 0.0, 1.0)
            
            # Track best parameters
            if error < best_error:
                best_error = error
                best_params = params
            
            errors.append(float(error))
            
            # Print progress
            if (i+1) % 5 == 0 or i == 0:
                print(f"   Iteration {i+1}/{max_iterations}: Error = {error:.6f}")
        
        # Convert best parameters to dictionary
        best_param_dict = self.vector_to_parameters(best_params)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Initial error: {errors[0]:.6f}")
        print(f"   Final error: {best_error:.6f}")
        print(f"   Improvement: {((errors[0] - best_error) / errors[0] * 100):.2f}%")
        
        return {
            'parameters': best_param_dict,
            'error': best_error,
            'error_history': errors,
            'initial_error': errors[0]
        }
    
    def save_optimized_parameters(self, params: Dict[str, Any], output_path: str):
        """
        Save optimized parameters to file.
        
        Args:
            params: Dictionary with optimized parameters
            output_path: Path to save parameters
        """
        # Read original config file
        with open(self.config_path, 'r') as f:
            config_lines = f.readlines()
        
        # Update parameter values in config
        for i, line in enumerate(config_lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('=', 1)
            if len(parts) != 2:
                continue
                
            param_name = parts[0].strip()
            
            if param_name in params['parameters']:
                # Update parameter value in config file
                new_value = params['parameters'][param_name]
                config_lines[i] = f"{param_name} = {new_value}\n"
        
        # Write updated config
        with open(output_path, 'w') as f:
            f.writelines(config_lines)
            
            # Add optimization summary at the end
            f.write("\n# Optimization Summary\n")
            f.write(f"# Initial error: {params['initial_error']:.6f}\n")
            f.write(f"# Final error: {params['error']:.6f}\n")
            f.write(f"# Improvement: {((params['initial_error'] - params['error']) / params['initial_error'] * 100):.2f}%\n")
        
        print(f"✅ Optimized parameters saved to: {output_path}")

def main():
    """Run calibration when script is executed directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="JAX C-GEM Parameter Calibration")
    parser.add_argument("--config", default="config/model_config.txt",
                      help="Path to model configuration file")
    parser.add_argument("--output", default="config/calibrated_model_config.txt",
                      help="Path to save calibrated configuration")
    parser.add_argument("--iterations", type=int, default=50,
                      help="Maximum number of optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                      help="Initial learning rate for optimizer")
    
    args = parser.parse_args()
    
    # Create calibrator
    calibrator = JAXCalibrator(args.config)
    
    # Load observations (example paths - update as needed)
    calibrator.load_observations({
        'longitudinal': 'INPUT/Calibration/longitudinal_profiles.csv',
        'seasonal': 'INPUT/Calibration/seasonal_cycles.csv',
        'variability': 'INPUT/Calibration/variability.csv'
    })
    
    # Set component weights
    calibrator.set_weights({
        'longitudinal': 1.0,
        'seasonal': 0.8,
        'variability': 0.5
    })
    
    # Run optimization
    result = calibrator.optimize_parameters(
        max_iterations=args.iterations,
        learning_rate=args.learning_rate
    )
    
    # Save optimized parameters
    calibrator.save_optimized_parameters(result, args.output)

if __name__ == "__main__":
    main()
