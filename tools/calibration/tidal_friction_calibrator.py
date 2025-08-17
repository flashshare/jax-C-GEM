#!/usr/bin/env python
"""
Tidal Friction Coefficient Calibrator for JAX C-GEM

This script systematically tests different Chezy friction coefficients to resolve
the 2x tidal range over-prediction issue identified in Phase I validation.

Key Features:
- Systematic friction coefficient testing
- Automated model runs with different Chezy values  
- Tidal range validation against SIHYMECC observations
- Optimization to minimize tidal range errors
- Real-time progress tracking and results

Target Performance:
- BD Station: Reduce 6.20m → 2.92m (±30%)
- BK Station: Reduce 5.60m → 3.22m (±30%)
- PC Station: Reduce 6.63m → 2.07m (±30%)

Author: Phase II Hydrodynamic Calibration Team
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import subprocess
import shutil
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.config_parser import parse_model_config
    from core.main_utils import load_configurations
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

class TidalFrictionCalibrator:
    """Systematic tidal friction coefficient calibrator."""
    
    def __init__(self, base_config_file="config/model_config.txt"):
        self.base_config_file = base_config_file
        self.results_dir = Path("OUT/tidal_calibration")
        self.results_dir.mkdir(exist_ok=True)
        
        # Target tidal ranges from SIHYMECC observations
        self.target_ranges = {
            'BD': 2.92,  # Observed tidal range at BD station
            'BK': 3.22,  # Observed tidal range at BK station  
            'PC': 2.07   # Observed tidal range at PC station
        }
        
        # Current model predictions (2x over-prediction)
        self.current_ranges = {
            'BD': 6.20,
            'BK': 5.60,
            'PC': 6.63
        }
        
        # Friction coefficient test ranges
        self.chezy_test_ranges = {
            'Chezy1': np.arange(12.0, 22.0, 2.0),  # Downstream: 12,14,16,18,20
            'Chezy2': np.arange(18.0, 30.0, 2.0)   # Upstream: 18,20,22,24,26,28
        }
        
        self.calibration_results = []
        
    def create_test_config(self, chezy1: float, chezy2: float) -> str:
        """Create a test configuration file with modified Chezy coefficients."""
        
        # Read base configuration
        with open(self.base_config_file, 'r') as f:
            config_content = f.read()
        
        # Replace Chezy coefficients
        config_content = config_content.replace(
            f"Chezy1 = 25.0", f"Chezy1 = {chezy1:.1f}"
        )
        config_content = config_content.replace(
            f"Chezy2 = 35.0", f"Chezy2 = {chezy2:.1f}"
        )
        
        # Create test config filename
        test_config = f"config/model_config_chezy{chezy1:.0f}_{chezy2:.0f}.txt"
        
        # Write test configuration
        with open(test_config, 'w') as f:
            f.write(config_content)
            
        return test_config
    
    def run_model_simulation(self, config_file: str) -> bool:
        """Run JAX C-GEM simulation with specified configuration."""
        
        try:
            # Run model with specified config
            cmd = [
                "python", "src/main.py",
                "--mode", "run",
                "--output-format", "auto",
                "--no-physics-check"
            ]
            
            print(f"🚀 Running simulation with config: {config_file}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Simulation completed successfully")
                return True
            else:
                print(f"❌ Simulation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Simulation timed out (5 minutes)")
            return False
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            return False
    
    def extract_tidal_ranges(self) -> Dict[str, float]:
        """Extract tidal ranges from simulation results."""
        
        try:
            # Load simulation results
            results_file = "OUT/complete_simulation_results.npz"
            results = np.load(results_file)
            
            # Extract water levels
            H_data = results['H']  # Shape: (time_steps, grid_points)
            
            # Station locations (indices in grid)
            station_indices = {
                'BD': 65,   # ~130km from mouth
                'BK': 78,   # ~156km from mouth
                'PC': 43    # ~86km from mouth
            }
            
            # Calculate tidal ranges at each station
            tidal_ranges = {}
            
            for station, idx in station_indices.items():
                # Extract water levels at station
                h_station = H_data[:, idx]
                
                # Calculate tidal range (max - min over simulation)
                tidal_range = np.max(h_station) - np.min(h_station)
                tidal_ranges[station] = tidal_range
                
            return tidal_ranges
            
        except Exception as e:
            print(f"❌ Error extracting tidal ranges: {e}")
            return {}
    
    def calculate_error_metrics(self, predicted_ranges: Dict[str, float]) -> Dict[str, float]:
        """Calculate validation metrics for tidal ranges."""
        
        errors = {}
        total_rmse = 0
        total_mae = 0
        n_stations = len(self.target_ranges)
        
        for station in self.target_ranges:
            if station in predicted_ranges:
                target = self.target_ranges[station]
                predicted = predicted_ranges[station]
                
                # Calculate individual station errors
                rmse = (predicted - target) ** 2
                mae = abs(predicted - target)
                rel_error = abs(predicted - target) / target * 100
                
                errors[f'{station}_rmse'] = rmse
                errors[f'{station}_mae'] = mae
                errors[f'{station}_rel_error'] = rel_error
                
                total_rmse += rmse
                total_mae += mae
        
        # Overall metrics
        errors['overall_rmse'] = np.sqrt(total_rmse / n_stations)
        errors['overall_mae'] = total_mae / n_stations
        errors['overall_rel_error'] = np.mean([
            errors[f'{station}_rel_error'] for station in self.target_ranges
            if f'{station}_rel_error' in errors
        ])
        
        return errors
    
    def run_calibration_grid_search(self):
        """Run systematic grid search over Chezy coefficient combinations."""
        
        print("🌊 Starting Tidal Friction Calibration")
        print("=" * 50)
        print(f"Target Ranges: {self.target_ranges}")
        print(f"Current Ranges: {self.current_ranges}")
        print(f"Chezy1 test range: {self.chezy_test_ranges['Chezy1']}")
        print(f"Chezy2 test range: {self.chezy_test_ranges['Chezy2']}")
        print()
        
        best_error = float('inf')
        best_config = None
        best_ranges = None
        
        # Grid search over Chezy combinations
        total_tests = len(self.chezy_test_ranges['Chezy1']) * len(self.chezy_test_ranges['Chezy2'])
        test_count = 0
        
        for chezy1 in self.chezy_test_ranges['Chezy1']:
            for chezy2 in self.chezy_test_ranges['Chezy2']:
                
                test_count += 1
                print(f"🧪 Test {test_count}/{total_tests}: Chezy1={chezy1:.1f}, Chezy2={chezy2:.1f}")
                
                # Create test configuration
                test_config = self.create_test_config(chezy1, chezy2)
                
                # Run simulation
                if self.run_model_simulation(test_config):
                    
                    # Extract tidal ranges
                    predicted_ranges = self.extract_tidal_ranges()
                    
                    if predicted_ranges:
                        # Calculate error metrics
                        errors = self.calculate_error_metrics(predicted_ranges)
                        
                        # Store results
                        result = {
                            'chezy1': chezy1,
                            'chezy2': chezy2,
                            'predicted_ranges': predicted_ranges,
                            'target_ranges': self.target_ranges,
                            'errors': errors,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.calibration_results.append(result)
                        
                        # Check if this is the best result so far
                        overall_error = errors['overall_rmse']
                        if overall_error < best_error:
                            best_error = overall_error
                            best_config = (chezy1, chezy2)
                            best_ranges = predicted_ranges
                            
                            print(f"✅ NEW BEST: RMSE={overall_error:.3f}")
                            print(f"   Predicted: {predicted_ranges}")
                            print(f"   Errors: BD={errors.get('BD_rel_error', 0):.1f}%, "
                                  f"BK={errors.get('BK_rel_error', 0):.1f}%, "
                                  f"PC={errors.get('PC_rel_error', 0):.1f}%")
                        else:
                            print(f"   RMSE={overall_error:.3f} (not best)")
                            
                    else:
                        print("❌ Failed to extract tidal ranges")
                else:
                    print("❌ Simulation failed")
                    
                # Clean up test config
                if Path(test_config).exists():
                    Path(test_config).unlink()
                    
                print()
        
        # Report best results
        if best_config and best_ranges:
            print("🏆 CALIBRATION COMPLETED")
            print("=" * 50)
            print(f"Best Configuration: Chezy1={best_config[0]:.1f}, Chezy2={best_config[1]:.1f}")
            print(f"Best RMSE: {best_error:.3f} m")
            print("Tidal Range Comparison:")
            for station in self.target_ranges:
                if station in best_ranges:
                    target = self.target_ranges[station]
                    predicted = best_ranges[station]
                    error = abs(predicted - target) / target * 100
                    print(f"  {station}: {target:.2f}m (target) → {predicted:.2f}m (model) | Error: {error:.1f}%")
                    
            # Save best configuration
            self.save_best_configuration(best_config[0], best_config[1])
            
        else:
            print("❌ No successful calibration found")
            
        # Save all results
        self.save_calibration_results()
        self.plot_calibration_results()
    
    def save_best_configuration(self, chezy1: float, chezy2: float):
        """Save the best configuration to a new file."""
        
        # Read base configuration
        with open(self.base_config_file, 'r') as f:
            config_content = f.read()
        
        # Replace Chezy coefficients
        config_content = config_content.replace(
            f"Chezy1 = 25.0", f"Chezy1 = {chezy1:.1f}    # CALIBRATED for tidal ranges"
        )
        config_content = config_content.replace(
            f"Chezy2 = 35.0", f"Chezy2 = {chezy2:.1f}    # CALIBRATED for tidal ranges"
        )
        
        # Save calibrated configuration
        calibrated_config = "config/model_config_tidal_calibrated.txt"
        with open(calibrated_config, 'w') as f:
            f.write(config_content)
            
        print(f"💾 Best configuration saved to: {calibrated_config}")
    
    def save_calibration_results(self):
        """Save calibration results to JSON file."""
        
        results_file = self.results_dir / "tidal_calibration_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.calibration_results, f, indent=2)
            
        print(f"💾 Calibration results saved to: {results_file}")
    
    def plot_calibration_results(self):
        """Create visualization of calibration results."""
        
        if not self.calibration_results:
            return
            
        # Extract data for plotting
        chezy1_values = [r['chezy1'] for r in self.calibration_results]
        chezy2_values = [r['chezy2'] for r in self.calibration_results]
        rmse_values = [r['errors']['overall_rmse'] for r in self.calibration_results]
        
        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Parameter space exploration
        scatter = ax1.scatter(chezy1_values, chezy2_values, c=rmse_values, 
                            cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Chezy1 (Downstream)')
        ax1.set_ylabel('Chezy2 (Upstream)')
        ax1.set_title('Friction Coefficient Calibration Results')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('RMSE (m)')
        
        # Mark best result
        if rmse_values:
            best_idx = np.argmin(rmse_values)
            ax1.scatter(chezy1_values[best_idx], chezy2_values[best_idx], 
                       color='red', s=200, marker='*', 
                       label=f'Best: ({chezy1_values[best_idx]:.1f}, {chezy2_values[best_idx]:.1f})')
            ax1.legend()
        
        # 2. Tidal range comparison for best result
        if self.calibration_results:
            best_result = min(self.calibration_results, key=lambda x: x['errors']['overall_rmse'])
            
            stations = list(self.target_ranges.keys())
            targets = [self.target_ranges[s] for s in stations]
            predictions = [best_result['predicted_ranges'][s] for s in stations]
            
            x = np.arange(len(stations))
            width = 0.35
            
            ax2.bar(x - width/2, targets, width, label='Observed', alpha=0.7)
            ax2.bar(x + width/2, predictions, width, label='Model (Best)', alpha=0.7)
            
            ax2.set_xlabel('Station')
            ax2.set_ylabel('Tidal Range (m)')
            ax2.set_title('Tidal Range Validation - Best Configuration')
            ax2.set_xticks(x)
            ax2.set_xticklabels(stations)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add error annotations
            for i, (target, pred) in enumerate(zip(targets, predictions)):
                error = abs(pred - target) / target * 100
                ax2.annotate(f'{error:.1f}%', 
                           xy=(i, max(target, pred) + 0.1),
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "tidal_calibration_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Calibration analysis plot saved to: {plot_file}")

def main():
    """Run tidal friction calibration."""
    
    print("🌊 JAX C-GEM Tidal Friction Calibrator")
    print("=" * 50)
    
    # Initialize calibrator
    calibrator = TidalFrictionCalibrator()
    
    # Run calibration
    calibrator.run_calibration_grid_search()
    
    print("\n✅ Tidal friction calibration completed!")

if __name__ == "__main__":
    main()