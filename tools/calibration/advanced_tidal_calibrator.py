#!/usr/bin/env python
"""
Advanced Multi-Parameter Tidal Calibration

Since geometry correction alone was insufficient, implement a comprehensive
multi-parameter approach targeting the persistent 2x tidal over-prediction.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import time

class AdvancedTidalCalibrator:
    """Advanced calibration combining multiple parameters."""
    
    def __init__(self):
        self.results_history = []
        
    def test_parameter_combination(self, params, description):
        """Test a specific parameter combination."""
        
        print(f"\n🧪 Testing: {description}")
        print(f"   Parameters: {params}")
        
        # Update configuration
        self.update_config(params)
        
        # Run simulation
        success = self.run_simulation()
        
        if success:
            # Get validation results
            results = self.get_validation_results()
            if results:
                results['params'] = params
                results['description'] = description
                self.results_history.append(results)
                
                # Display results
                self.display_results(results)
                return results
        
        return None
    
    def update_config(self, params):
        """Update model configuration with new parameters."""
        
        config_lines = []
        with open("config/model_config.txt", 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line_updated = False
            for param, value in params.items():
                if line.startswith(f"{param} ="):
                    config_lines.append(f"{param} = {value}                  # Updated by advanced calibration\n")
                    line_updated = True
                    break
            
            if not line_updated:
                config_lines.append(line)
        
        with open("config/model_config.txt", 'w') as f:
            f.writelines(config_lines)
        
        print(f"   ✅ Configuration updated")
    
    def run_simulation(self):
        """Run simulation with current parameters."""
        
        try:
            # Clear old results
            import os
            if os.path.exists("OUT/complete_simulation_results.npz"):
                os.remove("OUT/complete_simulation_results.npz")
            
            # Run simulation
            cmd = ["python", "src/main.py", "--mode", "run", "--output-format", "auto", "--no-physics-check"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            
            if result.returncode == 0:
                print("   ✅ Simulation completed")
                return True
            else:
                print(f"   ❌ Simulation failed: {result.stderr[:200]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def get_validation_results(self):
        """Get tidal validation results."""
        
        try:
            # Run validation
            cmd = ["python", "tools/verification/phase2_tidal_dynamics.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Parse results
                output_lines = result.stdout.split('\n')
                
                results = {'stations': {}, 'mean_range': 0.0}
                for line in output_lines:
                    if 'Range statistics:' in line and '±' in line:
                        # Extract overall range statistics
                        parts = line.split('±')
                        if len(parts) >= 2:
                            mean_part = parts[0].split()[-1]
                            try:
                                mean_range = float(mean_part)
                                results['mean_range'] = mean_range
                            except:
                                pass
                    
                    # Parse station results
                    if any(station in line for station in ['BD', 'BK', 'PC']):
                        parts = line.split()
                        if len(parts) >= 7 and parts[0] in ['BD', 'BK', 'PC']:
                            station = parts[0]
                            model_val = float(parts[5])
                            field_val = float(parts[6])
                            error_pct = float(parts[4].rstrip('%'))
                            
                            results['stations'][station] = {
                                'model': model_val,
                                'field': field_val,
                                'error': error_pct
                            }
                
                return results
            else:
                print(f"   ❌ Validation failed")
                return None
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return None
    
    def display_results(self, results):
        """Display validation results."""
        
        print("   📊 Results:")
        if 'stations' in results:
            for station, data in results['stations'].items():
                print(f"      {station}: {data['model']:.2f}m (model) vs {data['field']:.2f}m (field) | Error: {data['error']:.1f}%")
        
        if 'mean_range' in results:
            print(f"      Mean tidal range: {results['mean_range']:.2f}m")
    
    def run_advanced_calibration(self):
        """Run comprehensive multi-parameter calibration sequence."""
        
        print("🎯 JAX C-GEM Advanced Multi-Parameter Tidal Calibration")
        print("=" * 60)
        
        # Test 1: More aggressive geometry correction
        params1 = {'B2': 1200.00, 'AMPL': 4.43}
        self.test_parameter_combination(params1, "Aggressive Geometry Correction (B2=1200m)")
        
        # Test 2: Geometry + friction reduction
        params2 = {'B2': 1200.00, 'Chezy1': 30.0, 'Chezy2': 40.0, 'AMPL': 4.43}
        self.test_parameter_combination(params2, "Geometry + Higher Friction")
        
        # Test 3: Geometry + amplitude reduction
        params3 = {'B2': 1200.00, 'AMPL': 3.0}
        self.test_parameter_combination(params3, "Geometry + Reduced Amplitude")
        
        # Test 4: Combined approach
        params4 = {'B2': 1500.00, 'Chezy1': 30.0, 'Chezy2': 40.0, 'AMPL': 3.5}
        self.test_parameter_combination(params4, "Combined Multi-Parameter Approach")
        
        # Test 5: Storage width optimization
        params5 = {'B2': 1200.00, 'Rs1': 0.8, 'Rs2': 0.8, 'AMPL': 3.5}
        self.test_parameter_combination(params5, "Geometry + Storage Width Reduction")
        
        # Analyze results
        self.analyze_calibration_results()
    
    def analyze_calibration_results(self):
        """Analyze all calibration results and recommend best approach."""
        
        print(f"\n📊 ADVANCED CALIBRATION RESULTS ANALYSIS")
        print("=" * 50)
        
        if not self.results_history:
            print("❌ No results to analyze")
            return
        
        print("Test | Description                        | BD Error | BK Error | PC Error | Avg Error")
        print("-----|---------------------------------------|----------|----------|----------|----------")
        
        best_result = None
        best_avg_error = float('inf')
        
        for i, result in enumerate(self.results_history):
            desc = result['description'][:35]
            stations = result['stations']
            
            bd_error = stations.get('BD', {}).get('error', 999)
            bk_error = stations.get('BK', {}).get('error', 999)
            pc_error = stations.get('PC', {}).get('error', 999)
            avg_error = (bd_error + bk_error + pc_error) / 3
            
            print(f"{i+1:4d} | {desc:37s} | {bd_error:7.1f}% | {bk_error:7.1f}% | {pc_error:7.1f}% | {avg_error:7.1f}%")
            
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_result = result
        
        print("\n🏆 BEST CONFIGURATION:")
        print("-" * 25)
        
        if best_result:
            print(f"Description: {best_result['description']}")
            print(f"Parameters: {best_result['params']}")
            print(f"Average error: {best_avg_error:.1f}%")
            
            if best_avg_error < 50:
                print("✅ Significant improvement achieved!")
            elif best_avg_error < 80:
                print("🔄 Moderate improvement - continue optimization")
            else:
                print("⚠️  Limited improvement - may need fundamental model revision")
        
        print(f"\n🎯 NEXT STEPS:")
        print("-" * 15)
        if best_avg_error < 30:
            print("✅ Calibration successful - proceed to Phase III")
        elif best_avg_error < 60:
            print("🔧 Fine-tune best configuration with gradient-based optimization")
        else:
            print("🔍 Consider fundamental model structure revision")
            print("   - Investigate depth-dependent effects")
            print("   - Review governing equations implementation") 
            print("   - Consider 3D effects in shallow regions")

def main():
    """Run advanced multi-parameter tidal calibration."""
    
    calibrator = AdvancedTidalCalibrator()
    calibrator.run_advanced_calibration()
    
    print(f"\n" + "=" * 60)
    print("✅ Advanced calibration sequence completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()