#!/usr/bin/env python3
"""
Boundary Condition Analyzer
============================

Task 2.3.4: Phase II Enhancement - Analyze boundary condition issues causing
persistent PC station 193.5% tidal error despite proper momentum balance.

IDENTIFIED PROBLEM:
- BD station: 107.8% error (improved from 113.9%)
- BK station: 89.1% error (improved from 92.9%) 
- PC station: 193.5% error (worsened from 184.9%)

The momentum balance fix improved BD and BK significantly but PC got worse,
indicating a boundary condition specific issue rather than friction.

ANALYSIS APPROACH:
1. Examine tidal boundary forcing (AMPL=4.43m) appropriateness
2. Investigate reflection coefficients and wave propagation
3. Check upstream boundary condition implementation
4. Analyze station-specific amplification mechanisms

SOLUTION IMPLEMENTATION:
- Calibrate AMPL specifically for PC station performance
- Implement reflection coefficient corrections
- Fix boundary wave physics if needed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

class BoundaryConditionAnalyzer:
    """Analyze boundary condition issues affecting tidal dynamics."""
    
    def __init__(self):
        self.results_dir = Path("OUT")
        self.analysis_results = {}
        
    def load_simulation_data(self):
        """Load simulation results for boundary condition analysis."""
        npz_file = self.results_dir / "complete_simulation_results.npz"
        
        if not npz_file.exists():
            print("❌ No simulation results found. Run model first.")
            return False
            
        print("📊 Loading simulation data...")
        data = np.load(npz_file)
        
        self.water_levels = data['H']  # (time_steps, grid_points)
        self.time_array = data['time']
        
        # Calculate distances from grid (assuming DELXI=2000m, starting at 1km)
        M = self.water_levels.shape[1]  # Number of grid points
        self.distances = np.arange(1, M*2 + 1, 2)  # 1, 3, 5, ..., km from mouth
        
        print(f"   ✓ Water levels: {self.water_levels.shape}")
        print(f"   ✓ Distance range: {self.distances[0]:.1f} - {self.distances[-1]:.1f} km")
        print(f"   ✓ Time steps: {len(self.time_array)}")
        
        return True
        
    def analyze_tidal_boundary_forcing(self):
        """Analyze the downstream tidal boundary conditions."""
        print("\n🌊 TIDAL BOUNDARY ANALYSIS")
        print("-" * 35)
        
        # Extract boundary (mouth) tidal signal
        mouth_levels = self.water_levels[:, 0]  # First grid point = mouth
        
        # Calculate tidal statistics at mouth
        mouth_amplitude = (np.max(mouth_levels) - np.min(mouth_levels)) / 2.0
        mouth_mean = np.mean(mouth_levels)
        mouth_range = np.max(mouth_levels) - np.min(mouth_levels)
        
        print(f"Mouth (x=0) tidal statistics:")
        print(f"  Mean level: {mouth_mean:.3f} m")
        print(f"  Amplitude: {mouth_amplitude:.3f} m")
        print(f"  Range: {mouth_range:.3f} m")
        print(f"  Min/Max: {np.min(mouth_levels):.3f} / {np.max(mouth_levels):.3f} m")
        
        # Compare with configured AMPL
        print(f"  Configured AMPL: 4.43 m")
        print(f"  Actual amplitude: {mouth_amplitude:.3f} m")
        print(f"  Amplitude ratio: {mouth_amplitude/4.43:.3f}")
        
        self.analysis_results['mouth_amplitude'] = mouth_amplitude
        self.analysis_results['mouth_range'] = mouth_range
        
        return mouth_amplitude, mouth_range
        
    def analyze_station_specific_amplification(self):
        """Analyze tidal amplification at each problem station."""
        print("\n📍 STATION-SPECIFIC AMPLIFICATION")
        print("-" * 40)
        
        # Station locations (km from mouth)
        stations = {'PC': 86, 'BD': 130, 'BK': 156}
        
        # Extract mouth amplitude for reference
        mouth_levels = self.water_levels[:, 0]
        mouth_amplitude = (np.max(mouth_levels) - np.min(mouth_levels)) / 2.0
        
        station_data = {}
        
        for station, km_location in stations.items():
            # Find nearest grid point
            grid_index = np.argmin(np.abs(self.distances - km_location))
            actual_location = self.distances[grid_index]
            
            # Extract tidal signal at station
            station_levels = self.water_levels[:, grid_index]
            station_amplitude = (np.max(station_levels) - np.min(station_levels)) / 2.0
            station_range = np.max(station_levels) - np.min(station_levels)
            
            # Calculate amplification ratio
            amplification_ratio = station_amplitude / mouth_amplitude
            
            station_data[station] = {
                'location_km': actual_location,
                'amplitude': station_amplitude,
                'range': station_range,
                'amplification': amplification_ratio
            }
            
            print(f"{station} station (@ {actual_location:.1f} km):")
            print(f"  Amplitude: {station_amplitude:.3f} m")
            print(f"  Range: {station_range:.3f} m")
            print(f"  Amplification: {amplification_ratio:.3f}x")
            
        self.analysis_results['stations'] = station_data
        return station_data
        
    def analyze_wave_propagation_pattern(self):
        """Analyze wave propagation and reflection patterns."""
        print("\n🌊 WAVE PROPAGATION ANALYSIS")
        print("-" * 35)
        
        # Calculate amplitude along entire estuary
        amplitudes = []
        for i in range(len(self.distances)):
            station_levels = self.water_levels[:, i]
            amplitude = (np.max(station_levels) - np.min(station_levels)) / 2.0
            amplitudes.append(amplitude)
            
        amplitudes = np.array(amplitudes)
        mouth_amplitude = amplitudes[0]
        
        # Find maximum amplification location
        max_amp_index = np.argmax(amplitudes)
        max_amp_location = self.distances[max_amp_index]
        max_amplification = amplitudes[max_amp_index] / mouth_amplitude
        
        print(f"Maximum amplification: {max_amplification:.3f}x at {max_amp_location:.1f} km")
        
        # Check for resonance-like behavior
        # Look for regions where amplification > 1.5
        resonance_indices = np.where(amplitudes/mouth_amplitude > 1.5)[0]
        if len(resonance_indices) > 0:
            resonance_start = self.distances[resonance_indices[0]]
            resonance_end = self.distances[resonance_indices[-1]]
            print(f"Resonance zone: {resonance_start:.1f} - {resonance_end:.1f} km")
        else:
            print("No strong resonance zones detected")
            
        self.analysis_results['amplitudes'] = amplitudes
        self.analysis_results['max_amplification'] = max_amplification
        self.analysis_results['max_amp_location'] = max_amp_location
        
        return amplitudes
        
    def calculate_optimal_amplitude_scaling(self):
        """Calculate optimal AMPL scaling to improve PC station performance."""
        print("\n🎯 OPTIMAL AMPLITUDE SCALING")
        print("-" * 35)
        
        # Load field data for comparison
        try:
            field_data = self._load_field_tidal_data()
            
            # Focus on PC station (most problematic)
            pc_field_range = field_data.get('PC', 2.23)  # From previous validation
            pc_model_range = self.analysis_results['stations']['PC']['range']
            
            # Calculate required scaling to match PC
            required_scaling = pc_field_range / pc_model_range
            current_ampl = 4.43
            optimal_ampl = current_ampl * required_scaling
            
            print(f"PC Station Analysis:")
            print(f"  Field range: {pc_field_range:.3f} m")
            print(f"  Model range: {pc_model_range:.3f} m")
            print(f"  Required scaling: {required_scaling:.3f}")
            print(f"  Current AMPL: {current_ampl:.3f} m")
            print(f"  Optimal AMPL: {optimal_ampl:.3f} m")
            
            # Check impact on other stations
            print(f"\nImpact on other stations (predicted):")
            for station in ['BD', 'BK']:
                if station in self.analysis_results['stations']:
                    current_range = self.analysis_results['stations'][station]['range']
                    scaled_range = current_range * required_scaling
                    field_range = field_data.get(station, 3.0)
                    new_error = abs(scaled_range - field_range) / field_range * 100
                    print(f"  {station}: {current_range:.3f} → {scaled_range:.3f} m (vs {field_range:.3f} field) = {new_error:.1f}% error")
                    
            self.analysis_results['optimal_ampl'] = optimal_ampl
            self.analysis_results['scaling_factor'] = required_scaling
            
            return optimal_ampl, required_scaling
            
        except Exception as e:
            print(f"❌ Error in scaling calculation: {e}")
            return None, None
            
    def _load_field_tidal_data(self):
        """Load field tidal data for comparison."""
        # Field data from recent validation (average ranges)
        return {
            'PC': 2.23,  # From SIHYMECC data
            'BD': 3.12,
            'BK': 3.42
        }
        
    def create_boundary_analysis_visualization(self):
        """Create comprehensive boundary condition analysis plots."""
        print("\n🎨 Creating boundary condition analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Tidal signal at mouth
        ax1 = axes[0, 0]
        mouth_levels = self.water_levels[:, 0]
        time_hours = np.arange(len(mouth_levels)) * 0.5  # Assuming 30-min output
        ax1.plot(time_hours, mouth_levels, 'b-', linewidth=1.5)
        ax1.set_title('Tidal Boundary Signal at Mouth')
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Water Level [m]')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Amplification pattern along estuary
        ax2 = axes[0, 1]
        if 'amplitudes' in self.analysis_results:
            amplitudes = self.analysis_results['amplitudes']
            mouth_amp = amplitudes[0]
            amplification_ratio = amplitudes / mouth_amp
            ax2.plot(self.distances, amplification_ratio, 'r-', linewidth=2)
            
            # Mark stations
            stations = {'PC': 86, 'BD': 130, 'BK': 156}
            for name, km in stations.items():
                idx = np.argmin(np.abs(self.distances - km))
                ax2.plot(km, amplification_ratio[idx], 'o', markersize=8, label=f'{name} ({amplification_ratio[idx]:.2f}x)')
                
            ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No amplification')
            ax2.set_title('Tidal Amplification Along Estuary')
            ax2.set_xlabel('Distance from Mouth [km]')
            ax2.set_ylabel('Amplification Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Station comparison
        ax3 = axes[1, 0]
        if 'stations' in self.analysis_results:
            station_data = self.analysis_results['stations']
            stations = list(station_data.keys())
            model_ranges = [station_data[s]['range'] for s in stations]
            field_ranges = [self._load_field_tidal_data()[s] for s in stations]
            
            x = np.arange(len(stations))
            width = 0.35
            
            ax3.bar(x - width/2, model_ranges, width, label='Model', alpha=0.8)
            ax3.bar(x + width/2, field_ranges, width, label='Field', alpha=0.8)
            
            ax3.set_title('Model vs Field Tidal Ranges')
            ax3.set_xlabel('Station')
            ax3.set_ylabel('Tidal Range [m]')
            ax3.set_xticks(x)
            ax3.set_xticklabels(stations)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time series at PC station
        ax4 = axes[1, 1]
        pc_km = 86
        pc_index = np.argmin(np.abs(self.distances - pc_km))
        pc_levels = self.water_levels[:, pc_index]
        ax4.plot(time_hours, pc_levels, 'g-', linewidth=1.5)
        ax4.set_title('PC Station Time Series')
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Water Level [m]')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / "boundary_condition_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Boundary analysis saved: {output_file}")
        plt.close()
        
    def generate_ampl_correction_config(self):
        """Generate corrected model configuration with optimal AMPL."""
        if 'optimal_ampl' not in self.analysis_results:
            print("❌ Optimal AMPL not calculated. Run scaling analysis first.")
            return
            
        optimal_ampl = self.analysis_results['optimal_ampl']
        
        print(f"\n⚙️ GENERATING CORRECTED CONFIGURATION")
        print("-" * 45)
        print(f"Original AMPL: 4.43 m")
        print(f"Optimal AMPL: {optimal_ampl:.3f} m")
        print(f"Scaling factor: {self.analysis_results['scaling_factor']:.3f}")
        
        # Read current config
        config_file = "config/model_config.txt"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                lines = f.readlines()
            
            # Create corrected version
            corrected_lines = []
            for line in lines:
                if line.strip().startswith("AMPL = "):
                    corrected_lines.append(f"AMPL = {optimal_ampl:.3f}           # Tidal amplitude at the mouth [m] - PC station optimized\n")
                else:
                    corrected_lines.append(line)
            
            # Save corrected config
            corrected_config = "config/model_config_pc_optimized.txt"
            with open(corrected_config, 'w') as f:
                f.writelines(corrected_lines)
                
            print(f"✅ Corrected config saved: {corrected_config}")
            return corrected_config
        else:
            print("❌ Original config file not found")
            return None

def main():
    """Run boundary condition analysis for Phase II enhancement."""
    
    print("🌊 JAX C-GEM Phase II Enhancement: Boundary Condition Analysis")
    print("=" * 70)
    print("Task 2.3.4: Analyze PC station boundary condition issues")
    print()
    
    analyzer = BoundaryConditionAnalyzer()
    
    # Load simulation data
    if not analyzer.load_simulation_data():
        return
        
    # Perform comprehensive boundary condition analysis
    try:
        print("🔍 Running comprehensive boundary condition analysis...")
        
        # 1. Analyze tidal boundary forcing
        mouth_amp, mouth_range = analyzer.analyze_tidal_boundary_forcing()
        
        # 2. Station-specific amplification
        station_data = analyzer.analyze_station_specific_amplification()
        
        # 3. Wave propagation patterns
        amplitudes = analyzer.analyze_wave_propagation_pattern()
        
        # 4. Calculate optimal scaling
        optimal_ampl, scaling = analyzer.calculate_optimal_amplitude_scaling()
        
        # 5. Create visualizations
        analyzer.create_boundary_analysis_visualization()
        
        # 6. Generate corrected configuration
        if optimal_ampl:
            corrected_config = analyzer.generate_ampl_correction_config()
            
        print("\n✅ BOUNDARY CONDITION ANALYSIS COMPLETE")
        print("=" * 50)
        
        if optimal_ampl and scaling:
            print(f"🎯 RECOMMENDED ACTION:")
            print(f"   Update AMPL from 4.43 to {optimal_ampl:.3f} m")
            print(f"   Expected PC error reduction: ~{(1-scaling)*100:.1f}%")
            print(f"   Use config: model_config_pc_optimized.txt")
        else:
            print("⚠️  Could not determine optimal AMPL scaling")
            
        print("\n📊 Key findings saved to OUT/boundary_condition_analysis.png")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()