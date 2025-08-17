#!/usr/bin/env python3
"""
Momentum Balance Analyzer
=========================

Task 2.3.2: Phase II Enhancement - Analyze momentum balance components
to identify terms causing excessive tidal amplification in JAX C-GEM.

The momentum equation in 1D shallow water is:
∂u/∂t + u∂u/∂x + g∂h/∂x + g*u*|u|/(C²*h) = 0

Where:
- ∂u/∂t: Local acceleration (time rate of change of velocity)
- u∂u/∂x: Convective acceleration (advective term)
- g∂h/∂x: Pressure gradient force (hydrostatic pressure)
- g*u*|u|/(C²*h): Friction force (quadratic bottom friction)

This analysis will decompose the momentum balance to identify which
terms are causing the excessive 2x tidal over-prediction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

class MomentumBalanceAnalyzer:
    """Analyze momentum balance components in tidal simulations."""
    
    def __init__(self):
        """Initialize the momentum balance analyzer."""
        self.results = None
        self.geometry = None
        self.stations = {
            'PC': 86,    # 86 km from mouth (corresponds to index ~43)
            'BD': 130,   # 130 km from mouth (corresponds to index ~65) 
            'BK': 156    # 156 km from mouth (corresponds to index ~78)
        }
        
    def load_simulation_results(self, results_file="OUT/complete_simulation_results.npz"):
        """Load simulation results for momentum analysis."""
        
        print(f"🌊 Loading simulation results from {results_file}")
        print("-" * 40)
        
        try:
            data = np.load(results_file, allow_pickle=True)
            
            # Extract key variables - adapt to actual C-GEM output format
            self.results = {
                'water_level': data['H'],      # h(x,t) [m] - water level
                'velocity': data['U'],         # u(x,t) [m/s] - velocity
                'time': data['time']           # time [hours]
            }
            
            # Generate distance array based on grid (DELXI = 2000m, starting from 1 km)
            nt, nx = self.results['water_level'].shape
            self.results['distance'] = np.linspace(1, 1 + (nx-1)*2, nx)  # [km] from 1 to 201 km
            
            print(f"✅ Loaded simulation data:")
            print(f"   Time steps: {nt} (range: {self.results['time'][0]:.1f} - {self.results['time'][-1]:.1f} hours)")
            print(f"   Grid points: {nx} (range: {self.results['distance'][0]:.1f} - {self.results['distance'][-1]:.1f} km)")
            print(f"   Water level range: {np.min(self.results['water_level']):.2f} - {np.max(self.results['water_level']):.2f} m")
            print(f"   Velocity range: {np.min(self.results['velocity']):.2f} - {np.max(self.results['velocity']):.2f} m/s")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Results file not found: {results_file}")
            return False
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def load_geometry_data(self):
        """Load channel geometry for momentum analysis."""
        
        print(f"\n🏞️ Loading channel geometry")
        print("-" * 30)
        
        try:
            # Load from the same file as depth_dependent_friction.py
            geometry_file = "INPUT/Geometry/Geometry.csv"
            
            # Simple CSV parsing (assuming headers: DISTANCE,WIDTH,PROF)
            data = []
            with open(geometry_file, 'r') as f:
                lines = f.readlines()
                
            # Find header and data
            header_line = None
            for i, line in enumerate(lines):
                if 'DISTANCE' in line.upper() and 'WIDTH' in line.upper():
                    header_line = i
                    break
            
            if header_line is None:
                raise ValueError("Could not find geometry headers")
            
            # Parse data
            for line in lines[header_line + 1:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        try:
                            distance = float(parts[0]) / 1000  # Convert to km
                            width = float(parts[1])
                            depth = float(parts[2])
                            data.append([distance, width, depth])
                        except ValueError:
                            continue
            
            if not data:
                raise ValueError("No valid geometry data found")
            
            data = np.array(data)
            self.geometry = {
                'distance': data[:, 0],  # [km]
                'width': data[:, 1],     # [m]
                'depth': data[:, 2]      # [m]
            }
            
            print(f"✅ Loaded geometry data:")
            print(f"   Points: {len(self.geometry['distance'])}")
            print(f"   Distance range: {self.geometry['distance'].min():.1f} - {self.geometry['distance'].max():.1f} km")
            print(f"   Width range: {self.geometry['width'].min():.0f} - {self.geometry['width'].max():.0f} m")
            print(f"   Depth range: {self.geometry['depth'].min():.1f} - {self.geometry['depth'].max():.1f} m")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading geometry: {e}")
            return False
    
    def calculate_momentum_terms(self):
        """Calculate individual momentum balance terms."""
        
        if self.results is None:
            print("❌ No simulation results loaded")
            return None
        
        print(f"\n⚡ Calculating Momentum Balance Terms")
        print("-" * 40)
        
        # Extract variables
        h = self.results['water_level']    # [m]
        u = self.results['velocity']       # [m/s]
        t = self.results['time']           # [seconds] - corrected from hours 
        x = self.results['distance']       # [km]
        
        nt, nx = h.shape
        dt = t[1] - t[0]                   # Already in seconds (no conversion needed)
        dx = (x[1] - x[0]) * 1000          # Convert km to meters
        g = 9.81                           # [m/s²]
        
        print(f"   Grid spacing: dx = {dx:.0f} m, dt = {dt:.0f} s")
        print(f"   Time step info: dt = {dt:.0f} s = {dt/60:.1f} min (should be 180-600s)")
        print(f"   Tidal period = 12.42 h = {12.42*3600:.0f} s, dt/T_tidal = {dt/(12.42*3600):.4f}")
        
        # Verify time step is appropriate for tidal dynamics
        if dt > 3600:
            print(f"   ⚠️  WARNING: dt = {dt:.0f}s > 1 hour may be too large for accurate tidal dynamics")
        elif dt > 600:
            print(f"   ⚠️  CAUTION: dt = {dt:.0f}s is large - check numerical accuracy")
        else:
            print(f"   ✅ Time step dt = {dt:.0f}s is appropriate for tidal dynamics")
        
        # Initialize momentum terms
        momentum_terms = {
            'local_acceleration': np.zeros_like(h),     # ∂u/∂t
            'convective_acceleration': np.zeros_like(h), # u∂u/∂x
            'pressure_gradient': np.zeros_like(h),      # g∂h/∂x
            'friction_force': np.zeros_like(h)          # g*u*|u|/(C²*h)
        }
        
        # Calculate terms
        print("   Calculating local acceleration (∂u/∂t)...")
        for i in range(1, nt-1):
            momentum_terms['local_acceleration'][i, :] = (u[i+1, :] - u[i-1, :]) / (2 * dt)
        
        print("   Calculating convective acceleration (u∂u/∂x)...")
        for j in range(1, nx-1):
            momentum_terms['convective_acceleration'][:, j] = u[:, j] * (u[:, j+1] - u[:, j-1]) / (2 * dx)
        
        print("   Calculating pressure gradient (g∂h/∂x)...")
        for j in range(1, nx-1):
            momentum_terms['pressure_gradient'][:, j] = g * (h[:, j+1] - h[:, j-1]) / (2 * dx)
        
        print("   Calculating friction force...")
        # Assume typical Chezy coefficients from configuration
        chezy1, chezy2 = 30.0, 40.0  # [m^0.5/s]
        index_transition = nx // 3    # Approximate transition point
        
        # Create spatially-varying Chezy coefficient
        chezy = np.full(nx, chezy1)
        chezy[index_transition:] = chezy2
        
        for j in range(nx):
            momentum_terms['friction_force'][:, j] = g * u[:, j] * np.abs(u[:, j]) / (chezy[j]**2 * h[:, j])
        
        print("✅ Momentum terms calculated successfully")
        
        return momentum_terms
    
    def analyze_momentum_balance(self, momentum_terms):
        """Analyze momentum balance at key stations."""
        
        if momentum_terms is None or self.results is None:
            return None
        
        print(f"\n📊 Analyzing Momentum Balance at Key Stations")
        print("-" * 50)
        
        # Find station indices in simulation grid
        x = self.results['distance']  # [km]
        station_indices = {}
        
        for station_name, station_km in self.stations.items():
            # Find closest grid point
            idx = np.argmin(np.abs(x - station_km))
            station_indices[station_name] = idx
            print(f"   {station_name}: {station_km} km → grid index {idx} ({x[idx]:.1f} km)")
        
        print()
        
        # Calculate RMS values for each term at each station
        analysis_results = {}
        
        for station_name, idx in station_indices.items():
            print(f"📍 Station {station_name} ({self.stations[station_name]} km):")
            print("-" * 30)
            
            station_results = {}
            
            for term_name, term_data in momentum_terms.items():
                # Use data excluding warmup period (first 100 time steps)
                warmup_steps = min(100, len(term_data) // 4)
                term_series = term_data[warmup_steps:, idx]
                
                # Calculate statistics
                rms_value = np.sqrt(np.mean(term_series**2))
                mean_value = np.mean(term_series)
                max_abs_value = np.max(np.abs(term_series))
                
                station_results[term_name] = {
                    'rms': rms_value,
                    'mean': mean_value,
                    'max_abs': max_abs_value,
                    'series': term_series
                }
                
                print(f"   {term_name:25}: RMS = {rms_value:8.4f} m/s², Mean = {mean_value:+8.4f} m/s², Max = {max_abs_value:8.4f} m/s²")
            
            # Calculate momentum balance residual
            residual = (momentum_terms['local_acceleration'][warmup_steps:, idx] + 
                       momentum_terms['convective_acceleration'][warmup_steps:, idx] + 
                       momentum_terms['pressure_gradient'][warmup_steps:, idx] + 
                       momentum_terms['friction_force'][warmup_steps:, idx])
            
            residual_rms = np.sqrt(np.mean(residual**2))
            station_results['residual'] = {'rms': residual_rms, 'series': residual}
            
            print(f"   {'momentum_residual':25}: RMS = {residual_rms:8.4f} m/s² (should be ~0)")
            print()
            
            analysis_results[station_name] = station_results
        
        return analysis_results
    
    def create_momentum_visualization(self, momentum_terms, analysis_results):
        """Create momentum balance visualization."""
        
        if self.results is None or momentum_terms is None or analysis_results is None:
            print("❌ Missing required data for visualization")
            return None
        
        print(f"📊 Creating Momentum Balance Visualization")
        print("-" * 40)
        
        # Create comprehensive momentum balance figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('JAX C-GEM Momentum Balance Analysis\nTask 2.3.2: Phase II Enhancement', fontsize=16, fontweight='bold')
        
        # Colors for different terms
        colors = {
            'local_acceleration': 'blue',
            'convective_acceleration': 'red', 
            'pressure_gradient': 'green',
            'friction_force': 'orange'
        }
        
        # Panel 1: RMS momentum terms by station
        ax1 = axes[0, 0]
        stations = list(analysis_results.keys())
        terms = ['local_acceleration', 'convective_acceleration', 'pressure_gradient', 'friction_force']
        term_labels = ['Local Accel', 'Convective', 'Pressure Grad', 'Friction']
        
        x_pos = np.arange(len(stations))
        width = 0.2
        
        for i, (term, label) in enumerate(zip(terms, term_labels)):
            values = [analysis_results[station][term]['rms'] for station in stations]
            ax1.bar(x_pos + i * width, values, width, label=label, color=colors[term], alpha=0.7)
        
        ax1.set_xlabel('Station')
        ax1.set_ylabel('RMS Acceleration [m/s²]')
        ax1.set_title('Momentum Terms by Station')
        ax1.set_xticks(x_pos + 1.5 * width)
        ax1.set_xticklabels(stations)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Dominant term analysis
        ax2 = axes[0, 1]
        
        for station in stations:
            term_values = [analysis_results[station][term]['rms'] for term in terms]
            total_magnitude = sum(term_values)
            percentages = [100 * val / total_magnitude for val in term_values]
            
            station_idx = stations.index(station)
            bottom = 0
            for i, (term, percentage) in enumerate(zip(terms, percentages)):
                ax2.bar(station_idx, percentage, bottom=bottom, color=colors[term], alpha=0.7)
                if percentage > 10:  # Only label if > 10%
                    ax2.text(station_idx, bottom + percentage/2, f'{percentage:.0f}%', 
                            ha='center', va='center', fontweight='bold')
                bottom += percentage
        
        ax2.set_xlabel('Station')
        ax2.set_ylabel('Momentum Term Contribution [%]')
        ax2.set_title('Relative Importance of Momentum Terms')
        ax2.set_xticks(range(len(stations)))
        ax2.set_xticklabels(stations)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Time series at most problematic station (highest tidal error)
        ax3 = axes[1, 0]
        
        # Use BK station (highest error from friction analysis)
        problem_station = 'BK'
        if problem_station in analysis_results:
            warmup_hours = 100 * 0.5  # Assuming 30-min time steps
            time_subset = self.results['time'][100:200]  # Show 50 time steps
            
            for term, label in zip(terms, term_labels):
                series = analysis_results[problem_station][term]['series'][:100]  # First 100 steps after warmup
                ax3.plot(time_subset, series, label=label, color=colors[term], linewidth=2)
            
            ax3.set_xlabel('Time [hours]')
            ax3.set_ylabel('Acceleration [m/s²]')
            ax3.set_title(f'Momentum Terms Time Series at {problem_station} Station')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Momentum balance quality
        ax4 = axes[1, 1]
        
        residuals = [analysis_results[station]['residual']['rms'] for station in stations]
        max_terms = [max([analysis_results[station][term]['rms'] for term in terms]) for station in stations]
        balance_quality = [res / max_term * 100 for res, max_term in zip(residuals, max_terms)]
        
        bars = ax4.bar(stations, balance_quality, color='red', alpha=0.7)
        ax4.axhline(y=5, color='orange', linestyle='--', label='Good Balance (5%)')
        ax4.axhline(y=10, color='red', linestyle='--', label='Poor Balance (10%)')
        
        ax4.set_xlabel('Station')
        ax4.set_ylabel('Momentum Balance Error [%]')
        ax4.set_title('Momentum Balance Quality Check')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, value in zip(bars, balance_quality):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{value:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_file = "OUT/momentum_balance_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_file}")
        
        plt.show()
        
        return output_file

def main():
    """Run momentum balance analysis for Phase II enhancement."""
    
    print("⚡ JAX C-GEM Phase II Enhancement: Momentum Balance Analysis")
    print("=" * 65)
    print("Task 2.3.2: Identify terms causing excessive tidal amplification")
    print()
    
    analyzer = MomentumBalanceAnalyzer()
    
    # Load simulation results
    if not analyzer.load_simulation_results():
        print("❌ Cannot proceed without simulation results")
        return
    
    # Load geometry data  
    if not analyzer.load_geometry_data():
        print("⚠️  Proceeding without detailed geometry data")
    
    # Calculate momentum terms
    momentum_terms = analyzer.calculate_momentum_terms()
    if momentum_terms is None:
        print("❌ Failed to calculate momentum terms")
        return
    
    # Analyze momentum balance
    analysis_results = analyzer.analyze_momentum_balance(momentum_terms)
    if analysis_results is None:
        print("❌ Failed to analyze momentum balance")
        return
    
    # Create visualization
    analyzer.create_momentum_visualization(momentum_terms, analysis_results)
    
    print("\n🎯 MOMENTUM ANALYSIS SUMMARY:")
    print("=" * 35)
    
    # Identify dominant terms causing problems
    for station in ['PC', 'BD', 'BK']:
        if station in analysis_results:
            terms = ['local_acceleration', 'convective_acceleration', 'pressure_gradient', 'friction_force']
            rms_values = [analysis_results[station][term]['rms'] for term in terms]
            dominant_idx = np.argmax(rms_values)
            dominant_term = terms[dominant_idx]
            dominant_value = rms_values[dominant_idx]
            
            print(f"Station {station}: Dominant term = {dominant_term} (RMS = {dominant_value:.4f} m/s²)")
    
    print()
    print("🔬 EXPECTED INSIGHTS:")
    print("-" * 20)
    print("• If pressure gradient dominates: Tidal boundary conditions may be excessive")
    print("• If convective acceleration dominates: Non-linear advection causing amplification")
    print("• If friction is too weak: Need stronger friction coefficients")
    print("• Large momentum residuals: Numerical issues or missing physics")
    
    print()
    print("🚀 NEXT STEPS based on results:")
    print("-" * 30)
    print("1. Review dominant terms identified above")
    print("2. If pressure gradient dominates → Task 2.3.4 (Boundary Conditions)")
    print("3. If convective terms dominate → Investigate advection schemes")
    print("4. If friction insufficient → Task 2.3.6 (Advanced Friction Models)")

if __name__ == "__main__":
    # Change to project directory
    project_dir = r"c:\Users\nguytruo\Documents\C-GEM\jax-C-GEM"
    os.chdir(project_dir)
    
    main()