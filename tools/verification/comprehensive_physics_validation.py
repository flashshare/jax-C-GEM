#!/usr/bin/env python3
"""
Comprehensive Physics Validation for JAX C-GEM
==============================================

Creates comprehensive validation plots showing:
1. Tidal amplitude longitudinal profiles (mean/min/max over tidal cycles)
2. Salinity longitudinal profiles (mean/min/max over tidal cycles) 
3. Dispersion coefficient longitudinal profiles
4. Comparison with expected C-GEM behavior

This addresses the core request to validate physics and understand why
salinity gradient remains inverted despite implementing C-GEM transport scheme.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PhysicsValidator:
    """Comprehensive physics validation with tidal cycle analysis."""
    
    def __init__(self, results_dir: str = "OUT", field_data_dir: str = "INPUT/Calibration"):
        self.results_dir = Path(results_dir)
        self.field_data_dir = Path(field_data_dir)
        self.distance_km = None
        self.setup_geometry()
        
    def setup_geometry(self):
        """Setup the longitudinal distance array."""
        try:
            # Load geometry
            geom_file = Path("INPUT/Geometry/Geometry.csv")
            if geom_file.exists():
                geom = pd.read_csv(geom_file)
                # Distance from mouth in km (convert from m)
                self.distance_km = geom['DIST'].values / 1000.0
                print(f"✅ Loaded geometry: {len(self.distance_km)} points, 0-{self.distance_km[-1]:.1f} km")
            else:
                # Fallback: create uniform grid
                self.distance_km = np.linspace(0, 160, 102)  # Default Mekong Delta length
                print(f"⚠️  Using default geometry: 0-160 km")
        except Exception as e:
            self.distance_km = np.linspace(0, 160, 102)
            print(f"⚠️  Geometry error: {e}, using default 0-160 km")

    def load_jax_results(self) -> Dict:
        """Load JAX C-GEM results from CSV files."""
        try:
            # Load hydrodynamics
            H = pd.read_csv(self.results_dir / "Hydrodynamics" / "H.csv")
            U = pd.read_csv(self.results_dir / "Hydrodynamics" / "U.csv")
            
            # Load key water quality variables
            S = pd.read_csv(self.results_dir / "Reaction" / "S.csv") if (self.results_dir / "Reaction" / "S.csv").exists() else None
            O2 = pd.read_csv(self.results_dir / "Reaction" / "O2.csv") if (self.results_dir / "Reaction" / "O2.csv").exists() else None
            
            # Extract data arrays (skip time column)
            results = {
                'time': H.iloc[:, 0].values,
                'water_level': H.iloc[:, 1:].values,
                'velocity': U.iloc[:, 1:].values,
                'salinity': S.iloc[:, 1:].values if S is not None else None,
                'oxygen': O2.iloc[:, 1:].values if O2 is not None else None,
            }
            
            print(f"✅ Loaded JAX results: {results['time'].shape[0]} timesteps, {results['water_level'].shape[1]} grid points")
            return results
            
        except Exception as e:
            print(f"❌ Error loading JAX results: {e}")
            return None

    def compute_tidal_statistics(self, data: np.ndarray, tidal_period_hours: float = 12.42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean, min, max over tidal cycles."""
        if data is None:
            return None, None, None
            
        # Assume timestep is 3 minutes (180 seconds) based on C-GEM setup
        timestep_hours = 3.0 / 60.0  # 0.05 hours
        steps_per_tidal_cycle = int(tidal_period_hours / timestep_hours)
        
        # Reshape to analyze complete tidal cycles
        n_timesteps, n_points = data.shape
        n_complete_cycles = n_timesteps // steps_per_tidal_cycle
        
        if n_complete_cycles < 1:
            # Not enough data for full tidal cycles - use overall statistics
            return np.mean(data, axis=0), np.min(data, axis=0), np.max(data, axis=0)
        
        # Trim to complete cycles
        trimmed_data = data[:n_complete_cycles * steps_per_tidal_cycle].reshape(
            n_complete_cycles, steps_per_tidal_cycle, n_points
        )
        
        # Compute statistics over tidal cycles
        mean_profile = np.mean(trimmed_data, axis=(0, 1))  # Mean over cycles and time within cycles
        min_profile = np.min(trimmed_data, axis=(0, 1))    # Min over cycles and time within cycles  
        max_profile = np.max(trimmed_data, axis=(0, 1))    # Max over cycles and time within cycles
        
        return mean_profile, min_profile, max_profile

    def load_field_data(self) -> Dict:
        """Load field observation data for comparison."""
        field_data = {}
        
        try:
            # Load CEM longitudinal profile data
            cem_file = self.field_data_dir / "CEM_2017-2018.csv"
            if cem_file.exists():
                cem = pd.read_csv(cem_file)
                # Convert distance from mouth (assume CEM gives km from mouth)
                if 'Distance_km' in cem.columns and 'Salinity_psu' in cem.columns:
                    field_data['cem_distance'] = cem['Distance_km'].values
                    field_data['cem_salinity'] = cem['Salinity_psu'].values
                    print(f"✅ Loaded CEM field data: {len(field_data['cem_salinity'])} points")
                    
        except Exception as e:
            print(f"⚠️  Could not load CEM data: {e}")
            
        try:
            # Load SIHYMECC tidal data  
            sihymecc_file = self.field_data_dir / "SIHYMECC_Tidal-range2017-2018.csv"
            if sihymecc_file.exists():
                sihymecc = pd.read_csv(sihymecc_file)
                # Extract tidal ranges at known stations
                if 'Station' in sihymecc.columns and 'Tidal_Range_m' in sihymecc.columns:
                    field_data['tidal_stations'] = sihymecc['Station'].values
                    field_data['tidal_ranges'] = sihymecc['Tidal_Range_m'].values
                    print(f"✅ Loaded SIHYMECC tidal data: {len(field_data['tidal_ranges'])} stations")
                    
        except Exception as e:
            print(f"⚠️  Could not load SIHYMECC data: {e}")
            
        return field_data

    def create_comprehensive_validation_plot(self, output_dir: str = "OUT/Validation"):
        """Create comprehensive physics validation plots."""
        
        # Load data
        jax_results = self.load_jax_results()
        field_data = self.load_field_data()
        
        if jax_results is None:
            print("❌ Cannot create validation plots without JAX results")
            return
            
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up the comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. TIDAL AMPLITUDE VALIDATION
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_tidal_amplitude_validation(ax1, jax_results, field_data)
        
        # 2. SALINITY PROFILE VALIDATION  
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_salinity_validation(ax2, jax_results, field_data)
        
        # 3. VELOCITY PROFILES
        ax3 = fig.add_subplot(gs[2, 0])
        self.plot_velocity_profiles(ax3, jax_results)
        
        # 4. DISPERSION ANALYSIS
        ax4 = fig.add_subplot(gs[2, 1])
        self.plot_dispersion_analysis(ax4, jax_results)
        
        # 5. TRANSPORT DIAGNOSTICS
        ax5 = fig.add_subplot(gs[3, 0])
        self.plot_transport_diagnostics(ax5, jax_results)
        
        # 6. MASS BALANCE CHECK
        ax6 = fig.add_subplot(gs[3, 1])
        self.plot_mass_balance(ax6, jax_results)
        
        # Add overall title
        fig.suptitle('JAX C-GEM Comprehensive Physics Validation\n'
                    f'🔬 Tidal Dynamics • 🌊 Salinity Transport • ⚗️ Mass Conservation', 
                    fontsize=16, fontweight='bold')
                    
        # Save the comprehensive plot
        output_file = output_path / "comprehensive_physics_validation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive validation plot: {output_file}")
        
        # Show the plot
        plt.show()

    def plot_tidal_amplitude_validation(self, ax, jax_results: Dict, field_data: Dict):
        """Plot tidal amplitude with mean/min/max envelopes."""
        
        # Compute tidal statistics
        h_mean, h_min, h_max = self.compute_tidal_statistics(jax_results['water_level'])
        
        # Tidal amplitude = (max - min) / 2
        tidal_amplitude = (h_max - h_min) / 2.0
        
        # Plot tidal amplitude profile
        ax.plot(self.distance_km, tidal_amplitude, 'b-', linewidth=3, label='JAX C-GEM Tidal Amplitude')
        ax.fill_between(self.distance_km, h_min, h_max, alpha=0.3, color='blue', 
                       label='Water Level Range (Min-Max)')
        
        # Add field data if available
        if 'tidal_ranges' in field_data and 'tidal_stations' in field_data:
            # Map station names to approximate distances (customize for your estuary)
            station_distances = {'PC': 86, 'BD': 130, 'BK': 156}  # Example distances in km
            
            for station, tidal_range in zip(field_data['tidal_stations'], field_data['tidal_ranges']):
                if station in station_distances:
                    dist = station_distances[station]
                    ax.scatter(dist, tidal_range, s=100, c='red', marker='o', 
                             label=f'SIHYMECC {station}' if station == field_data['tidal_stations'][0] else "")
        
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_ylabel('Tidal Amplitude (m)')
        ax.set_title('🌊 Tidal Amplitude Longitudinal Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add diagnostic text
        max_amplitude = np.max(tidal_amplitude)
        min_amplitude = np.min(tidal_amplitude)
        ax.text(0.02, 0.98, f'Max Amplitude: {max_amplitude:.2f} m\nMin Amplitude: {min_amplitude:.2f} m', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_salinity_validation(self, ax, jax_results: Dict, field_data: Dict):
        """Plot salinity profile with critical gradient analysis."""
        
        if jax_results['salinity'] is None:
            ax.text(0.5, 0.5, 'No Salinity Data Available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            return
            
        # Compute salinity statistics
        s_mean, s_min, s_max = self.compute_tidal_statistics(jax_results['salinity'])
        
        # Plot salinity profiles
        ax.plot(self.distance_km, s_mean, 'g-', linewidth=3, label='JAX C-GEM Mean Salinity')
        ax.fill_between(self.distance_km, s_min, s_max, alpha=0.3, color='green', 
                       label='Salinity Range (Min-Max)')
        
        # Add field data if available
        if 'cem_distance' in field_data and 'cem_salinity' in field_data:
            ax.scatter(field_data['cem_distance'], field_data['cem_salinity'], 
                      s=50, c='red', marker='s', label='CEM Field Data', alpha=0.7)
        
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_ylabel('Salinity (PSU)')
        ax.set_title('🧂 Salinity Longitudinal Profile - GRADIENT ANALYSIS')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # CRITICAL GRADIENT ANALYSIS
        mouth_salinity = s_mean[0]
        head_salinity = s_mean[-1]
        gradient_direction = "CORRECT ✅" if mouth_salinity > head_salinity else "INVERTED ❌"
        
        # Expected vs actual
        expected_mouth = "~25-30 PSU (Ocean)"
        expected_head = "~0.01-0.1 PSU (River)"
        actual_mouth = f"{mouth_salinity:.2f} PSU"
        actual_head = f"{head_salinity:.2f} PSU"
        
        ax.text(0.02, 0.98, f'GRADIENT: {gradient_direction}\n'
                           f'Mouth: {actual_mouth} (expect {expected_mouth})\n'
                           f'Head: {actual_head} (expect {expected_head})', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', 
                         facecolor='lightcoral' if 'INVERTED' in gradient_direction else 'lightgreen', 
                         alpha=0.8))

    def plot_velocity_profiles(self, ax, jax_results: Dict):
        """Plot velocity statistics."""
        
        # Compute velocity statistics
        u_mean, u_min, u_max = self.compute_tidal_statistics(jax_results['velocity'])
        
        # Plot velocity profiles
        ax.plot(self.distance_km, u_mean, 'purple', linewidth=2, label='Mean Velocity')
        ax.fill_between(self.distance_km, u_min, u_max, alpha=0.3, color='purple',
                       label='Velocity Range')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('🌊 Velocity Profiles')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_dispersion_analysis(self, ax, jax_results: Dict):
        """Plot dispersion coefficient analysis."""
        
        # Estimate dispersion from salinity gradients (van der Burgh approximation)
        if jax_results['salinity'] is not None:
            s_mean, _, _ = self.compute_tidal_statistics(jax_results['salinity'])
            u_mean, _, _ = self.compute_tidal_statistics(jax_results['velocity'])
            
            # Simple dispersion estimate: D ≈ u * L_scale
            # where L_scale is characteristic mixing length (~1 km for estuaries)
            L_scale = 1000.0  # 1 km in meters
            D_est = np.abs(u_mean) * L_scale
            
            ax.plot(self.distance_km, D_est, 'orange', linewidth=2, label='Estimated Dispersion')
            ax.set_ylabel('Dispersion Coeff (m²/s)')
        else:
            # Plot velocity magnitude as proxy
            u_mean, _, _ = self.compute_tidal_statistics(jax_results['velocity'])
            ax.plot(self.distance_km, np.abs(u_mean), 'orange', linewidth=2, label='|Velocity|')
            ax.set_ylabel('|Velocity| (m/s)')
            
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_title('⚗️ Transport Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_transport_diagnostics(self, ax, jax_results: Dict):
        """Plot transport diagnostics."""
        
        if jax_results['salinity'] is not None:
            # Compute salinity gradients
            s_mean, _, _ = self.compute_tidal_statistics(jax_results['salinity'])
            dx = (self.distance_km[-1] - self.distance_km[0]) * 1000 / len(self.distance_km)  # Convert to meters
            ds_dx = np.gradient(s_mean, dx)
            
            ax.plot(self.distance_km, ds_dx * 1000, 'brown', linewidth=2, label='dS/dx (×1000 m⁻¹)')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Salinity Gradient (×1000 m⁻¹)')
        else:
            # Plot water level gradients
            h_mean, _, _ = self.compute_tidal_statistics(jax_results['water_level'])
            dx = (self.distance_km[-1] - self.distance_km[0]) * 1000 / len(self.distance_km)
            dh_dx = np.gradient(h_mean, dx)
            
            ax.plot(self.distance_km, dh_dx * 1000, 'brown', linewidth=2, label='dH/dx (×1000 m⁻¹)')
            ax.set_ylabel('Water Level Gradient (×1000 m⁻¹)')
            
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_title('📊 Transport Gradients')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_mass_balance(self, ax, jax_results: Dict):
        """Plot mass balance analysis."""
        
        if jax_results['salinity'] is not None:
            # Compute total salt mass over time
            salinity_data = jax_results['salinity']
            
            # Estimate cell volumes (simple approximation)
            dx = (self.distance_km[-1] - self.distance_km[0]) * 1000 / len(self.distance_km)  # meters
            typical_depth = 10.0  # meters
            typical_width = 1000.0  # meters  
            cell_volume = dx * typical_depth * typical_width
            
            # Total salt mass over time
            total_salt_mass = np.sum(salinity_data * cell_volume, axis=1)  # Sum over space
            
            # Plot mass conservation
            time_hours = jax_results['time'] / 3600.0  # Convert to hours
            ax.plot(time_hours, total_salt_mass / total_salt_mass[0], 'red', linewidth=2, 
                   label='Total Salt Mass (normalized)')
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Conservation')
            
            # Calculate mass change
            mass_change_percent = (total_salt_mass[-1] - total_salt_mass[0]) / total_salt_mass[0] * 100
            
            ax.text(0.02, 0.98, f'Mass Change: {mass_change_percent:.2f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', 
                            facecolor='lightcoral' if abs(mass_change_percent) > 5 else 'lightgreen',
                            alpha=0.8))
        else:
            # Plot water mass balance
            h_data = jax_results['water_level']
            total_water_volume = np.sum(h_data, axis=1)
            time_hours = jax_results['time'] / 3600.0
            
            ax.plot(time_hours, total_water_volume / np.mean(total_water_volume), 'blue', linewidth=2,
                   label='Total Water Volume (normalized)')
            
        ax.set_xlabel('Time (hours)')  
        ax.set_ylabel('Normalized Mass/Volume')
        ax.set_title('⚖️ Mass Balance Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Comprehensive Physics Validation for JAX C-GEM')
    parser.add_argument('--results-dir', default='OUT', help='Results directory')
    parser.add_argument('--field-data-dir', default='INPUT/Calibration', help='Field data directory')
    parser.add_argument('--output-dir', default='OUT/Validation', help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("🔬 JAX C-GEM Comprehensive Physics Validation")
    print("=" * 50)
    
    # Initialize validator
    validator = PhysicsValidator(args.results_dir, args.field_data_dir)
    
    # Create comprehensive validation plot
    validator.create_comprehensive_validation_plot(args.output_dir)
    
    print("\n✅ Physics validation complete!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()