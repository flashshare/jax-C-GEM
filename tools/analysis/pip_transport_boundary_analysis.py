#!/usr/bin/env python
"""
PIP Transport Boundary Analysis for Phase VII Task 16

This script provides detailed analysis of PIP transport boundary behavior,
mass flux patterns, and initialization effects to identify the root cause
of the 13.27% mass loss in particulate inorganic phosphorus.

Scientific Methodology:
1. Analyze boundary flux patterns for PIP at upstream/downstream boundaries
2. Investigate initialization effects on PIP mass balance over simulation period
3. Optimize transport boundary conditions for particulate species conservation
4. Validate complete PIP mass balance across transport-biogeochemistry coupling

Author: Nguyen Truong An
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.model_config import SPECIES_NAMES
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import core modules: {e}")
    SPECIES_NAMES = ['PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2']

def load_simulation_data(results_dir="OUT"):
    """Load complete simulation results for boundary analysis."""
    print(f"📂 Loading simulation data from {results_dir}/")
    
    # Try NPZ format first (high performance)
    npz_file = Path(results_dir) / "simulation_results.npz"
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        print("✅ Using NPZ format data")
        return convert_npz_to_analysis_format(data)
    else:
        print("❌ NPZ file not found. Run simulation first.")
        return None

def convert_npz_to_analysis_format(npz_data):
    """Convert NPZ data to analysis format with boundary focus."""
    print("🔄 Converting NPZ data for boundary analysis...")
    
    # Extract core data
    results = {
        'time': np.array(npz_data['time']),
        'x': np.arange(npz_data['PIP'].shape[1]) * 2000,  # 2km grid spacing
        'velocity': np.array(npz_data['U']),  # Velocity field
        'water_level': np.array(npz_data['H']),  # Water level
        'concentrations': {}
    }
    
    # Calculate cross-section from water level (simplified)
    # Assume channel width of 1000m for flux calculations
    channel_width = 1000.0  # meters
    results['cross_section'] = results['water_level'] * channel_width
    
    # Extract concentrations for all species
    for species in SPECIES_NAMES:
        if species in npz_data:
            results['concentrations'][species] = np.array(npz_data[species])
    
    print(f"✅ Loaded data: {results['time'].shape[0]} time steps, {results['x'].shape[0]} grid points")
    return results

def analyze_pip_boundary_fluxes(results: Dict[str, Any]) -> Dict[str, Any]:
    """Detailed analysis of PIP boundary flux patterns."""
    print("\n🔬 ANALYZING PIP BOUNDARY FLUX PATTERNS")
    print("=" * 60)
    
    pip_data = results['concentrations']['PIP']
    velocity = results['velocity']
    cross_section = results['cross_section']
    time = results['time']
    x = results['x']
    
    # Calculate boundary fluxes
    # Upstream boundary (x=0, index 0)
    upstream_flux = velocity[:, 0] * cross_section[:, 0] * pip_data[:, 0]
    
    # Downstream boundary (x=max, index -1)
    downstream_flux = velocity[:, -1] * cross_section[:, -1] * pip_data[:, -1]
    
    # Net boundary flux (positive = loss from system)
    net_boundary_flux = downstream_flux - upstream_flux
    
    analysis = {
        'upstream_flux': upstream_flux,
        'downstream_flux': downstream_flux,
        'net_boundary_flux': net_boundary_flux,
        'cumulative_loss': np.cumsum(net_boundary_flux * np.gradient(time)),
        'flux_statistics': {
            'upstream_mean': np.mean(upstream_flux),
            'upstream_std': np.std(upstream_flux),
            'downstream_mean': np.mean(downstream_flux),
            'downstream_std': np.std(downstream_flux),
            'net_mean': np.mean(net_boundary_flux),
            'net_std': np.std(net_boundary_flux),
        }
    }
    
    print(f"📊 Upstream flux - Mean: {analysis['flux_statistics']['upstream_mean']:.2e}, Std: {analysis['flux_statistics']['upstream_std']:.2e}")
    print(f"📊 Downstream flux - Mean: {analysis['flux_statistics']['downstream_mean']:.2e}, Std: {analysis['flux_statistics']['downstream_std']:.2e}")
    print(f"📊 Net flux - Mean: {analysis['flux_statistics']['net_mean']:.2e}, Std: {analysis['flux_statistics']['net_std']:.2e}")
    print(f"📊 Total cumulative loss: {analysis['cumulative_loss'][-1]:.2e}")
    
    return analysis

def analyze_pip_initialization_effects(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze initialization effects on PIP mass balance."""
    print("\n🔬 ANALYZING PIP INITIALIZATION EFFECTS")
    print("=" * 60)
    
    pip_data = results['concentrations']['PIP']
    time = results['time']
    cross_section = results['cross_section']
    x = results['x']
    
    # Calculate total mass at each time step
    dx = np.gradient(x)
    total_mass = np.sum(pip_data * cross_section * dx[np.newaxis, :], axis=1)
    
    # Analyze initialization period (first 10% of simulation)
    init_period = int(0.1 * len(time))
    
    analysis = {
        'total_mass': total_mass,
        'initial_mass': total_mass[0],
        'final_mass': total_mass[-1],
        'mass_change': total_mass[-1] - total_mass[0],
        'relative_change': (total_mass[-1] - total_mass[0]) / total_mass[0] * 100,
        'initialization_analysis': {
            'mass_at_10_percent': total_mass[init_period],
            'init_period_change': total_mass[init_period] - total_mass[0],
            'post_init_change': total_mass[-1] - total_mass[init_period],
            'init_period_rate': (total_mass[init_period] - total_mass[0]) / time[init_period],
            'post_init_rate': (total_mass[-1] - total_mass[init_period]) / (time[-1] - time[init_period])
        }
    }
    
    print(f"📊 Initial mass: {analysis['initial_mass']:.2e}")
    print(f"📊 Final mass: {analysis['final_mass']:.2e}")
    print(f"📊 Total change: {analysis['mass_change']:.2e} ({analysis['relative_change']:.2f}%)")
    print(f"📊 Initialization period change: {analysis['initialization_analysis']['init_period_change']:.2e}")
    print(f"📊 Post-initialization change: {analysis['initialization_analysis']['post_init_change']:.2e}")
    print(f"📊 Initialization rate: {analysis['initialization_analysis']['init_period_rate']:.2e} mass/time")
    print(f"📊 Post-init rate: {analysis['initialization_analysis']['post_init_rate']:.2e} mass/time")
    
    return analysis

def analyze_pip_spatial_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze spatial patterns in PIP transport."""
    print("\n🔬 ANALYZING PIP SPATIAL TRANSPORT PATTERNS")
    print("=" * 60)
    
    pip_data = results['concentrations']['PIP']
    velocity = results['velocity']
    x = results['x']
    time = results['time']
    
    # Calculate spatial gradients
    spatial_gradients = np.gradient(pip_data, x[1] - x[0], axis=1)
    
    # Calculate advective flux
    advective_flux = velocity * pip_data
    
    # Find regions of highest mass loss
    mass_loss_rate = np.gradient(pip_data, np.gradient(time)[0], axis=0)
    
    analysis = {
        'spatial_gradients': spatial_gradients,
        'advective_flux': advective_flux,
        'mass_loss_rate': mass_loss_rate,
        'spatial_statistics': {
            'max_gradient_location': np.unravel_index(np.argmax(np.abs(spatial_gradients)), spatial_gradients.shape),
            'max_loss_location': np.unravel_index(np.argmax(np.abs(mass_loss_rate)), mass_loss_rate.shape),
            'mean_upstream_conc': np.mean(pip_data[:, :10]),  # First 10 grid points
            'mean_downstream_conc': np.mean(pip_data[:, -10:]),  # Last 10 grid points
            'concentration_ratio': np.mean(pip_data[:, -10:]) / np.mean(pip_data[:, :10])
        }
    }
    
    print(f"📊 Max gradient location: Time step {analysis['spatial_statistics']['max_gradient_location'][0]}, Grid {analysis['spatial_statistics']['max_gradient_location'][1]}")
    print(f"📊 Max loss location: Time step {analysis['spatial_statistics']['max_loss_location'][0]}, Grid {analysis['spatial_statistics']['max_loss_location'][1]}")
    print(f"📊 Mean upstream concentration: {analysis['spatial_statistics']['mean_upstream_conc']:.3f}")
    print(f"📊 Mean downstream concentration: {analysis['spatial_statistics']['mean_downstream_conc']:.3f}")
    print(f"📊 Downstream/Upstream ratio: {analysis['spatial_statistics']['concentration_ratio']:.3f}")
    
    return analysis

def create_boundary_analysis_plots(flux_analysis, init_analysis, spatial_analysis, output_dir="OUT/Analysis"):
    """Create comprehensive boundary analysis visualization."""
    print(f"\n📊 Creating boundary analysis plots in {output_dir}/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PIP Transport Boundary Analysis - Phase VII Task 16', fontsize=16, fontweight='bold')
    
    # 1. Boundary flux time series
    ax = axes[0, 0]
    time_hours = np.arange(len(flux_analysis['upstream_flux'])) * 0.05  # 3-minute steps to hours
    ax.plot(time_hours, flux_analysis['upstream_flux'], 'b-', label='Upstream flux', alpha=0.7)
    ax.plot(time_hours, flux_analysis['downstream_flux'], 'r-', label='Downstream flux', alpha=0.7)
    ax.plot(time_hours, flux_analysis['net_boundary_flux'], 'k-', label='Net flux', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('PIP Flux (mass/time)')
    ax.set_title('Boundary Flux Patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative mass loss
    ax = axes[0, 1]
    ax.plot(time_hours, flux_analysis['cumulative_loss'], 'r-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cumulative Mass Loss')
    ax.set_title('Cumulative PIP Mass Loss')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Total mass evolution
    ax = axes[0, 2]
    ax.plot(time_hours, init_analysis['total_mass'], 'g-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Total PIP Mass')
    ax.set_title('Total Mass Evolution')
    ax.grid(True, alpha=0.3)
    
    # Add initialization period marker
    init_marker = int(0.1 * len(time_hours))
    ax.axvline(x=time_hours[init_marker], color='r', linestyle='--', alpha=0.7, label='End of init period')
    ax.legend()
    
    # 4. Spatial concentration pattern (average)
    ax = axes[1, 0]
    x_km = np.arange(spatial_analysis['advective_flux'].shape[1]) * 2  # 2km grid spacing
    mean_concentration = np.mean(spatial_analysis['advective_flux'], axis=0)
    ax.plot(x_km, mean_concentration, 'b-', linewidth=2)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Mean PIP Concentration')
    ax.set_title('Spatial Distribution Pattern')
    ax.grid(True, alpha=0.3)
    
    # 5. Mass loss rate spatial pattern
    ax = axes[1, 1]
    mean_loss_rate = np.mean(spatial_analysis['mass_loss_rate'], axis=0)
    ax.plot(x_km, mean_loss_rate, 'r-', linewidth=2)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Mean Mass Loss Rate')
    ax.set_title('Spatial Loss Rate Distribution')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 6. Flux statistics summary
    ax = axes[1, 2]
    flux_stats = flux_analysis['flux_statistics']
    labels = ['Upstream\nFlux', 'Downstream\nFlux', 'Net\nFlux']
    means = [flux_stats['upstream_mean'], flux_stats['downstream_mean'], flux_stats['net_mean']]
    stds = [flux_stats['upstream_std'], flux_stats['downstream_std'], flux_stats['net_std']]
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['blue', 'red', 'black'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Flux (mean ± std)')
    ax.set_title('Boundary Flux Statistics')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'pip_boundary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Plots saved to {output_dir}/pip_boundary_analysis.png")

def generate_boundary_analysis_report(flux_analysis, init_analysis, spatial_analysis, output_dir="OUT/Analysis"):
    """Generate comprehensive boundary analysis report."""
    report_file = Path(output_dir) / "pip_boundary_analysis_report.txt"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("PIP TRANSPORT BOUNDARY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write("Phase VII Task 16: PIP Transport Boundary Investigation\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"PIP Mass Loss: {init_analysis['relative_change']:.2f}%\n")
        f.write(f"Total Mass Change: {init_analysis['mass_change']:.2e}\n")
        f.write(f"Net Boundary Flux: {flux_analysis['flux_statistics']['net_mean']:.2e} ± {flux_analysis['flux_statistics']['net_std']:.2e}\n")
        f.write(f"Cumulative Boundary Loss: {flux_analysis['cumulative_loss'][-1]:.2e}\n\n")
        
        f.write("DETAILED FINDINGS\n")
        f.write("-" * 20 + "\n")
        
        f.write("1. BOUNDARY FLUX ANALYSIS:\n")
        f.write(f"   Upstream flux mean: {flux_analysis['flux_statistics']['upstream_mean']:.2e}\n")
        f.write(f"   Downstream flux mean: {flux_analysis['flux_statistics']['downstream_mean']:.2e}\n")
        f.write(f"   Net flux mean: {flux_analysis['flux_statistics']['net_mean']:.2e}\n")
        f.write(f"   Net flux variability: {flux_analysis['flux_statistics']['net_std']:.2e}\n\n")
        
        f.write("2. INITIALIZATION EFFECTS:\n")
        f.write(f"   Initial mass: {init_analysis['initial_mass']:.2e}\n")
        f.write(f"   Mass at 10% simulation: {init_analysis['initialization_analysis']['mass_at_10_percent']:.2e}\n")
        f.write(f"   Initialization period change: {init_analysis['initialization_analysis']['init_period_change']:.2e}\n")
        f.write(f"   Post-initialization change: {init_analysis['initialization_analysis']['post_init_change']:.2e}\n")
        f.write(f"   Initialization rate: {init_analysis['initialization_analysis']['init_period_rate']:.2e}\n")
        f.write(f"   Post-initialization rate: {init_analysis['initialization_analysis']['post_init_rate']:.2e}\n\n")
        
        f.write("3. SPATIAL PATTERNS:\n")
        f.write(f"   Upstream concentration: {spatial_analysis['spatial_statistics']['mean_upstream_conc']:.3f}\n")
        f.write(f"   Downstream concentration: {spatial_analysis['spatial_statistics']['mean_downstream_conc']:.3f}\n")
        f.write(f"   Downstream/Upstream ratio: {spatial_analysis['spatial_statistics']['concentration_ratio']:.3f}\n\n")
        
        f.write("RECOMMENDATIONS FOR TASK 17:\n")
        f.write("-" * 30 + "\n")
        
        # Determine primary cause based on analysis
        if abs(flux_analysis['flux_statistics']['net_mean']) > abs(init_analysis['initialization_analysis']['post_init_rate']):
            primary_cause = "Boundary flux imbalance"
            f.write("PRIMARY ISSUE: Boundary flux imbalance dominates mass loss\n")
            f.write("RECOMMENDED ACTION: Focus on boundary condition optimization\n")
        else:
            primary_cause = "Initialization transients"
            f.write("PRIMARY ISSUE: Initialization effects dominate mass loss\n")
            f.write("RECOMMENDED ACTION: Focus on equilibrium initialization\n")
        
        if spatial_analysis['spatial_statistics']['concentration_ratio'] < 0.5:
            f.write("SECONDARY ISSUE: Strong downstream depletion suggests advection issues\n")
            f.write("RECOMMENDED ACTION: Investigate TVD advection scheme for particulates\n")
        
        f.write("\nNEXT STEPS:\n")
        f.write("1. Implement mass-conserving boundary conditions for PIP\n")
        f.write("2. Optimize TVD advection scheme for particulate transport\n")
        f.write("3. Enhance equilibrium initialization for PIP\n")
        f.write("4. Validate improvements with mass conservation tests\n")
    
    print(f"📋 Report saved to: {report_file}")
    return report_file

def main():
    """Main analysis routine for PIP boundary investigation."""
    print("🚀 PHASE VII TASK 16: PIP TRANSPORT BOUNDARY INVESTIGATION")
    print("=" * 70)
    
    # Load simulation data
    results = load_simulation_data()
    if results is None:
        print("❌ Cannot proceed without simulation data")
        return
    
    # Perform detailed analyses
    flux_analysis = analyze_pip_boundary_fluxes(results)
    init_analysis = analyze_pip_initialization_effects(results)
    spatial_analysis = analyze_pip_spatial_patterns(results)
    
    # Generate visualizations and report
    create_boundary_analysis_plots(flux_analysis, init_analysis, spatial_analysis)
    report_file = generate_boundary_analysis_report(flux_analysis, init_analysis, spatial_analysis)
    
    print(f"\n✅ PIP BOUNDARY ANALYSIS COMPLETE")
    print(f"📋 Detailed report: {report_file}")
    print(f"📊 Visualization: OUT/Analysis/pip_boundary_analysis.png")
    
    # Summary recommendations
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   • PIP mass loss: {init_analysis['relative_change']:.2f}%")
    print(f"   • Net boundary flux: {flux_analysis['flux_statistics']['net_mean']:.2e}")
    print(f"   • Cumulative loss: {flux_analysis['cumulative_loss'][-1]:.2e}")
    print(f"   • Spatial concentration ratio: {spatial_analysis['spatial_statistics']['concentration_ratio']:.3f}")
    
    print(f"\n➡️  READY FOR TASK 17: Advanced Transport Solver Optimization")

if __name__ == "__main__":
    main()