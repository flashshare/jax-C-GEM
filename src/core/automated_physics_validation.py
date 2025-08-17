"""
Automated Physics Validation Module for JAX C-GEM

This module provides comprehensive automated physics validation that runs after
each simulation to verify estuarine dynamics and provide detailed diagnostic output.

Features:
1. Downstream-to-upstream longitudinal profile analysis
2. Tidal range, salinity, and water quality validation
3. Automated figure generation and CSV export
4. Physics quality assessment with recommendations
5. Integration with 3-phase verification workflow

Author: Nguyen Truong An
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_simulation_results(results_file="OUT/complete_simulation_results.npz"):
    """Load simulation results from NPZ or CSV format"""
    
    if os.path.exists(results_file):
        print(f"📂 Loading NPZ results from: {results_file}")
        return np.load(results_file)
    
    # Fall back to CSV format
    print("📂 Loading CSV results...")
    csv_results = {}
    
    # Load hydrodynamics
    if os.path.exists("OUT/Hydrodynamics/H.csv"):
        csv_results['H'] = np.loadtxt("OUT/Hydrodynamics/H.csv", delimiter=',')
        csv_results['U'] = np.loadtxt("OUT/Hydrodynamics/U.csv", delimiter=',')
    
    # Load species data
    species_files = ['NH4', 'NO3', 'PO4', 'O2', 'S', 'SPM', 'DIC']
    for species in species_files:
        csv_file = f"OUT/Reaction/{species}.csv"
        if os.path.exists(csv_file):
            csv_results[species] = np.loadtxt(csv_file, delimiter=',')
    
    return csv_results


def calculate_mean_profiles(results: Dict[str, np.ndarray], warmup_fraction: float = 0.3) -> Dict[str, np.ndarray]:
    """Calculate time-averaged profiles excluding warmup period"""
    
    print("📊 Calculating mean longitudinal profiles...")
    
    mean_profiles = {}
    
    for species, data in results.items():
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            n_timesteps = data.shape[0]
            warmup_steps = int(n_timesteps * warmup_fraction)
            
            # Time average excluding warmup
            mean_profiles[species] = np.mean(data[warmup_steps:], axis=0)
    
    return mean_profiles


def calculate_tidal_amplitudes(H_data: np.ndarray, warmup_fraction: float = 0.3) -> np.ndarray:
    """Calculate tidal amplitude at each grid point"""
    
    n_timesteps = H_data.shape[0]
    warmup_steps = int(n_timesteps * warmup_fraction)
    
    # Calculate tidal amplitude as standard deviation of water level
    stable_H = H_data[warmup_steps:]
    tidal_amplitudes = np.std(stable_H, axis=0)
    
    return tidal_amplitudes


def create_distance_grid(EL: float = 202000.0, DELXI: float = 2000.0) -> np.ndarray:
    """Create distance array from mouth (km)"""
    
    M = int(EL // DELXI) + 1
    distances = np.linspace(0, EL/1000, M)  # Convert to km
    
    return distances


def validate_estuarine_physics(mean_profiles: Dict[str, np.ndarray], 
                             tidal_amplitudes: np.ndarray,
                             distance_km: np.ndarray) -> Dict[str, Any]:
    """Comprehensive physics validation"""
    
    print("🔬 Validating estuarine physics...")
    
    validation_results = {
        'physics_quality': {},
        'gradient_analysis': {},
        'recommendations': [],
        'overall_status': 'UNKNOWN'
    }
    
    # Salinity validation
    if 'S' in mean_profiles:
        salinity = mean_profiles['S']
        
        # Check salinity gradient (should increase downstream)
        if len(salinity) > 1:
            sal_gradient = np.gradient(salinity)
            increasing_fraction = np.sum(sal_gradient > 0) / len(sal_gradient)
            
            validation_results['physics_quality']['salinity'] = {
                'gradient_direction': 'CORRECT' if increasing_fraction > 0.7 else 'INCORRECT',
                'smoothness': calculate_smoothness(salinity),
                'range': f"{salinity.min():.2f} - {salinity.max():.2f}",
                'increasing_fraction': increasing_fraction
            }
    
    # Nutrient validation (should decrease downstream)
    for nutrient in ['NH4', 'NO3', 'PO4']:
        if nutrient in mean_profiles:
            conc = mean_profiles[nutrient]
            if len(conc) > 1:
                nutrient_gradient = np.gradient(conc)
                decreasing_fraction = np.sum(nutrient_gradient < 0) / len(nutrient_gradient)
                
                validation_results['physics_quality'][nutrient] = {
                    'gradient_direction': 'CORRECT' if decreasing_fraction > 0.7 else 'INCORRECT',
                    'smoothness': calculate_smoothness(conc),
                    'range': f"{conc.min():.3f} - {conc.max():.3f}",
                    'decreasing_fraction': decreasing_fraction
                }
    
    # Oxygen validation (should increase downstream)
    if 'O2' in mean_profiles:
        oxygen = mean_profiles['O2']
        if len(oxygen) > 1:
            o2_gradient = np.gradient(oxygen)
            increasing_fraction = np.sum(o2_gradient > 0) / len(o2_gradient)
            
            validation_results['physics_quality']['O2'] = {
                'gradient_direction': 'CORRECT' if increasing_fraction > 0.6 else 'INCORRECT',
                'smoothness': calculate_smoothness(oxygen),
                'range': f"{oxygen.min():.3f} - {oxygen.max():.3f}",
                'increasing_fraction': increasing_fraction
            }
    
    # Tidal amplitude validation
    if len(tidal_amplitudes) > 1:
        tidal_gradient = np.gradient(tidal_amplitudes)
        decreasing_fraction = np.sum(tidal_gradient < 0) / len(tidal_gradient)
        
        validation_results['physics_quality']['tidal_amplitude'] = {
            'gradient_direction': 'CORRECT' if decreasing_fraction > 0.6 else 'INCORRECT',
            'smoothness': calculate_smoothness(tidal_amplitudes),
            'range': f"{tidal_amplitudes.min():.3f} - {tidal_amplitudes.max():.3f}",
            'decreasing_fraction': decreasing_fraction
        }
    
    # Overall assessment
    correct_count = 0
    total_count = 0
    
    for species, metrics in validation_results['physics_quality'].items():
        if 'gradient_direction' in metrics:
            total_count += 1
            if metrics['gradient_direction'] == 'CORRECT':
                correct_count += 1
    
    if total_count > 0:
        success_rate = correct_count / total_count
        if success_rate >= 0.8:
            validation_results['overall_status'] = 'EXCELLENT'
        elif success_rate >= 0.6:
            validation_results['overall_status'] = 'GOOD'
        elif success_rate >= 0.4:
            validation_results['overall_status'] = 'FAIR'
        else:
            validation_results['overall_status'] = 'POOR'
    
    return validation_results


def calculate_smoothness(profile: np.ndarray) -> str:
    """Calculate profile smoothness metric"""
    
    if len(profile) < 3:
        return "INSUFFICIENT_DATA"
    
    # Calculate second derivative as smoothness measure
    second_deriv = np.gradient(np.gradient(profile))
    roughness = np.std(second_deriv)
    
    if roughness < 0.01:
        return "EXCELLENT"
    elif roughness < 0.05:
        return "GOOD"
    elif roughness < 0.1:
        return "FAIR"
    else:
        return "POOR"


def print_longitudinal_profiles(mean_profiles: Dict[str, np.ndarray],
                              tidal_amplitudes: np.ndarray,
                              distance_km: np.ndarray,
                              validation_results: Dict[str, Any]):
    """Print detailed downstream-to-upstream profiles"""
    
    print("\n" + "="*80)
    print("🌊 LONGITUDINAL PROFILES: DOWNSTREAM → UPSTREAM")
    print("="*80)
    
    # Print every 5 grid points for readability
    step = max(1, len(distance_km) // 20)  # ~20 data points
    
    print(f"{'Distance (km)':<12} {'Tidal Amp (m)':<12} {'Salinity':<10} {'NH4':<8} {'NO3':<8} {'PO4':<8} {'O2':<8}")
    print("-" * 80)
    
    for i in range(0, len(distance_km), step):
        row = f"{distance_km[i]:8.1f}     "
        
        # Tidal amplitude
        row += f"{tidal_amplitudes[i]:8.3f}     "
        
        # Salinity
        if 'S' in mean_profiles:
            row += f"{mean_profiles['S'][i]:6.2f}   "
        else:
            row += f"{'N/A':>6}   "
        
        # Nutrients and oxygen
        for species in ['NH4', 'NO3', 'PO4', 'O2']:
            if species in mean_profiles:
                row += f"{mean_profiles[species][i]:6.3f}  "
            else:
                row += f"{'N/A':>6}  "
        
        print(row)
    
    print("-" * 80)


def create_validation_figures(mean_profiles: Dict[str, np.ndarray],
                            tidal_amplitudes: np.ndarray,
                            distance_km: np.ndarray,
                            validation_results: Dict[str, Any],
                            output_dir: str = "OUT/Validation"):
    """Create comprehensive validation figures"""
    
    print(f"🎨 Creating validation figures in {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Automated Physics Validation - Longitudinal Profiles', fontsize=16, fontweight='bold')
    
    # Panel 1: Tidal Amplitude and Salinity
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    # Tidal amplitude
    line1 = ax1.plot(distance_km, tidal_amplitudes, 'b-', linewidth=2, label='Tidal Amplitude')
    ax1.set_ylabel('Tidal Amplitude (m)', color='b', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Salinity
    if 'S' in mean_profiles:
        line2 = ax1_twin.plot(distance_km, mean_profiles['S'], 'r-', linewidth=2, label='Salinity')
        ax1_twin.set_ylabel('Salinity (PSU)', color='r', fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # Add status indicator
        sal_status = validation_results['physics_quality'].get('S', {}).get('gradient_direction', 'UNKNOWN')
        ax1.text(0.05, 0.95, f'Salinity Gradient: {sal_status}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if sal_status == 'CORRECT' else 'lightcoral'),
                fontweight='bold', verticalalignment='top')
    
    ax1.set_xlabel('Distance from Mouth (km)', fontweight='bold')
    ax1.set_title('Hydrodynamics & Salinity', fontweight='bold')
    
    # Panel 2: Nutrients
    ax2 = axes[0, 1]
    colors = ['green', 'orange', 'purple']
    nutrients = ['NH4', 'NO3', 'PO4']
    
    for i, nutrient in enumerate(nutrients):
        if nutrient in mean_profiles:
            ax2.plot(distance_km, mean_profiles[nutrient], color=colors[i], 
                    linewidth=2, label=f'{nutrient}', marker='o', markersize=3)
    
    ax2.set_xlabel('Distance from Mouth (km)', fontweight='bold')
    ax2.set_ylabel('Concentration (mmol/m³)', fontweight='bold')
    ax2.set_title('Nutrient Profiles', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Dissolved Oxygen
    ax3 = axes[1, 0]
    if 'O2' in mean_profiles:
        ax3.plot(distance_km, mean_profiles['O2'], 'cyan', linewidth=2, marker='s', markersize=3)
        o2_status = validation_results['physics_quality'].get('O2', {}).get('gradient_direction', 'UNKNOWN')
        ax3.text(0.05, 0.95, f'O2 Gradient: {o2_status}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if o2_status == 'CORRECT' else 'lightcoral'),
                fontweight='bold', verticalalignment='top')
    
    ax3.set_xlabel('Distance from Mouth (km)', fontweight='bold')
    ax3.set_ylabel('Dissolved Oxygen (mmol/m³)', fontweight='bold')
    ax3.set_title('Dissolved Oxygen Profile', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Validation Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create validation summary text
    summary_text = "PHYSICS VALIDATION SUMMARY\n\n"
    summary_text += f"Overall Status: {validation_results['overall_status']}\n\n"
    
    for species, metrics in validation_results['physics_quality'].items():
        if 'gradient_direction' in metrics:
            status = "✅" if metrics['gradient_direction'] == 'CORRECT' else "❌"
            summary_text += f"{status} {species}: {metrics['gradient_direction']}\n"
            summary_text += f"   Range: {metrics['range']}\n"
            summary_text += f"   Smoothness: {metrics['smoothness']}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Path(output_dir) / "automated_physics_validation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved validation figure: {fig_path}")
    
    plt.close()
    
    return fig_path


def save_profiles_csv(mean_profiles: Dict[str, np.ndarray],
                     tidal_amplitudes: np.ndarray,
                     distance_km: np.ndarray,
                     output_dir: str = "OUT/Validation"):
    """Save longitudinal profiles to CSV for easy analysis"""
    
    print(f"💾 Saving profiles CSV to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    data = {
        'Distance_km': distance_km,
        'Tidal_Amplitude_m': tidal_amplitudes
    }
    
    # Add all available species
    for species, profile in mean_profiles.items():
        if isinstance(profile, np.ndarray) and len(profile) == len(distance_km):
            data[f'{species}_mean'] = profile
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = Path(output_dir) / "longitudinal_profiles_mean.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"📊 Saved profiles CSV: {csv_path}")
    
    return csv_path


def generate_recommendations(validation_results: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on validation results"""
    
    recommendations = []
    
    overall_status = validation_results['overall_status']
    
    if overall_status == 'EXCELLENT':
        recommendations.append("🎉 Physics validation PASSED! Model exhibits excellent estuarine behavior.")
        recommendations.append("✅ Recommended next step: Run 3-phase verification with observed data")
        recommendations.append("   Command: python tools/verification/phase1_longitudinal_profiles.py")
        recommendations.append("   Command: python tools/verification/phase2_tidal_dynamics.py") 
        recommendations.append("   Command: python tools/verification/phase3_seasonal_cycles.py")
        
    elif overall_status in ['GOOD', 'FAIR']:
        recommendations.append(f"⚠️  Physics validation shows {overall_status} results. Some issues detected:")
        
        # Analyze specific problems
        for species, metrics in validation_results['physics_quality'].items():
            if metrics.get('gradient_direction') == 'INCORRECT':
                if species == 'S':
                    recommendations.append(f"❌ Salinity gradient incorrect - check boundary conditions")
                    recommendations.append("   Debug: Verify downstream salinity > upstream salinity")
                elif species in ['NH4', 'NO3', 'PO4']:
                    recommendations.append(f"❌ {species} gradient incorrect - check nutrient sources")
                    recommendations.append("   Debug: Verify upstream nutrients > downstream nutrients")
                elif species == 'O2':
                    recommendations.append(f"❌ Oxygen gradient incorrect - check biogeochemical processes")
                    recommendations.append("   Debug: Verify downstream O2 > upstream O2")
        
        recommendations.append("🔧 Suggested fixes: Check boundary conditions and biogeochemical parameters")
        
    else:  # POOR
        recommendations.append("❌ Physics validation FAILED! Major issues detected:")
        recommendations.append("🚨 Model does not exhibit realistic estuarine behavior")
        
        # Provide detailed debugging
        recommendations.append("\n🔍 Debugging checklist:")
        recommendations.append("   1. Check boundary condition files in INPUT/Boundary/")
        recommendations.append("   2. Verify tributary discharge and concentrations")
        recommendations.append("   3. Check biogeochemical parameter values")
        recommendations.append("   4. Review transport solver stability")
        recommendations.append("   5. Examine initial conditions")
        
        recommendations.append("\n📋 DO NOT proceed to 3-phase verification until physics is corrected")
    
    return recommendations


def run_automated_physics_validation(results_file: str = "OUT/complete_simulation_results.npz",
                                   output_dir: str = "OUT/Validation") -> Dict[str, Any]:
    """Main function to run complete automated physics validation"""
    
    print("\n" + "="*80)
    print("🔬 AUTOMATED PHYSICS VALIDATION")
    print("="*80)
    
    try:
        # Load results
        results = load_simulation_results(results_file)
        
        # Calculate mean profiles
        mean_profiles = calculate_mean_profiles(results)
        
        # Calculate tidal amplitudes
        tidal_amplitudes = None
        if 'H' in results:
            tidal_amplitudes = calculate_tidal_amplitudes(results['H'])
        
        # Create distance grid
        distance_km = create_distance_grid()
        
        # Ensure arrays have consistent length
        min_length = min(len(distance_km), 
                        min(len(profile) for profile in mean_profiles.values() if isinstance(profile, np.ndarray)))
        
        distance_km = distance_km[:min_length]
        
        for species in mean_profiles:
            if isinstance(mean_profiles[species], np.ndarray):
                mean_profiles[species] = mean_profiles[species][:min_length]
        
        if tidal_amplitudes is not None:
            tidal_amplitudes = tidal_amplitudes[:min_length]
        else:
            tidal_amplitudes = np.zeros(min_length)
        
        # Run validation
        validation_results = validate_estuarine_physics(mean_profiles, tidal_amplitudes, distance_km)
        
        # Print detailed profiles
        print_longitudinal_profiles(mean_profiles, tidal_amplitudes, distance_km, validation_results)
        
        # Create figures
        fig_path = create_validation_figures(mean_profiles, tidal_amplitudes, distance_km, 
                                           validation_results, output_dir)
        
        # Save CSV
        csv_path = save_profiles_csv(mean_profiles, tidal_amplitudes, distance_km, output_dir)
        
        # Generate recommendations
        recommendations = generate_recommendations(validation_results)
        
        # Print recommendations
        print("\n" + "="*80)
        print("🎯 VALIDATION RECOMMENDATIONS")
        print("="*80)
        for rec in recommendations:
            print(rec)
        print("="*80)
        
        # Return comprehensive results
        return {
            'validation_results': validation_results,
            'recommendations': recommendations,
            'figure_path': str(fig_path),
            'csv_path': str(csv_path),
            'mean_profiles': mean_profiles
        }
        
    except Exception as e:
        print(f"❌ Physics validation failed: {e}")
        return {
            'validation_results': {'overall_status': 'ERROR'},
            'recommendations': [f"❌ Validation error: {e}"],
            'figure_path': None,
            'csv_path': None,
            'mean_profiles': {}
        }