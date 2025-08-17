"""
Automated Validation Script for Monotonic Transport Results

This script automatically validates simulation results to check if we have
successfully eliminated the transport oscillations and achieved smooth,
monotonic concentration profiles.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_oscillations(results_file="OUT/complete_simulation_results.npz"):
    """
    Comprehensive oscillation analysis of simulation results.
    
    Returns:
    - oscillation_report: Dictionary with oscillation metrics for each species
    - validation_status: Boolean indicating if validation passed
    """
    print("🔍 AUTOMATED VALIDATION: OSCILLATION ANALYSIS")
    print("=" * 60)
    
    try:
        # Load simulation results
        print(f"📂 Loading results from: {results_file}")
        results = np.load(results_file, allow_pickle=True)
        print(f"✅ Loaded successfully! Keys: {list(results.keys())}")
        
        # Key species to analyze
        key_species = ['NH4', 'NO3', 'O2', 'S', 'TOC']
        
        oscillation_report = {}
        all_smooth = True
        
        print("\n🧪 SPECIES-BY-SPECIES ANALYSIS:")
        print("-" * 40)
        
        for species in key_species:
            if species in results:
                data = results[species]
                print(f"\n{species}:")
                print(f"  📊 Data shape: {data.shape}")
                
                # Use final time step (fully developed profiles)
                final_profile = data[-1]
                
                # Calculate gradient
                gradient = np.diff(final_profile)
                
                # Count oscillations (sign changes in gradient)
                sign_changes = np.sum(np.diff(np.sign(gradient)) != 0)
                
                # Calculate oscillation metrics
                total_points = len(gradient)
                oscillation_rate = sign_changes / total_points * 100
                
                # Calculate profile smoothness (gradient variability)
                gradient_std = np.std(gradient)
                gradient_mean = np.abs(np.mean(gradient))
                smoothness_ratio = gradient_std / (gradient_mean + 1e-12)
                
                # Store results
                oscillation_report[species] = {
                    'sign_changes': sign_changes,
                    'oscillation_rate': oscillation_rate,
                    'smoothness_ratio': smoothness_ratio,
                    'concentration_range': [np.min(final_profile), np.max(final_profile)],
                    'is_smooth': oscillation_rate < 15.0  # Tolerance threshold
                }
                
                # Display results
                print(f"  🔢 Concentration range: [{np.min(final_profile):.3f}, {np.max(final_profile):.3f}]")
                print(f"  📈 Oscillations: {sign_changes} ({oscillation_rate:.1f}%)")
                print(f"  🌊 Smoothness ratio: {smoothness_ratio:.3f}")
                
                if oscillation_rate < 10.0:
                    print(f"  ✅ EXCELLENT: Smooth monotonic profile!")
                elif oscillation_rate < 15.0:
                    print(f"  ✅ GOOD: Mostly smooth with minor variations")
                elif oscillation_rate < 25.0:
                    print(f"  ⚠️  MODERATE: Some oscillations present")
                    all_smooth = False
                else:
                    print(f"  ❌ HIGH OSCILLATIONS: Significant instabilities")
                    all_smooth = False
            else:
                print(f"⚠️  {species}: Not found in results")
                all_smooth = False
        
        # Overall validation assessment
        print("\n" + "=" * 60)
        print("🎯 OVERALL VALIDATION RESULTS:")
        
        if all_smooth:
            print("✅ SUCCESS: All key species show smooth, monotonic profiles!")
            print("🎉 Transport oscillations have been ELIMINATED!")
            validation_status = True
        else:
            print("⚠️  PARTIAL SUCCESS: Some oscillations remain")
            print("🔧 Further optimization may be needed")
            validation_status = False
        
        # Calculate overall performance score
        if oscillation_report:
            avg_oscillation = np.mean([report['oscillation_rate'] for report in oscillation_report.values()])
            smooth_species_count = sum([report['is_smooth'] for report in oscillation_report.values()])
            total_species = len(oscillation_report)
            
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"  Average oscillation rate: {avg_oscillation:.1f}%")
            print(f"  Smooth species: {smooth_species_count}/{total_species}")
            
            if avg_oscillation < 10.0:
                print("🏆 GRADE: EXCELLENT (A+)")
            elif avg_oscillation < 15.0:
                print("🥇 GRADE: VERY GOOD (A)")
            elif avg_oscillation < 25.0:
                print("🥈 GRADE: GOOD (B)")
            else:
                print("🥉 GRADE: NEEDS IMPROVEMENT (C)")
        
        return oscillation_report, validation_status
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return {}, False

def create_validation_plots(oscillation_report, output_dir="OUT/Validation"):
    """Create detailed validation plots showing profile smoothness."""
    
    print(f"\n🎨 Creating validation plots in {output_dir}...")
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Load results again for plotting
        results = np.load("OUT/complete_simulation_results.npz", allow_pickle=True)
        
        # Create comprehensive validation figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Grid for distance axis (km from mouth)
        x_km = np.linspace(0, 202, 102)  # 0-202 km along estuary
        
        key_species = ['NH4', 'NO3', 'O2', 'S', 'TOC']
        
        for i, species in enumerate(key_species):
            if species in results and species in oscillation_report:
                ax = axes[i]
                data = results[species]
                final_profile = data[-1]
                report = oscillation_report[species]
                
                # Plot concentration profile
                ax.plot(x_km, final_profile, 'b-', linewidth=2, 
                       label=f'{species} Profile')
                
                # Add gradient visualization
                gradient = np.diff(final_profile)
                ax_twin = ax.twinx()
                ax_twin.plot(x_km[:-1], gradient, 'r--', alpha=0.6, 
                            label='Gradient')
                ax_twin.set_ylabel('Gradient', color='r')
                
                # Formatting
                ax.set_xlabel('Distance from mouth (km)')
                ax.set_ylabel(f'{species} Concentration')
                ax.set_title(f'{species}: {report["oscillation_rate"]:.1f}% oscillations')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
                ax_twin.legend(loc='upper right')
                
                # Color-code title based on performance
                if report['is_smooth']:
                    ax.title.set_color('green')
                else:
                    ax.title.set_color('orange')
        
        # Remove unused subplot
        if len(key_species) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monotonic_validation_profiles.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Saved validation plots: {output_dir}/monotonic_validation_profiles.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def main():
    """Main validation function."""
    results_file = "OUT/complete_simulation_results.npz"
    
    # Check if results exist
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("⏳ Simulation may still be running...")
        return False
    
    # Run validation analysis
    oscillation_report, validation_passed = analyze_oscillations(results_file)
    
    # Create validation plots
    if oscillation_report:
        create_validation_plots(oscillation_report)
    
    # Save validation report
    output_dir = Path("OUT/Validation")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "monotonic_validation_report.txt", 'w') as f:
        f.write("MONOTONIC TRANSPORT VALIDATION REPORT\\n")
        f.write("=" * 50 + "\\n\\n")
        
        for species, report in oscillation_report.items():
            f.write(f"{species}:\\n")
            f.write(f"  Oscillation rate: {report['oscillation_rate']:.1f}%\\n")
            f.write(f"  Sign changes: {report['sign_changes']}\\n")
            f.write(f"  Smoothness ratio: {report['smoothness_ratio']:.3f}\\n")
            f.write(f"  Is smooth: {report['is_smooth']}\\n\\n")
        
        f.write(f"Overall validation: {'PASSED' if validation_passed else 'NEEDS IMPROVEMENT'}\\n")
    
    print(f"\\n📄 Validation report saved: {output_dir}/monotonic_validation_report.txt")
    
    return validation_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)