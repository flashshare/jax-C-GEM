#!/usr/bin/env python
"""
Standalone Estuarine Physics Validation Script

This script runs comprehensive physics validation on JAX C-GEM simulation results
without requiring any observed data. It validates against fundamental estuarine
principles and theoretical expectations.

Validation Categories:
1. 🌊 Salinity Intrusion Length (Savenije theory)
2. 🌀 Tidal Amplitude Decay (geometric/friction effects)  
3. 💧 Water Quality Gradients (mass balance/dilution)
4. ⚡ Tidal Flow Dynamics (flood/ebb reversals)

Usage:
    python tools/validation/standalone_physics_validation.py
    python tools/validation/standalone_physics_validation.py --results-dir OUT --config config/model_config.txt
    python tools/validation/standalone_physics_validation.py --npz-file OUT/simulation_results.npz

Author: Nguyen Truong An
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Standalone Estuarine Physics Validation")
    
    parser.add_argument('--results-dir', default='OUT', 
                       help='Directory containing simulation results (default: OUT)')
    parser.add_argument('--config', default='config/model_config.txt',
                       help='Model configuration file (default: config/model_config.txt)')
    parser.add_argument('--npz-file', 
                       help='Specific NPZ file to validate (overrides results-dir)')
    parser.add_argument('--output-dir', default='OUT/PhysicsValidation',
                       help='Output directory for validation reports and figures')
    parser.add_argument('--create-figures', action='store_true',
                       help='Create validation figures (requires matplotlib)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output (default: True)')
    
    return parser

def load_simulation_results(npz_file=None, results_dir="OUT"):
    """Load simulation results from NPZ file."""
    
    if npz_file:
        target_file = Path(npz_file)
    else:
        # Look for NPZ files in results directory
        results_path = Path(results_dir)
        npz_files = list(results_path.glob("*.npz"))
        
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {results_dir}")
        
        # Use the most recent file
        target_file = max(npz_files, key=lambda f: f.stat().st_mtime)
    
    if not target_file.exists():
        raise FileNotFoundError(f"Results file not found: {target_file}")
    
    print(f"📂 Loading simulation results from: {target_file}")
    
    # Load NPZ data
    npz_data = np.load(target_file)
    
    # Convert to structured results dictionary
    results = {
        'time': npz_data['time'],
        'H': npz_data['H'],
        'U': npz_data['U'],
        'species': {}
    }
    
    # Load species data
    try:
        from core.model_config import SPECIES_NAMES
    except ImportError:
        SPECIES_NAMES = ['PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 
                        'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2']
    
    available_species = []
    for species in SPECIES_NAMES:
        if species in npz_data:
            results['species'][species] = npz_data[species]
            available_species.append(species)
    
    print(f"   ✅ Loaded data:")
    print(f"      ⏰ Time steps: {len(results['time']):,}")
    print(f"      🌊 Grid cells: {results['H'].shape[1]}")
    print(f"      🧪 Species: {len(available_species)} ({', '.join(available_species[:5])}{'...' if len(available_species) > 5 else ''})")
    
    return results

def create_physics_validation_figures(validation_report, output_dir="OUT/PhysicsValidation"):
    """Create comprehensive physics validation figures."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Theoretical vs Simulated Profiles
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Estuarine Physics Validation - Theoretical vs Simulated', fontsize=16, fontweight='bold')
    
    # Extract data from validation report
    geometry = validation_report['geometry']
    theory = validation_report['theory']
    results = validation_report['validation_results']
    
    grid_locations = geometry['grid_locations']
    
    # Subplot 1: Salinity Intrusion
    ax = axes[0, 0]
    if 'salinity_intrusion' in results:
        si = results['salinity_intrusion']
        
        # Plot theoretical intrusion length
        theory_intrusion = theory['salinity_intrusion_length']
        ax.axvline(theory_intrusion, color='blue', linestyle='--', linewidth=2, 
                  label=f'Theory: {theory_intrusion:.1f} km')
        
        if 'simulated_intrusion_km' in si:
            sim_intrusion = si['simulated_intrusion_km']
            ax.axvline(sim_intrusion, color='red', linestyle='-', linewidth=2,
                      label=f'Simulated: {sim_intrusion:.1f} km')
        
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_ylabel('Salinity Intrusion')
        ax.set_title(f'Salinity Intrusion Length\nValidation: {si.get("status", "N/A").upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Subplot 2: Tidal Amplitude Decay
    ax = axes[0, 1]
    if 'tidal_dynamics' in results and 'tidal_amplitude' in results['tidal_dynamics']:
        ta = results['tidal_dynamics']['tidal_amplitude']
        
        # Theoretical decay
        theory_mouth = theory['expected_tidal_range_decay'][0]
        theory_upstream = theory['expected_tidal_range_decay'][1]
        theory_decay = np.exp(-theory['tidal_amplitude_decay_rate'] * grid_locations)
        theory_profile = theory_mouth * theory_decay
        
        ax.plot(grid_locations, theory_profile, 'b--', linewidth=2, 
               label='Theoretical Decay')
        
        # Add simulated points if available
        ax.axhline(ta['mouth_range_m'], color='red', linestyle='-', alpha=0.7,
                  label=f'Simulated Mouth: {ta["mouth_range_m"]:.1f}m')
        ax.axhline(ta['upstream_range_m'], color='orange', linestyle='-', alpha=0.7,
                  label=f'Simulated Upstream: {ta["upstream_range_m"]:.1f}m')
        
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_ylabel('Tidal Range (m)')
        ax.set_title(f'Tidal Amplitude Decay\nValidation: {ta.get("status", "N/A").upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Subplot 3: Flow Reversals
    ax = axes[1, 0]
    if 'tidal_dynamics' in results and 'flow_reversals' in results['tidal_dynamics']:
        fr = results['tidal_dynamics']['flow_reversals']
        
        # Create bar chart showing reversal fraction
        categories = ['Simulated', 'Expected (>80%)', 'Good (>60%)']
        values = [fr['reversal_fraction'] * 100, 80, 60]
        colors = ['red' if fr['status'] == 'warning' else 'green', 'blue', 'orange']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add percentage labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Percentage of Estuary (%)')
        ax.set_title(f'Tidal Flow Reversals\nValidation: {fr.get("status", "N/A").upper()}')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    
    # Subplot 4: Water Quality Validation Summary
    ax = axes[1, 1]
    if 'water_quality' in results:
        wq = results['water_quality']
        
        species_list = list(wq.keys())
        correlations = [wq[sp]['profile_correlation'] for sp in species_list]
        statuses = [wq[sp]['status'] for sp in species_list]
        
        # Color code by status
        colors = ['green' if s == 'excellent' else 'orange' if s == 'good' else 'red' 
                 for s in statuses]
        
        bars = ax.bar(range(len(species_list)), correlations, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(species_list)))
        ax.set_xticklabels(species_list, rotation=45)
        ax.set_ylabel('Profile Correlation')
        ax.set_title('Water Quality Profile Validation')
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5, label='Good Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    figure_file = Path(output_dir) / "physics_validation_comprehensive.png"
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    print(f"📊 Physics validation figure saved: {figure_file}")
    
    plt.show()
    plt.close()
    
    return str(figure_file)

def main():
    """Main function for standalone physics validation."""
    
    print("🔬 STANDALONE ESTUARINE PHYSICS VALIDATION")
    print("=" * 60)
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load model configuration
        from core.config_parser import parse_model_config
        print(f"📋 Loading configuration from: {args.config}")
        model_config = parse_model_config(args.config)
        
        # Load simulation results
        results = load_simulation_results(args.npz_file, args.results_dir)
        
        # Run physics validation
        from core.estuarine_physics_validator import (
            run_estuarine_physics_validation,
            create_physics_validation_summary
        )
        
        print("\n🔬 Running physics validation...")
        validation_report = run_estuarine_physics_validation(results, model_config)
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        report_file = create_physics_validation_summary(validation_report, args.output_dir)
        
        # Create figures if requested
        if args.create_figures:
            try:
                figure_file = create_physics_validation_figures(validation_report, args.output_dir)
                print(f"📊 Validation figures created: {figure_file}")
            except ImportError:
                print("⚠️ Matplotlib not available - skipping figure creation")
            except Exception as e:
                print(f"⚠️ Figure creation failed: {e}")
        
        # Print summary
        overall = validation_report['overall_assessment']
        print(f"\n🏆 VALIDATION COMPLETE")
        print(f"   Status: {overall['status'].upper()}")
        print(f"   Success Rate: {overall['success_rate']:.1%}")
        print(f"   Tests: {overall['excellent']} excellent, {overall['good']} good, {overall['warnings']} warnings")
        print(f"   Report: {report_file}")
        
        # Return appropriate exit code
        if overall['status'] in ['excellent', 'good']:
            print("✅ Physics validation PASSED")
            return 0
        else:
            print("⚠️ Physics validation needs attention")
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available in src/core/")
        return 1
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())