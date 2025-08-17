#!/usr/bin/env python
"""
Oxygen Mass Balance Debugger for JAX C-GEM

This script provides comprehensive debugging of the dissolved oxygen system
to diagnose the critical failure causing 0.003 mg/L throughout the estuary.

Key Features:
- Oxygen production/consumption tracking
- Mass balance verification
- Parameter validation against literature
- Atmospheric reaeration analysis
- Species coupling verification

Author: Phase I Critical Fix Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path
import sys
import warnings
from typing import Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.biogeochemistry import biogeochemical_step, create_biogeo_params
    from core.model_config import SPECIES_NAMES, DEFAULT_BIO_PARAMS
    from core.config_parser import parse_model_config
    from core.data_loader import DataLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    exit(1)

warnings.filterwarnings('ignore')

class OxygenDebugger:
    """Comprehensive oxygen system debugger."""
    
    def __init__(self, results_file="OUT/complete_simulation_results.npz"):
        self.results_file = results_file
        self.results = None
        self.o2_index = 7  # O2 is at index 7 in SPECIES_NAMES
        self.load_results()
        
    def load_results(self):
        """Load simulation results."""
        if Path(self.results_file).exists():
            print(f"📊 Loading results from {self.results_file}")
            self.results = np.load(self.results_file)
            print(f"   Available keys: {list(self.results.keys())}")
        else:
            print(f"❌ Results file not found: {self.results_file}")
            print("   Please run a simulation first with: python src/main.py")
            
    def analyze_oxygen_levels(self):
        """Analyze current oxygen levels and identify problems."""
        print("\n" + "="*60)
        print("🔍 OXYGEN LEVEL ANALYSIS")
        print("="*60)
        
        if self.results is None:
            print("❌ No results available for analysis")
            return
        
        # Extract oxygen data
        o2_data = self.results['O2']  # Shape: (time_steps, grid_points)
        time_data = self.results['time']
        
        print(f"📊 Oxygen data shape: {o2_data.shape}")
        print(f"⏱️  Time steps: {len(time_data)}")
        
        # Convert from mmol/m³ to mg/L
        o2_mg_per_l = o2_data * 32.0 / 1000.0
        
        # Statistics
        o2_mean = np.mean(o2_mg_per_l)
        o2_min = np.min(o2_mg_per_l)
        o2_max = np.max(o2_mg_per_l)
        o2_std = np.std(o2_mg_per_l)
        
        print(f"🧪 Current Oxygen Levels:")
        print(f"   Mean: {o2_mean:.6f} mg/L")
        print(f"   Min:  {o2_min:.6f} mg/L") 
        print(f"   Max:  {o2_max:.6f} mg/L")
        print(f"   Std:  {o2_std:.6f} mg/L")
        
        # Expected levels for comparison
        expected_min = 4.0  # mg/L
        expected_max = 8.0  # mg/L
        
        print(f"\n🎯 Expected Oxygen Levels:")
        print(f"   Typical range: {expected_min}-{expected_max} mg/L")
        print(f"   Upstream expected: 6-8 mg/L")
        print(f"   Downstream expected: 4-6 mg/L")
        
        # Problem diagnosis
        if o2_max < 1.0:
            print(f"\n🚨 CRITICAL FAILURE: Maximum O2 ({o2_max:.6f} mg/L) is {expected_min/o2_max:.0f}x too low!")
            print("   This indicates complete system collapse")
        elif o2_mean < expected_min:
            print(f"\n⚠️  WARNING: Mean O2 ({o2_mean:.3f} mg/L) is below expected minimum")
        else:
            print(f"\n✅ Oxygen levels appear reasonable")
            
        return o2_data, o2_mg_per_l
    
    def check_mass_balance(self):
        """Check oxygen mass balance over the simulation."""
        print("\n" + "="*60)
        print("⚖️  OXYGEN MASS BALANCE CHECK")
        print("="*60)
        
        if self.results is None:
            return
            
        o2_data = self.results['O2']
        
        # Calculate total oxygen in system over time
        total_o2_time = np.sum(o2_data, axis=1)  # Sum over space for each time
        
        # Check for monotonic decrease (mass loss)
        initial_total = total_o2_time[0]
        final_total = total_o2_time[-1] 
        total_loss = initial_total - final_total
        percent_loss = (total_loss / initial_total) * 100
        
        print(f"💧 Initial total O2: {initial_total:.2f} mmol")
        print(f"💧 Final total O2:   {final_total:.2f} mmol") 
        print(f"📉 Total loss:       {total_loss:.2f} mmol ({percent_loss:.1f}%)")
        
        if percent_loss > 10:
            print("🚨 CRITICAL: >10% oxygen loss indicates mass balance failure")
        elif percent_loss > 5:
            print("⚠️  WARNING: >5% oxygen loss may indicate issues")
        else:
            print("✅ Mass balance appears reasonable")
            
        # Plot mass balance over time
        plt.figure(figsize=(10, 6))
        plt.plot(total_o2_time)
        plt.title('Total Oxygen in System Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Total O2 [mmol]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('OUT/oxygen_mass_balance.png', dpi=300, bbox_inches='tight')
        print("📊 Mass balance plot saved: OUT/oxygen_mass_balance.png")
        
        return total_o2_time
    
    def validate_biogeochemical_parameters(self):
        """Validate biogeochemical parameters against literature."""
        print("\n" + "="*60)
        print("📚 BIOGEOCHEMICAL PARAMETER VALIDATION")
        print("="*60)
        
        # Load current parameters
        try:
            config = parse_model_config("config/model_config.txt")
            biogeo_params = create_biogeo_params(config)
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            return
            
        # Literature values for comparison (from Volta et al. 2016, Savenije 2012)
        literature_ranges = {
            'mumax_phy1': (0.5, 3.0),      # day⁻¹ - diatom max growth rate
            'mumax_phy2': (0.5, 3.0),      # day⁻¹ - non-diatom max growth rate
            'alpha': (1.0, 8.0),           # μE⁻¹ m² s day⁻¹ - photosynthetic efficiency
            'resp': (0.01, 0.1),           # day⁻¹ - respiration rate
            'mort': (0.01, 0.1),           # day⁻¹ - mortality rate
            'knit': (0.05, 0.2),           # day⁻¹ - nitrification rate
            'q10_growth': (1.5, 3.0),      # Q10 for growth
            'q10_resp': (1.5, 3.0),        # Q10 for respiration
        }
        
        print("🔍 Parameter validation:")
        issues_found = 0
        
        for param, (min_val, max_val) in literature_ranges.items():
            if param in biogeo_params:
                current_val = biogeo_params[param]
                if current_val < min_val or current_val > max_val:
                    print(f"   ❌ {param}: {current_val:.3f} (literature: {min_val}-{max_val})")
                    issues_found += 1
                else:
                    print(f"   ✅ {param}: {current_val:.3f} (within {min_val}-{max_val})")
            else:
                print(f"   ⚠️  {param}: Not found in current parameters")
                issues_found += 1
                
        if issues_found == 0:
            print("\n✅ All parameters within literature ranges")
        else:
            print(f"\n⚠️  {issues_found} parameter issues found")
            
        return biogeo_params
    
    def check_atmospheric_reaeration(self):
        """Check if atmospheric reaeration is implemented."""
        print("\n" + "="*60) 
        print("🌬️  ATMOSPHERIC REAERATION CHECK")
        print("="*60)
        
        # This requires examining the biogeochemistry code
        print("🔍 Checking biogeochemistry.py for reaeration terms...")
        
        bio_file = Path("src/core/biogeochemistry.py")
        if bio_file.exists():
            with open(bio_file, 'r') as f:
                content = f.read()
                
            # Look for reaeration-related terms
            reaeration_terms = ['reaeration', 'k_o2', 'k_rea', 'atmospheric', 'air_sea', 'oxygen_exchange']
            found_terms = [term for term in reaeration_terms if term.lower() in content.lower()]
            
            if found_terms:
                print(f"✅ Found reaeration terms: {found_terms}")
            else:
                print("❌ No reaeration terms found in biogeochemistry.py")
                print("   This may be the primary cause of oxygen depletion!")
                
            # Look for O'Connor-Dobbins or similar models
            models = ['o_connor', 'dobbins', 'wind_speed', 'schmidt']
            found_models = [model for model in models if model.lower() in content.lower()]
            
            if found_models:
                print(f"✅ Found reaeration models: {found_models}")
            else:
                print("❌ No standard reaeration models found")
                
        else:
            print("❌ biogeochemistry.py not found")
    
    def analyze_species_coupling(self):
        """Analyze oxygen coupling with other species."""
        print("\n" + "="*60)
        print("🔗 SPECIES COUPLING ANALYSIS") 
        print("="*60)
        
        if self.results is None:
            return
            
        # Key species indices
        species_indices = {
            'O2': 7,   'NH4': 4,  'NO3': 3,
            'PHY1': 0, 'PHY2': 1, 'TOC': 8
        }
        
        # Extract time-averaged profiles
        warmup_steps = 2400  # Skip warmup period
        
        correlations = {}
        for species, idx in species_indices.items():
            if species == 'O2':
                continue
                
            species_data = self.results[species][warmup_steps:]
            o2_data = self.results['O2'][warmup_steps:]
            
            # Calculate correlation
            correlation = np.corrcoef(species_data.flatten(), o2_data.flatten())[0,1]
            correlations[species] = correlation
            
            print(f"🔗 O2 vs {species}: r = {correlation:.3f}")
            
        # Expected relationships:
        print("\n🎯 Expected correlations:")
        print("   O2 vs NH4: negative (NH4 oxidation consumes O2)")
        print("   O2 vs NO3: positive (nitrification produces NO3, consumes O2)")
        print("   O2 vs PHY: positive (photosynthesis produces O2)")
        print("   O2 vs TOC: negative (decomposition consumes O2)")
        
        return correlations
    
    def create_diagnostic_report(self):
        """Create comprehensive diagnostic report."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE OXYGEN DIAGNOSTIC REPORT")
        print("="*80)
        
        # Run all analyses
        o2_data, o2_mg_per_l = self.analyze_oxygen_levels()
        mass_balance = self.check_mass_balance()
        parameters = self.validate_biogeochemical_parameters()
        self.check_atmospheric_reaeration()
        correlations = self.analyze_species_coupling()
        
        # Generate summary
        print("\n" + "="*60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        if o2_data is not None:
            o2_mean = np.mean(o2_mg_per_l)
            if o2_mean < 1.0:
                print("🚨 CRITICAL FAILURE: Oxygen system has collapsed")
                print("   Primary suspects:")
                print("   1. Missing atmospheric reaeration")
                print("   2. Excessive respiration rates") 
                print("   3. Mass balance errors")
                print("   4. Missing photosynthetic oxygen production")
            elif o2_mean < 4.0:
                print("⚠️  WARNING: Oxygen levels below expected range")
            else:
                print("✅ Oxygen levels appear reasonable")
        
        # Save diagnostic data
        diagnostic_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'oxygen_stats': {
                'mean_mg_per_l': float(np.mean(o2_mg_per_l)) if o2_data is not None else 0,
                'min_mg_per_l': float(np.min(o2_mg_per_l)) if o2_data is not None else 0,
                'max_mg_per_l': float(np.max(o2_mg_per_l)) if o2_data is not None else 0,
            },
            'correlations': correlations if correlations else {},
            'parameters_validated': parameters is not None
        }
        
        # Save as JSON for further analysis
        import json
        with open('OUT/oxygen_diagnosis_report.json', 'w') as f:
            json.dump(diagnostic_data, f, indent=2)
            
        print("\n📄 Diagnostic report saved: OUT/oxygen_diagnosis_report.json")
        
        return diagnostic_data

def main():
    """Main debugging workflow."""
    print("🔬 JAX C-GEM Oxygen Mass Balance Debugger")
    print("="*50)
    
    # Create output directory
    Path("OUT").mkdir(exist_ok=True)
    
    # Initialize debugger
    debugger = OxygenDebugger()
    
    # Run comprehensive analysis
    diagnostic_data = debugger.create_diagnostic_report()
    
    print("\n✅ Oxygen debugging complete!")
    print(f"📊 Results saved in OUT/ directory")
    
    return diagnostic_data

if __name__ == "__main__":
    main()