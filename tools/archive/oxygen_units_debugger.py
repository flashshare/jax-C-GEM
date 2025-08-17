#!/usr/bin/env python3
"""
🔬 Oxygen Units Debugging Tool
===============================================
Diagnoses unit conversion issues in oxygen calculations.
Checks internal units vs output units to identify discrepancies.

Focus: Resolve O2 values showing 211-240 mg/L instead of expected 6-8 mg/L
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

def main():
    print("🔬 JAX C-GEM Oxygen Units Debugger")
    print("=" * 50)
    
    # Load NPZ results (internal format)
    npz_file = Path("OUT/complete_simulation_results.npz")
    if npz_file.exists():
        print("📊 Loading NPZ results...")
        data_npz = np.load(npz_file)
        o2_npz = data_npz['O2']  # Internal units
        print(f"   NPZ O2 shape: {o2_npz.shape}")
        print(f"   NPZ O2 range: {o2_npz.min():.6f} - {o2_npz.max():.6f}")
        print(f"   NPZ O2 mean: {o2_npz.mean():.6f}")
        print(f"   NPZ O2 final values (last 5 grid points): {o2_npz[-1, -5:]}")
    else:
        print("❌ NPZ file not found")
        o2_npz = None
    
    # Load CSV results (output format)
    csv_file = Path("OUT/Reaction/O2.csv")
    if csv_file.exists():
        print("\n📊 Loading CSV results...")
        # Read last few lines for comparison
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            last_line = lines[-1].strip()
            values = [float(x) for x in last_line.split(',')[1:]]  # Skip time
            print(f"   CSV O2 count: {len(values)} values")
            print(f"   CSV O2 range: {min(values):.6f} - {max(values):.6f}")
            print(f"   CSV O2 mean: {np.mean(values):.6f}")
            print(f"   CSV O2 final values (last 5 grid points): {values[-5:]}")
        else:
            print("❌ CSV file is empty")
            values = None
    else:
        print("❌ CSV file not found")
        values = None
    
    print("\n" + "=" * 60)
    print("🔍 UNIT CONVERSION ANALYSIS")
    print("=" * 60)
    
    if o2_npz is not None and values is not None:
        # Compare internal vs output values
        npz_final = o2_npz[-1, :]
        csv_final = np.array(values)
        
        print(f"📊 Final time step comparison:")
        print(f"   NPZ final mean: {npz_final.mean():.6f}")
        print(f"   CSV final mean: {csv_final.mean():.6f}")
        
        # Calculate conversion ratio
        ratio = csv_final.mean() / npz_final.mean()
        print(f"   Ratio (CSV/NPZ): {ratio:.6f}")
        
        # Check if this matches any known unit conversions
        print(f"\n🔍 Potential unit conversion factors:")
        print(f"   mg/L to mmol/m³ (O2): ~31.25")
        print(f"   mmol/m³ to mg/L (O2): ~0.032")
        print(f"   Observed ratio: {ratio:.6f}")
        
        if abs(ratio - 31.25) < 1.0:
            print("   ✅ Likely issue: CSV values are in wrong units (mmol/m³ instead of mg/L)")
        elif abs(ratio - 1.0) < 0.1:
            print("   ✅ Units appear consistent")
        else:
            print(f"   ⚠️ Unexpected conversion ratio: {ratio:.6f}")
        
        # Plot comparison
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # NPZ values
        ax1.plot(npz_final, 'b-', label='NPZ Internal')
        ax1.set_ylabel('NPZ O2 (internal units)')
        ax1.set_title('Internal NPZ Values')
        ax1.grid(True, alpha=0.3)
        
        # CSV values
        ax2.plot(csv_final, 'r-', label='CSV Output')
        ax2.set_ylabel('CSV O2 (output units)')
        ax2.set_title('CSV Output Values')
        ax2.grid(True, alpha=0.3)
        
        # Ratio
        ax3.plot(csv_final / npz_final, 'g-', label='Ratio CSV/NPZ')
        ax3.set_ylabel('Conversion Ratio')
        ax3.set_xlabel('Grid Point')
        ax3.set_title('Unit Conversion Ratio')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('OUT/oxygen_units_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Analysis plot saved: OUT/oxygen_units_analysis.png")
    
    print("\n" + "=" * 60)
    print("🔧 RECOMMENDATIONS")
    print("=" * 60)
    
    if o2_npz is not None and values is not None:
        if ratio > 30:
            print("🎯 LIKELY ISSUE: CSV output is in mmol/m³ but should be mg/L")
            print("   Solution: Check CSV writer unit conversion in result_writer.py")
            print("   Expected: O2_mg_L = O2_mmol_m3 * 0.032")
        elif ratio > 20:
            print("🎯 POSSIBLE ISSUE: Incorrect molecular weight conversion")
            print("   Solution: Check O2 molecular weight (32 g/mol) in calculations")
        else:
            print("🎯 Units appear reasonable - investigate other causes")
    
    # Check biogeochemistry.py for unit issues
    print(f"\n🔍 Checking biogeochemistry.py for unit conversions...")
    bio_file = Path("src/core/biogeochemistry.py")
    if bio_file.exists():
        with open(bio_file, 'r') as f:
            content = f.read()
        
        # Look for O2 unit conversions
        if '0.032' in content:
            print("   ✅ Found mmol/m³ → mg/L conversion factor (0.032)")
        if '31.25' in content:
            print("   ⚠️ Found mg/L → mmol/m³ conversion factor (31.25)")
        if 'mg/L' in content and 'mmol' in content:
            print("   ✅ Found mixed unit handling in biogeochemistry")
    
    print(f"\n✅ Unit debugging complete!")
    print(f"📁 Results saved in OUT/ directory")

if __name__ == "__main__":
    main()