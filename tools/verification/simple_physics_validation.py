#!/usr/bin/env python3
"""
Simple Physics Validation - Salinity Gradient Analysis
======================================================

Quick validation of the key physics issues:
1. Salinity gradient direction 
2. Tidal amplitude
3. Basic transport validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_salinity_gradient():
    """Analyze salinity gradient and create diagnostic plots."""
    
    # Load results
    try:
        S = pd.read_csv("OUT/Reaction/S.csv")
        H = pd.read_csv("OUT/Hydrodynamics/H.csv") 
        U = pd.read_csv("OUT/Hydrodynamics/U.csv")
        
        print(f"✅ Loaded data: {S.shape[0]} timesteps, {S.shape[1]-1} grid points")
        
        # Extract data arrays (skip time column)
        salinity = S.iloc[:, 1:].values  # [time, space]
        water_level = H.iloc[:, 1:].values
        velocity = U.iloc[:, 1:].values
        time_hours = S.iloc[:, 0].values / 3600.0  # Convert to hours
        
        # Create distance array (assume 160km estuary)
        distance_km = np.linspace(0, 160, salinity.shape[1])
        
        # Final salinity profile
        final_salinity = salinity[-1, :]
        
        print("\n🧂 SALINITY GRADIENT ANALYSIS:")
        print(f"   Mouth (0 km): {final_salinity[0]:.3f} PSU")
        print(f"   Head (160 km): {final_salinity[-1]:.3f} PSU") 
        print(f"   Mid-estuary: {final_salinity[len(final_salinity)//2]:.3f} PSU")
        
        gradient_correct = final_salinity[0] > final_salinity[-1]
        print(f"   Gradient: {'CORRECT ✅' if gradient_correct else 'INVERTED ❌'}")
        
        if not gradient_correct:
            print("\n❌ SALINITY INVERSION DETECTED!")
            print("   Expected: Mouth (30 PSU) → Head (0.01 PSU)")
            print(f"   Actual:   Mouth ({final_salinity[0]:.3f} PSU) → Head ({final_salinity[-1]:.3f} PSU)")
            
        # Create diagnostic plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Salinity profile
        ax1 = axes[0, 0]
        ax1.plot(distance_km, final_salinity, 'g-', linewidth=3, label='JAX C-GEM Final Salinity')
        ax1.axhline(y=30, color='blue', linestyle='--', alpha=0.7, label='Expected Ocean (30 PSU)')
        ax1.axhline(y=0.01, color='brown', linestyle='--', alpha=0.7, label='Expected River (0.01 PSU)')
        ax1.set_xlabel('Distance from Mouth (km)')
        ax1.set_ylabel('Salinity (PSU)')
        ax1.set_title(f'🧂 Salinity Profile - {"CORRECT ✅" if gradient_correct else "INVERTED ❌"}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Tidal amplitude
        ax2 = axes[0, 1]
        tidal_amplitude = (np.max(water_level, axis=0) - np.min(water_level, axis=0)) / 2.0
        ax2.plot(distance_km, tidal_amplitude, 'b-', linewidth=3, label='Tidal Amplitude')
        ax2.set_xlabel('Distance from Mouth (km)')
        ax2.set_ylabel('Tidal Amplitude (m)')
        ax2.set_title('🌊 Tidal Amplitude Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Velocity profile
        ax3 = axes[1, 0]
        mean_velocity = np.mean(velocity, axis=0)
        max_velocity = np.max(velocity, axis=0)
        min_velocity = np.min(velocity, axis=0)
        ax3.plot(distance_km, mean_velocity, 'purple', linewidth=2, label='Mean Velocity')
        ax3.fill_between(distance_km, min_velocity, max_velocity, alpha=0.3, color='purple', label='Velocity Range')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Distance from Mouth (km)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('🌊 Velocity Profile')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Salinity time series at key locations
        ax4 = axes[1, 1]
        mouth_idx = 0
        mid_idx = len(distance_km) // 2
        head_idx = -1
        
        ax4.plot(time_hours[:100], salinity[:100, mouth_idx], 'b-', label='Mouth (0 km)', linewidth=2)
        ax4.plot(time_hours[:100], salinity[:100, mid_idx], 'g-', label=f'Mid ({distance_km[mid_idx]:.0f} km)', linewidth=2)
        ax4.plot(time_hours[:100], salinity[:100, head_idx], 'r-', label='Head (160 km)', linewidth=2)
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Salinity (PSU)')
        ax4.set_title('🧂 Salinity Time Series (First 100 timesteps)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add overall title
        status = "TRANSPORT PHYSICS VALIDATION" 
        color = "red" if not gradient_correct else "green"
        fig.suptitle(f'{status} - {"SALINITY INVERTED ❌" if not gradient_correct else "SALINITY CORRECT ✅"}', 
                     fontsize=16, fontweight='bold', color=color)
                    
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("OUT/Validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "physics_validation_simple.png", dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved validation plot: {output_dir / 'physics_validation_simple.png'}")
        
        plt.show()
        
        return gradient_correct
        
    except Exception as e:
        print(f"❌ Error in validation: {e}")
        return False

def compare_with_cgem_expectations():
    """Compare results with expected C-GEM behavior."""
    
    print("\n🔬 C-GEM PHYSICS EXPECTATION ANALYSIS:")
    print("=" * 50)
    
    try:
        # Check if original C-GEM results exist for comparison
        original_results = Path("deprecated/original-C-GEM/OUT")
        if original_results.exists():
            print("✅ Original C-GEM results found - detailed comparison possible")
        else:
            print("⚠️  Original C-GEM results not found - using expected values")
            
        print("\n📊 EXPECTED C-GEM BEHAVIOR:")
        print("   🌊 Tidal amplitude: 1-3m decreasing landward")
        print("   🧂 Salinity: 25-30 PSU (ocean) → 0.01-0.1 PSU (river)")
        print("   💨 Velocity: ±1-3 m/s with tidal reversals")
        print("   ⚗️  Dispersion: 10-300 m²/s increasing seaward")
        
        print("\n🎯 KEY DIAGNOSTIC QUESTIONS:")
        print("   1. Are boundary conditions correctly applied?")
        print("   2. Is the velocity-dependent logic working?")  
        print("   3. Is the centered difference scheme correct?")
        print("   4. Are coordinate systems properly mapped?")
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

def main():
    """Main validation."""
    print("🔬 JAX C-GEM Simple Physics Validation")
    print("=" * 50)
    
    # Analyze salinity gradient (core issue)
    gradient_correct = analyze_salinity_gradient()
    
    # Compare with expectations
    compare_with_cgem_expectations()
    
    # Summary
    print(f"\n{'='*50}")
    if gradient_correct:
        print("✅ PHYSICS VALIDATION PASSED")
        print("   Salinity gradient is correct")
    else:
        print("❌ PHYSICS VALIDATION FAILED")
        print("   Salinity gradient is INVERTED")
        print("   C-GEM transport implementation needs further debugging")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()