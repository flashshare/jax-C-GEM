#!/usr/bin/env python
"""
Hydrodynamic System Diagnostic Tool

This tool investigates the root cause of the systematic 2x tidal range 
over-prediction that persists despite friction and amplitude adjustments.

Potential issues to investigate:
1. Domain resonance effects
2. Channel geometry amplification
3. Numerical stability issues
4. Boundary condition implementation
5. Grid resolution effects

Author: Phase II Hydrodynamic Analysis Team
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def analyze_domain_resonance():
    """Analyze potential tidal resonance in the model domain."""
    
    print("🌊 Domain Resonance Analysis")
    print("=" * 35)
    
    # Model domain parameters
    EL = 202000  # Total length [m]
    AMPL = 4.43  # Boundary tidal amplitude [m] 
    g = 9.81     # Gravity [m/s²]
    
    # Typical estuarine depths
    depths = [10, 15, 20, 25]  # [m]
    
    print(f"Domain length: {EL/1000:.1f} km")
    print(f"Boundary amplitude: {AMPL:.2f} m")
    print()
    
    print("Tidal Wave Characteristics:")
    print("Depth [m]  Wave Speed [m/s]  Period [hr]  Wavelength [km]  Resonance Factor")
    print("-" * 75)
    
    resonance_factors = []
    
    for depth in depths:
        # Shallow water wave speed
        c = np.sqrt(g * depth)
        
        # M2 tidal period (12.42 hours)
        T_m2 = 12.42 * 3600  # Convert to seconds
        
        # Wavelength
        wavelength = c * T_m2
        
        # Resonance factor: Domain length / quarter wavelength
        # Values near 1, 3, 5, etc. indicate resonance
        resonance_factor = EL / (wavelength / 4)
        resonance_factors.append(resonance_factor)
        
        print(f"{depth:5.0f}    {c:7.1f}         {T_m2/3600:5.2f}    {wavelength/1000:7.0f}       {resonance_factor:6.2f}")
    
    print()
    
    # Check for resonance conditions
    problematic_depths = []
    for i, (depth, rf) in enumerate(zip(depths, resonance_factors)):
        if abs(rf - round(rf)) < 0.2:  # Close to integer (resonance condition)
            problematic_depths.append((depth, rf))
            
    if problematic_depths:
        print("⚠️  POTENTIAL RESONANCE DETECTED:")
        for depth, rf in problematic_depths:
            print(f"   Depth {depth}m: Resonance factor = {rf:.2f} (close to {round(rf)})")
        print("   This could cause tidal amplification!")
    else:
        print("✅ No obvious resonance conditions detected")
        
    return resonance_factors

def analyze_channel_geometry():
    """Analyze channel geometry for tidal amplification effects."""
    
    print("\n🏞️  Channel Geometry Analysis")  
    print("=" * 35)
    
    # Current geometry parameters
    B1 = 3887.0   # Width at mouth [m]
    B2 = 450.0    # Width at inflection [m] 
    LC1 = 50000.0 # Convergence length 1 [m]
    LC2 = 40000.0 # Convergence length 2 [m]
    
    print("Current Channel Geometry:")
    print(f"  Mouth width (B1): {B1:.0f} m")
    print(f"  Upstream width (B2): {B2:.0f} m")  
    print(f"  Width ratio (B1/B2): {B1/B2:.1f}x")
    print(f"  Convergence length 1: {LC1/1000:.0f} km")
    print(f"  Convergence length 2: {LC2/1000:.0f} km")
    
    # Calculate funnel effect
    funnel_ratio = B1 / B2
    print(f"\n📐 Funnel Effect Analysis:")
    print(f"  Width convergence ratio: {funnel_ratio:.1f}x")
    
    if funnel_ratio > 6:
        print("  ⚠️  STRONG CONVERGENCE: May cause significant tidal amplification")
    elif funnel_ratio > 4:  
        print("  ⚠️  MODERATE CONVERGENCE: Some tidal amplification expected")
    else:
        print("  ✅ MILD CONVERGENCE: Limited amplification expected")
        
    # Theoretical tidal amplification due to convergence
    # Green's law: A ∝ (B₀/B)^(1/4) * (h₀/h)^(1/4)
    theoretical_amplification = funnel_ratio ** 0.25
    print(f"  Theoretical amplification (Green's law): {theoretical_amplification:.2f}x")
    
    return funnel_ratio, theoretical_amplification

def analyze_model_results():
    """Analyze model results for diagnostic insights."""
    
    print("\n📊 Model Results Analysis")
    print("=" * 35)
    
    try:
        # Load results
        results = np.load("OUT/complete_simulation_results.npz")
        H = results['H']  # Water levels
        U = results['U']  # Velocities
        
        print(f"Results loaded: {H.shape[0]} time steps, {H.shape[1]} grid points")
        
        # Calculate spatial tidal amplitude profile
        tidal_amplitudes = []
        grid_points = H.shape[1]
        
        for i in range(grid_points):
            h_series = H[:, i]
            amplitude = (np.max(h_series) - np.min(h_series)) / 2
            tidal_amplitudes.append(amplitude)
            
        tidal_amplitudes = np.array(tidal_amplitudes)
        
        # Distance grid
        dx = 2000  # Grid spacing [m]
        distances = np.arange(grid_points) * dx / 1000  # Convert to km
        
        # Find maximum amplification
        boundary_amplitude = tidal_amplitudes[0]  # Amplitude at boundary
        max_amplitude = np.max(tidal_amplitudes)
        max_location = distances[np.argmax(tidal_amplitudes)]
        amplification_factor = max_amplitude / boundary_amplitude
        
        print(f"Boundary tidal amplitude: {boundary_amplitude:.2f} m")
        print(f"Maximum tidal amplitude: {max_amplitude:.2f} m")
        print(f"Maximum location: {max_location:.1f} km from mouth") 
        print(f"Amplification factor: {amplification_factor:.2f}x")
        
        if amplification_factor > 1.5:
            print(f"⚠️  HIGH AMPLIFICATION DETECTED: {amplification_factor:.2f}x")
            print("   This could explain the 2x tidal range over-prediction")
        else:
            print(f"✅ Reasonable amplification: {amplification_factor:.2f}x")
            
        # Plot tidal amplitude profile
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(distances, tidal_amplitudes, 'b-', linewidth=2)
        plt.axhline(y=boundary_amplitude, color='r', linestyle='--', 
                   label=f'Boundary: {boundary_amplitude:.2f}m')
        plt.axhline(y=max_amplitude, color='g', linestyle='--',
                   label=f'Maximum: {max_amplitude:.2f}m')
        plt.xlabel('Distance from mouth (km)')
        plt.ylabel('Tidal amplitude (m)')
        plt.title('Tidal Amplitude Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Velocity amplitude profile
        velocity_amplitudes = []
        for i in range(grid_points):
            u_series = U[:, i]
            vel_amplitude = (np.max(u_series) - np.min(u_series)) / 2
            velocity_amplitudes.append(vel_amplitude)
            
        velocity_amplitudes = np.array(velocity_amplitudes)
        
        plt.subplot(1, 2, 2) 
        plt.plot(distances, velocity_amplitudes, 'r-', linewidth=2)
        plt.xlabel('Distance from mouth (km)')
        plt.ylabel('Velocity amplitude (m/s)')
        plt.title('Velocity Amplitude Profile')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('OUT/hydrodynamic_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Diagnostic plots saved: OUT/hydrodynamic_diagnostics.png")
        
        return amplification_factor
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None

def main():
    """Run comprehensive hydrodynamic diagnostic analysis."""
    
    print("🔍 JAX C-GEM Hydrodynamic System Diagnostics")
    print("=" * 50)
    
    # 1. Check domain resonance
    resonance_factors = analyze_domain_resonance()
    
    # 2. Analyze channel geometry  
    funnel_ratio, theoretical_amp = analyze_channel_geometry()
    
    # 3. Analyze model results
    observed_amp = analyze_model_results()
    
    # 4. Summary and recommendations
    print("\n🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    print("Issues Identified:")
    
    # Check convergence amplification
    if funnel_ratio > 6:
        print("1. ⚠️  EXCESSIVE CHANNEL CONVERGENCE")
        print(f"   Width ratio: {funnel_ratio:.1f}x")
        print(f"   Theoretical amplification: {theoretical_amp:.2f}x")
        print("   → RECOMMENDATION: Reduce width convergence (increase B2 or decrease LC1/LC2)")
        
    # Check model amplification
    if observed_amp and observed_amp > 1.5:
        print("2. ⚠️  HIGH MODEL TIDAL AMPLIFICATION")
        print(f"   Observed amplification: {observed_amp:.2f}x")
        print("   → RECOMMENDATION: Investigate numerical schemes or adjust geometry")
        
    print("\nNext Steps for Phase II:")
    print("1. Test reduced channel convergence (increase B2 from 450m to 800-1000m)")
    print("2. Test shorter convergence lengths (reduce LC1/LC2 by 20-30%)")
    print("3. Verify numerical stability of hydrodynamic scheme")
    print("4. Compare with original C-GEM geometry parameters")
    
    print("\n✅ Hydrodynamic diagnostics completed!")

if __name__ == "__main__":
    main()