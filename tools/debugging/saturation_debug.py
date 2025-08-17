#!/usr/bin/env python
"""
Debug oxygen saturation calculation in detail.
"""

import numpy as np
import jax.numpy as jnp

def debug_oxygen_saturation():
    """Step-by-step debugging of Garcia & Gordon (1992) O2 saturation."""
    
    print("🫧 Debugging Oxygen Saturation Calculation")
    print("="*50)
    
    temperature = 28.0  # °C
    salinity = 15.0  # PSU
    
    print(f"Temperature: {temperature}°C")
    print(f"Salinity: {salinity} PSU")
    
    # Step 1: Temperature conversion and scaling
    temp_k = temperature + 273.15
    temp_s = np.log((298.15 - temperature) / temp_k)
    
    print(f"\nStep 1: Temperature scaling")
    print(f"T_K = {temp_k:.2f} K")
    print(f"T_s = ln((298.15 - {temperature}) / {temp_k:.2f}) = {temp_s:.6f}")
    
    # Step 2: Garcia & Gordon coefficients
    A0, A1, A2, A3 = 5.80871, 3.20291, 4.17887, 5.10006
    A4, A5 = -9.86643e-2, 3.80369
    B0, B1, B2, B3 = -7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3
    C0 = -2.75915e-7
    
    print(f"\nStep 2: Coefficient application")
    
    # Temperature terms
    temp_terms = A0 + A1 * temp_s + A2 * temp_s**2 + A3 * temp_s**3 + A4 * temp_s**4 + A5 * temp_s**5
    print(f"Temperature terms: {temp_terms:.6f}")
    
    # Salinity terms
    salinity_terms = salinity * (B0 + B1 * temp_s + B2 * temp_s**2 + B3 * temp_s**3)
    print(f"Salinity terms: {salinity_terms:.6f}")
    
    # Salinity squared term
    salinity2_term = C0 * salinity**2
    print(f"Salinity² term: {salinity2_term:.6f}")
    
    # Total
    ln_c_star = temp_terms + salinity_terms + salinity2_term
    print(f"ln(C*) = {ln_c_star:.6f}")
    
    # Step 3: Convert to concentration
    c_star_umol_kg = np.exp(ln_c_star)
    print(f"\nStep 3: Concentration conversion")
    print(f"C* = exp({ln_c_star:.6f}) = {c_star_umol_kg:.2f} μmol/kg")
    
    # Step 4: Unit conversion to mmol/m³
    c_star_mmol_m3 = c_star_umol_kg * 1.025 / 1000.0
    c_star_mg_l = c_star_mmol_m3 * 32.0 / 1000.0
    
    print(f"\nStep 4: Unit conversion")
    print(f"C* = {c_star_umol_kg:.2f} μmol/kg * 1.025/1000 = {c_star_mmol_m3:.6f} mmol/m³")
    print(f"C* = {c_star_mmol_m3:.6f} mmol/m³ * 32/1000 = {c_star_mg_l:.6f} mg/L")
    
    # Compare with expected values
    print(f"\n🎯 Expected vs Actual:")
    print(f"Expected for 28°C, S=15: ~7.0 mg/L (~220 mmol/m³)")
    print(f"Calculated: {c_star_mg_l:.6f} mg/L ({c_star_mmol_m3:.6f} mmol/m³)")
    
    if c_star_mg_l > 5.0:
        print("✅ Saturation value looks reasonable")
    else:
        print("❌ Saturation value is too low!")
        
    # Test model current oxygen vs saturation
    current_o2_mmol = 0.1  # mmol/m³ (current model value)
    current_o2_mg = current_o2_mmol * 32.0 / 1000.0
    deficit = c_star_mmol_m3 - current_o2_mmol
    
    print(f"\n🔍 Saturation deficit analysis:")
    print(f"Current O2: {current_o2_mmol:.3f} mmol/m³ ({current_o2_mg:.3f} mg/L)")
    print(f"Saturation: {c_star_mmol_m3:.3f} mmol/m³ ({c_star_mg_l:.3f} mg/L)")
    print(f"Deficit: {deficit:.3f} mmol/m³")
    print(f"Percent of saturation: {(current_o2_mmol / c_star_mmol_m3) * 100:.1f}%")
    
    if deficit < 0.1:
        print("⚠️ Very small deficit - explains low reaeration rates!")
    else:
        print("✅ Reasonable deficit for reaeration")

def test_alternative_saturation():
    """Test alternative O2 saturation formulation."""
    
    print(f"\n" + "="*50)
    print("🧪 Testing Alternative O2 Saturation")
    print("="*50)
    
    temperature = 28.0
    salinity = 15.0
    
    # Alternative: Benson & Krause (1984) - simpler and widely used
    def benson_krause_o2_sat(temp_c, salinity):
        """Benson & Krause (1984) O2 solubility - often more accurate."""
        # Temperature effect
        temp_k = temp_c + 273.15
        
        # Freshwater solubility (mg/L)
        ln_c_fw = -139.34411 + 1.575701e5/temp_k - 6.642308e7/temp_k**2 + 1.243800e10/temp_k**3 - 8.621949e11/temp_k**4
        c_fw_mg_l = np.exp(ln_c_fw)
        
        # Salinity correction (Weiss 1970)
        salinity_correction = np.exp(-salinity * (0.017674 - 10.754/temp_k + 2140.7/temp_k**2))
        
        c_sat_mg_l = c_fw_mg_l * salinity_correction
        c_sat_mmol_m3 = c_sat_mg_l / 32.0 * 1000.0  # Convert mg/L to mmol/m³
        
        return c_sat_mmol_m3, c_sat_mg_l
    
    alt_mmol, alt_mg = benson_krause_o2_sat(temperature, salinity)
    
    print(f"Alternative (Benson & Krause): {alt_mg:.2f} mg/L ({alt_mmol:.1f} mmol/m³)")
    
    # Simple empirical check
    # At 28°C, S=15, expected ~200-250 μmol/kg = ~6-8 mg/L
    expected_mg_l = 7.0
    expected_mmol_m3 = expected_mg_l / 32.0 * 1000.0
    
    print(f"Expected realistic value: {expected_mg_l:.1f} mg/L ({expected_mmol_m3:.1f} mmol/m³)")
    
if __name__ == "__main__":
    debug_oxygen_saturation()
    test_alternative_saturation()