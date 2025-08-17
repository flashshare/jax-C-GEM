#!/usr/bin/env python3
"""
🔬 Atmospheric Reaeration Debugger
===============================================
Analyzes atmospheric reaeration calculations to identify
why O2 levels are reaching 235-240 mg/L instead of 6-8 mg/L.

Focus: Check saturation values and reaeration rates
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

def compute_garcia_gordon_saturation(temperature, salinity):
    """Test O2 saturation calculation"""
    temp_k = temperature + 273.15
    temp_s = np.log((298.15 - temperature) / temp_k)
    
    # Garcia & Gordon (1992) coefficients
    A0, A1, A2, A3 = 5.80871, 3.20291, 4.17887, 5.10006
    A4, A5 = -9.86643e-2, 3.80369
    B0, B1, B2, B3 = -7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3
    C0 = -2.75915e-7
    
    # Garcia & Gordon equation
    ln_c_star = (A0 + A1 * temp_s + A2 * temp_s**2 + A3 * temp_s**3 + 
                 A4 * temp_s**4 + A5 * temp_s**5 +
                 salinity * (B0 + B1 * temp_s + B2 * temp_s**2 + B3 * temp_s**3) +
                 C0 * salinity**2)
    
    # Current conversion in model
    o2_sat_current = np.exp(ln_c_star) * 1.025  # mmol/m³
    
    # Expected conversion (should be μmol/kg → mmol/m³)
    # μmol/kg * (1.025 kg/L) * (1 L/0.001 m³) * (1 mmol/1000 μmol) = mmol/m³
    o2_sat_correct = np.exp(ln_c_star) * 1.025  # This might be wrong
    
    # Let's check what Garcia & Gordon actually gives in μmol/kg
    o2_sat_micromol_kg = np.exp(ln_c_star)
    
    return o2_sat_micromol_kg, o2_sat_current, o2_sat_correct

def main():
    print("🔬 JAX C-GEM Atmospheric Reaeration Debugger")
    print("=" * 55)
    
    # Test typical estuarine conditions
    temperature = 25.0  # °C
    salinity = 15.0     # PSU
    depth = 5.0         # m
    wind_speed = 5.0    # m/s
    
    print(f"🌊 Test conditions:")
    print(f"   Temperature: {temperature}°C")
    print(f"   Salinity: {salinity} PSU")
    print(f"   Depth: {depth} m")
    print(f"   Wind speed: {wind_speed} m/s")
    
    print(f"\n" + "=" * 55)
    print(f"🔍 OXYGEN SATURATION ANALYSIS")
    print(f"=" * 55)
    
    # Compute saturation
    o2_sat_raw, o2_sat_model, o2_sat_correct = compute_garcia_gordon_saturation(temperature, salinity)
    
    print(f"📊 Garcia & Gordon (1992) output:")
    print(f"   Raw value: {o2_sat_raw:.3f} μmol/kg")
    print(f"   Model conversion: {o2_sat_model:.3f} mmol/m³")
    print(f"   Expected in mg/L: {o2_sat_model * 0.032:.3f} mg/L")
    
    # Check against literature values
    print(f"\n📚 Literature comparison:")
    print(f"   Expected O2 saturation at 25°C, 15 PSU: ~7.5-8.5 mg/L")
    print(f"   Model prediction: {o2_sat_model * 0.032:.3f} mg/L")
    
    if o2_sat_model * 0.032 > 10:
        print(f"   ❌ ISSUE: Saturation value too high!")
        print(f"   Expected conversion: μmol/kg → mg/L should be: raw_value * 0.032")
        print(f"   Alternative approach: {o2_sat_raw * 0.032:.3f} mg/L")
    else:
        print(f"   ✅ Saturation appears reasonable")
    
    print(f"\n" + "=" * 55)
    print(f"🌬️ GAS TRANSFER VELOCITY ANALYSIS")  
    print(f"=" * 55)
    
    # Compute gas transfer velocity
    schmidt = (1953.4 - 128.0 * temperature + 
               3.9918 * temperature**2 - 
               0.050091 * temperature**3)
    schmidt = max(schmidt, 100.0)
    
    k_wanninkhof = 0.31 * wind_speed**2  # cm/hr
    k_corrected = k_wanninkhof * (schmidt / 660.0)**(-0.5)  # cm/hr
    k_gas = k_corrected * 0.01 / 3600.0  # m/s
    
    print(f"📊 Gas transfer calculations:")
    print(f"   Schmidt number: {schmidt:.1f}")
    print(f"   k_wanninkhof: {k_wanninkhof:.6f} cm/hr")
    print(f"   k_corrected: {k_corrected:.6f} cm/hr") 
    print(f"   k_gas: {k_gas:.9f} m/s")
    print(f"   k_gas (daily): {k_gas * 86400:.6f} m/day")
    
    print(f"\n" + "=" * 55)
    print(f"⚖️ REAERATION RATE ANALYSIS")
    print(f"=" * 55)
    
    # Test different O2 concentrations
    current_o2_conc = np.array([50.0, 150.0, 240.0])  # mmol/m³ (different scenarios)
    
    for i, o2_conc in enumerate(current_o2_conc):
        saturation_deficit = o2_sat_model - o2_conc
        reaeration_rate = k_gas * saturation_deficit / depth  # mmol/m³/s
        reaeration_rate_day = reaeration_rate * 86400  # mmol/m³/day
        reaeration_rate_mgL_day = reaeration_rate_day * 0.032  # mg/L/day
        
        print(f"📊 Scenario {i+1}: O2 = {o2_conc:.1f} mmol/m³ ({o2_conc*0.032:.1f} mg/L)")
        print(f"   Saturation deficit: {saturation_deficit:.1f} mmol/m³")
        print(f"   Reaeration rate: {reaeration_rate:.6f} mmol/m³/s")
        print(f"   Reaeration rate: {reaeration_rate_day:.1f} mmol/m³/day")
        print(f"   Reaeration rate: {reaeration_rate_mgL_day:.2f} mg/L/day")
        
        if abs(reaeration_rate_mgL_day) > 10:
            print(f"   ❌ EXCESSIVE reaeration rate!")
        elif abs(reaeration_rate_mgL_day) < 0.1:
            print(f"   ⚠️ Very low reaeration rate")
        else:
            print(f"   ✅ Reasonable reaeration rate")
        print()
    
    print(f"=" * 55)
    print(f"🔧 DIAGNOSTIC SUMMARY")
    print(f"=" * 55)
    
    if o2_sat_model * 0.032 > 15:
        print(f"❌ PRIMARY ISSUE: O2 saturation calculation is too high")
        print(f"   Current: {o2_sat_model * 0.032:.1f} mg/L")
        print(f"   Expected: ~8.5 mg/L at 25°C")
        print(f"   Solution: Check Garcia & Gordon unit conversion")
    elif max(current_o2_conc) * 0.032 > 10 and o2_sat_model * 0.032 < 10:
        print(f"❌ SECONDARY ISSUE: Reaeration is too strong")
        print(f"   O2 building up beyond saturation")
        print(f"   Solution: Reduce k_gas or limit O2 to saturation")
    else:
        print(f"✅ Reaeration calculations appear reasonable")
    
    print(f"\n✅ Reaeration debugging complete!")

if __name__ == "__main__":
    main()