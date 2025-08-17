#!/usr/bin/env python
"""
Debug and fix the gas transfer velocity calculation.
"""

import numpy as np

def debug_gas_transfer_velocity():
    """Debug the Wanninkhof (1992) gas transfer velocity calculation."""
    
    print("🌬️ Debugging Gas Transfer Velocity Calculation")
    print("="*55)
    
    # Test conditions
    wind_speed = 5.0  # m/s
    temperature = 28.0  # °C
    
    print(f"Wind speed: {wind_speed} m/s")
    print(f"Temperature: {temperature}°C")
    
    # Schmidt number calculation
    schmidt = (1953.4 - 128.0 * temperature + 
               3.9918 * temperature**2 - 
               0.050091 * temperature**3)
    print(f"Schmidt number: {schmidt:.1f}")
    
    # Wanninkhof (1992) calculation - step by step
    print("\n📊 Wanninkhof (1992) calculation:")
    
    # Step 1: Base relationship
    k_wanninkhof_cm_hr = 0.31 * wind_speed**2  # cm/hr
    print(f"1. k_base = 0.31 * {wind_speed}² = {k_wanninkhof_cm_hr:.2f} cm/hr")
    
    # Step 2: Schmidt number correction
    k_corrected_cm_hr = k_wanninkhof_cm_hr * (schmidt / 660.0)**(-0.5)
    print(f"2. k_corrected = {k_wanninkhof_cm_hr:.2f} * ({schmidt}/660)^(-0.5) = {k_corrected_cm_hr:.2f} cm/hr")
    
    # Step 3: Unit conversion to m/s (CURRENT - WRONG)
    k_wrong_m_s = k_corrected_cm_hr * 0.01 / 3600.0  # Current calculation
    print(f"3a. CURRENT (wrong): {k_corrected_cm_hr:.2f} cm/hr * 0.01/3600 = {k_wrong_m_s:.8f} m/s")
    
    # Step 3: Unit conversion to m/day (CORRECT)
    k_correct_m_day = k_corrected_cm_hr * 0.01 * 24.0  # Correct: cm/hr to m/day
    print(f"3b. CORRECT: {k_corrected_cm_hr:.2f} cm/hr * 0.01 * 24 = {k_correct_m_day:.2f} m/day")
    
    # Step 4: Convert m/day to m/s for model use
    k_correct_m_s = k_correct_m_day / 86400.0  # m/day to m/s
    print(f"4. Final: {k_correct_m_day:.2f} m/day / 86400 = {k_correct_m_s:.8f} m/s")
    
    print(f"\n🔍 Comparison:")
    print(f"Current k_gas: {k_wrong_m_s:.8f} m/s")
    print(f"Correct k_gas: {k_correct_m_s:.8f} m/s")
    print(f"Correction factor: {k_correct_m_s / k_wrong_m_s:.1f}x")
    
    # Expected reaeration rates
    depth = 5.0  # m
    saturation_deficit = 0.2  # mmol/m³
    
    current_reaeration = k_wrong_m_s * saturation_deficit / depth  # mmol/m³/s
    correct_reaeration = k_correct_m_s * saturation_deficit / depth  # mmol/m³/s
    
    print(f"\n🔄 Reaeration rate comparison (deficit = {saturation_deficit} mmol/m³, depth = {depth} m):")
    print(f"Current rate: {current_reaeration:.8f} mmol/m³/s = {current_reaeration * 86400:.3f} mmol/m³/day")
    print(f"Correct rate: {correct_reaeration:.8f} mmol/m³/s = {correct_reaeration * 86400:.3f} mmol/m³/day")
    
    # Literature comparison
    print(f"\n📚 Literature comparison:")
    print(f"Expected k_gas for 5 m/s wind: 2-4 m/day")
    print(f"Our corrected k_gas: {k_correct_m_day:.2f} m/day ✅")
    print(f"Expected reaeration rate: 5-20 mmol/m³/day for typical deficit")
    print(f"Our corrected rate: {correct_reaeration * 86400:.1f} mmol/m³/day ✅")

if __name__ == "__main__":
    debug_gas_transfer_velocity()