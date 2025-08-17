#!/usr/bin/env python
"""
CORRECTED oxygen saturation using proper Garcia & Gordon (1992) implementation.
"""
import numpy as np

def garcia_gordon_corrected(temp_c, salinity):
    """
    CORRECTED Garcia & Gordon (1992) O2 solubility.
    
    The issue was in the coefficient application and units.
    """
    # Convert temperature to Kelvin and create scaled temperature
    temp_k = temp_c + 273.15
    temp_s = np.log((298.15 - temp_c) / temp_k)  # Correct Garcia & Gordon scaling
    
    # Garcia & Gordon (1992) coefficients - Table I
    A0 = 5.80871
    A1 = 3.20291
    A2 = 4.17887
    A3 = 5.10006
    A4 = -9.86643e-2
    A5 = 3.80369
    
    B0 = -7.01577e-3
    B1 = -7.70028e-3
    B2 = -1.13864e-2  
    B3 = -9.51519e-3
    
    C0 = -2.75915e-7
    
    # Garcia & Gordon equation (1992) - Equation 8
    ln_c_star = (A0 + A1 * temp_s + A2 * temp_s**2 + A3 * temp_s**3 + 
                 A4 * temp_s**4 + A5 * temp_s**5 +
                 salinity * (B0 + B1 * temp_s + B2 * temp_s**2 + B3 * temp_s**3) +
                 C0 * salinity**2)
    
    # Result in μmol/kg-sw
    c_star_umol_kg = np.exp(ln_c_star)
    
    # Convert to mmol/m³ 
    # Density of seawater ≈ 1.025 kg/L = 1025 kg/m³
    c_star_mmol_m3 = c_star_umol_kg * 1.025 / 1000.0
    
    return c_star_mmol_m3, c_star_umol_kg

def simplified_oxygen_saturation(temp_c, salinity):
    """
    Simplified but accurate oxygen saturation for estuarine conditions.
    Based on empirical fits to Garcia & Gordon data.
    """
    # Temperature effect (exponential decrease with warming)
    temp_factor = np.exp(0.046 * (20.0 - temp_c))
    
    # Salinity effect (linear decrease with salinity)
    salinity_factor = 1.0 - 0.014 * salinity
    
    # Reference O2 saturation at 20°C, S=0: ~280 μmol/kg
    o2_sat_ref = 280.0  # μmol/kg
    
    # Apply corrections
    o2_sat_umol_kg = o2_sat_ref * temp_factor * salinity_factor
    
    # Convert to mmol/m³
    o2_sat_mmol_m3 = o2_sat_umol_kg * 1.025 / 1000.0
    
    return o2_sat_mmol_m3, o2_sat_umol_kg

# Test both methods
test_conditions = [
    (25.0, 0.0),   # Freshwater at 25°C  
    (25.0, 35.0),  # Seawater at 25°C
    (28.0, 15.0),  # Estuarine at 28°C
    (20.0, 10.0),  # Brackish at 20°C
]

print("🧪 CORRECTED Oxygen Saturation Validation")
print("="*60)
print("Method: Garcia & Gordon (1992) - CORRECTED")
print("Temp(°C)  Salinity  O2_sat(mmol/m³)  O2_sat(mg/L)  O2_sat(μmol/kg)")
print("-"*70)

for temp, sal in test_conditions:
    mmol_m3, umol_kg = garcia_gordon_corrected(temp, sal)
    mg_l = mmol_m3 * 32.0 / 1000.0
    print(f"{temp:6.1f}    {sal:6.1f}    {mmol_m3:10.2f}    {mg_l:8.2f}    {umol_kg:10.2f}")

print("\n" + "="*60)
print("Method: Simplified Empirical (for comparison)")
print("Temp(°C)  Salinity  O2_sat(mmol/m³)  O2_sat(mg/L)  O2_sat(μmol/kg)")
print("-"*70)

for temp, sal in test_conditions:
    mmol_m3, umol_kg = simplified_oxygen_saturation(temp, sal)
    mg_l = mmol_m3 * 32.0 / 1000.0
    print(f"{temp:6.1f}    {sal:6.1f}    {mmol_m3:10.2f}    {mg_l:8.2f}    {umol_kg:10.2f}")

print("\n🎯 Expected realistic ranges:")
print("   Tropical freshwater (25°C, S=0): ~250 μmol/kg = ~8.0 mg/L")
print("   Tropical seawater (25°C, S=35): ~200 μmol/kg = ~6.4 mg/L")
print("   Estuarine water (28°C, S=15): ~220 μmol/kg = ~7.0 mg/L")