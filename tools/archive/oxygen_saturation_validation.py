#!/usr/bin/env python
"""
Quick oxygen saturation validation using known reference values.
"""
import numpy as np
import matplotlib.pyplot as plt

def garcia_gordon_1992(temp_c, salinity):
    """Reference implementation of Garcia & Gordon (1992) oxygen solubility."""
    temp_k = temp_c + 273.15
    
    # Coefficients from Garcia & Gordon (1992) - Table 1
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
    
    # Temperature scaling
    temp_scaled = 298.15 / temp_k
    
    # Solubility equation
    ln_c_star = (A0 + A1 * temp_scaled + A2 * np.log(temp_scaled) + 
                 A3 * temp_scaled + A4 * temp_scaled**2 + A5 * temp_scaled**3 +
                 salinity * (B0 + B1 * temp_scaled + B2 * temp_scaled**2 + 
                           B3 * temp_scaled**3) +
                 C0 * salinity**2)
    
    # Result in μmol/kg
    c_star_umol_kg = np.exp(ln_c_star)
    
    # Convert to mmol/m³ (assuming density = 1.025 kg/L = 1025 kg/m³)
    c_star_mmol_m3 = c_star_umol_kg * 1.025 / 1000.0
    
    return c_star_mmol_m3, c_star_umol_kg

# Test cases
test_conditions = [
    (25.0, 0.0),   # Freshwater at 25°C
    (25.0, 35.0),  # Seawater at 25°C  
    (28.0, 15.0),  # Estuarine at 28°C
    (20.0, 10.0),  # Brackish at 20°C
]

print("🧪 Oxygen Saturation Validation")
print("="*50)
print("Temp(°C)  Salinity  O2_sat(mmol/m³)  O2_sat(mg/L)  O2_sat(μmol/kg)")
print("-"*70)

for temp, sal in test_conditions:
    mmol_m3, umol_kg = garcia_gordon_1992(temp, sal)
    mg_l = mmol_m3 * 32.0 / 1000.0  # Convert mmol/m³ to mg/L
    
    print(f"{temp:6.1f}    {sal:6.1f}    {mmol_m3:10.2f}    {mg_l:8.2f}    {umol_kg:10.2f}")

print("\n🎯 Expected ranges:")
print("   Tropical freshwater (25°C, S=0): ~260 μmol/kg = ~8.3 mg/L")
print("   Tropical seawater (25°C, S=35): ~210 μmol/kg = ~6.7 mg/L")  
print("   Estuarine water (28°C, S=15): ~230 μmol/kg = ~7.4 mg/L")