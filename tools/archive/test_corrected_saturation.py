#!/usr/bin/env python
"""
Test the corrected saturation directly in the biogeochemistry function.
"""

import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.biogeochemistry import compute_atmospheric_reaeration
from core.model_config import DEFAULT_BIO_PARAMS

def test_corrected_saturation():
    """Test the corrected saturation calculation."""
    
    print("🧪 Testing Corrected O2 Saturation in JAX Function")
    print("="*55)
    
    # Test parameters
    temperature = 28.0  # °C
    salinity = jnp.array([15.0])  # PSU
    depth = jnp.array([5.0])  # m
    current_o2 = jnp.array([0.1])  # mmol/m³
    params = DEFAULT_BIO_PARAMS.copy()
    params['wind_speed'] = 5.0
    dt = 180.0
    
    # Manual calculation (corrected)
    temp_k = temperature + 273.15
    temp_s = np.log((298.15 - temperature) / temp_k)
    
    # Garcia & Gordon coefficients
    A0, A1, A2, A3 = 5.80871, 3.20291, 4.17887, 5.10006
    A4, A5 = -9.86643e-2, 3.80369
    B0, B1, B2, B3 = -7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3
    C0 = -2.75915e-7
    
    ln_c_star = (A0 + A1 * temp_s + A2 * temp_s**2 + A3 * temp_s**3 + 
                 A4 * temp_s**4 + A5 * temp_s**5 +
                 salinity[0] * (B0 + B1 * temp_s + B2 * temp_s**2 + B3 * temp_s**3) +
                 C0 * salinity[0]**2)
    
    # CORRECTED: Remove the /1000 division!
    o2_sat_corrected = np.exp(ln_c_star) * 1.025  # mmol/m³
    o2_sat_mg_l = o2_sat_corrected * 32.0 / 1000.0  # mg/L
    
    print(f"Manual corrected calculation:")
    print(f"   O2 saturation: {o2_sat_corrected:.1f} mmol/m³ ({o2_sat_mg_l:.1f} mg/L)")
    
    # Test JAX function
    reaeration_rate = compute_atmospheric_reaeration(
        current_o2, temperature, salinity, depth, params, dt
    )
    
    print(f"\nJAX function results:")
    print(f"   Reaeration rate: {reaeration_rate[0]:.6f} mmol/m³/s")
    print(f"   Daily rate: {reaeration_rate[0] * 86400:.1f} mmol/m³/day")
    
    # Calculate expected final O2 after 1 day
    expected_change_per_day = reaeration_rate[0] * 86400
    final_o2_estimate = current_o2[0] + expected_change_per_day
    final_o2_mg_l = final_o2_estimate * 32.0 / 1000.0
    
    print(f"\nProjected recovery (1 day):")
    print(f"   Starting O2: {current_o2[0]:.3f} mmol/m³ ({current_o2[0] * 32/1000:.3f} mg/L)")
    print(f"   Change per day: +{expected_change_per_day:.1f} mmol/m³")
    print(f"   Final O2: {final_o2_estimate:.1f} mmol/m³ ({final_o2_mg_l:.1f} mg/L)")
    
    if final_o2_mg_l > 3.0:
        print("✅ Reaeration should restore oxygen to reasonable levels!")
    else:
        print("❌ Still insufficient reaeration")

if __name__ == "__main__":
    test_corrected_saturation()