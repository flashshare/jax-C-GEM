#!/usr/bin/env python
"""
Detailed oxygen reaeration rate debugging to identify why levels remain low.
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.biogeochemistry import compute_atmospheric_reaeration
    from core.config_parser import parse_model_config
    from core.model_config import DEFAULT_BIO_PARAMS
except ImportError as e:
    print(f"Import error: {e}")
    
def debug_reaeration_rates():
    """Debug the actual reaeration rates being computed."""
    
    print("🔬 Debugging Atmospheric Reaeration Rates")
    print("="*50)
    
    # Test conditions
    current_o2 = 0.1  # mmol/m³ (extremely low, like current model)
    temperature = 28.0  # °C
    salinity = jnp.array([15.0])  # PSU
    depth = jnp.array([5.0])  # m
    dt = 180.0  # s (3 minute timestep)
    
    # Load parameters
    config = parse_model_config("config/model_config.txt")
    params = DEFAULT_BIO_PARAMS.copy()
    for key, value in config.items():
        if key in params:
            params[key] = float(value)
    
    print(f"🧪 Test conditions:")
    print(f"   Current O2: {current_o2:.3f} mmol/m³ ({current_o2 * 32/1000:.3f} mg/L)")
    print(f"   Temperature: {temperature}°C")
    print(f"   Salinity: {salinity[0]}% PSU")
    print(f"   Depth: {depth[0]} m")
    print(f"   Time step: {dt} s")
    print(f"   Wind speed: {params.get('wind_speed', 5.0)} m/s")
    
    # Test reaeration rate
    o2_conc = jnp.array([current_o2])
    reaeration_rate = compute_atmospheric_reaeration(
        o2_conc, temperature, salinity, depth, params, dt
    )
    
    print(f"\n💨 Reaeration calculation:")
    print(f"   Reaeration rate: {reaeration_rate[0]:.6f} mmol/m³/s")
    print(f"   Rate per hour: {reaeration_rate[0] * 3600:.6f} mmol/m³/hr")
    print(f"   Rate per day: {reaeration_rate[0] * 86400:.6f} mmol/m³/day")
    
    # Calculate oxygen saturation manually for comparison
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
    
    o2_sat = np.exp(ln_c_star) * 1.025 / 1000.0
    
    print(f"\n🫧 Saturation calculation:")
    print(f"   O2 saturation: {o2_sat:.3f} mmol/m³ ({o2_sat * 32/1000:.3f} mg/L)")
    print(f"   Saturation deficit: {o2_sat - current_o2:.3f} mmol/m³")
    
    # Calculate time to reach various saturation levels
    if reaeration_rate[0] > 0:
        time_to_50_percent = (0.5 * o2_sat - current_o2) / reaeration_rate[0] / 3600.0  # hours
        time_to_90_percent = (0.9 * o2_sat - current_o2) / reaeration_rate[0] / 3600.0  # hours
        print(f"\n⏰ Time to recovery:")
        print(f"   Time to 50% saturation: {time_to_50_percent:.1f} hours")
        print(f"   Time to 90% saturation: {time_to_90_percent:.1f} hours")
    else:
        print(f"\n❌ No reaeration occurring (rate = 0)")
    
    # Test different scenarios
    print(f"\n📊 Reaeration rate sensitivity:")
    test_o2_levels = [0.001, 0.01, 0.1, 0.5, 1.0]  # mmol/m³
    
    for test_o2 in test_o2_levels:
        test_rate = compute_atmospheric_reaeration(
            jnp.array([test_o2]), temperature, salinity, depth, params, dt
        )
        daily_rate = test_rate[0] * 86400
        print(f"   O2 = {test_o2:5.3f} mmol/m³: rate = {daily_rate:8.3f} mmol/m³/day")

if __name__ == "__main__":
    debug_reaeration_rates()