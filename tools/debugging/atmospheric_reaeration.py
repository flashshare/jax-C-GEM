#!/usr/bin/env python
"""
Atmospheric Reaeration Implementation for JAX C-GEM

This script implements the missing atmospheric reaeration functionality
that is causing the dissolved oxygen system collapse.

Based on:
- O'Connor-Dobbins model for gas exchange
- Wanninkhof (1992) wind-based parameterization 
- Schmidt number temperature correction

Author: Phase I Critical Fix Team
"""

import jax.numpy as jnp
import jax
from typing import Dict, Any
import numpy as np

def compute_schmidt_number(temp_c: jnp.ndarray, gas='O2') -> jnp.ndarray:
    """
    Compute Schmidt number for gas exchange.
    
    Schmidt number = kinematic viscosity / molecular diffusivity
    Temperature dependence from Wanninkhof (1992)
    
    Args:
        temp_c: Temperature in Celsius [°C]
        gas: Gas species ('O2', 'CO2', etc.)
    
    Returns:
        Schmidt number [dimensionless]
    """
    if gas == 'O2':
        # Oxygen Schmidt number (Wanninkhof 1992)
        # Sc = 1953.4 - 128.0*T + 3.9918*T² - 0.050091*T³
        sc = (1953.4 - 128.0 * temp_c + 
              3.9918 * temp_c**2 - 
              0.050091 * temp_c**3)
    elif gas == 'CO2':
        # CO2 Schmidt number
        sc = (2073.1 - 125.62 * temp_c + 
              3.6276 * temp_c**2 - 
              0.043219 * temp_c**3)
    else:
        # Default to O2
        sc = (1953.4 - 128.0 * temp_c + 
              3.9918 * temp_c**2 - 
              0.050091 * temp_c**3)
    
    return jnp.maximum(sc, 100.0)  # Avoid unrealistic values

def compute_gas_transfer_velocity(wind_speed: jnp.ndarray, 
                                temp_c: jnp.ndarray,
                                gas='O2') -> jnp.ndarray:
    """
    Compute gas transfer velocity using Wanninkhof (1992) relationship.
    
    k = 0.31 * u² * (Sc/660)^(-0.5)
    
    Args:
        wind_speed: Wind speed at 10m height [m/s]
        temp_c: Temperature [°C]
        gas: Gas species
    
    Returns:
        Gas transfer velocity [m/day]
    """
    # Schmidt number
    sc = compute_schmidt_number(temp_c, gas)
    
    # Wanninkhof (1992) parameterization
    # k600 = 0.31 * u² for u in m/s, k in cm/hr
    # Convert to m/day: cm/hr * 0.01 * 24 = m/day * 0.24
    
    k_wanninkhof = 0.31 * wind_speed**2  # cm/hr
    
    # Schmidt number correction
    k_corrected = k_wanninkhof * (sc / 660.0)**(-0.5)  # cm/hr
    
    # Convert cm/hr to m/day
    k_m_per_day = k_corrected * 0.01 * 24.0
    
    return k_m_per_day

def compute_oxygen_saturation(temp_c: jnp.ndarray, 
                            salinity: jnp.ndarray) -> jnp.ndarray:
    """
    Compute oxygen saturation concentration using Garcia & Gordon (1992).
    
    Standard equation for dissolved oxygen solubility in seawater.
    
    Args:
        temp_c: Temperature [°C]
        salinity: Salinity [PSU]
    
    Returns:
        Oxygen saturation [mmol/m³]
    """
    # Convert temperature to absolute scale for calculations
    temp_k = temp_c + 273.15
    
    # Garcia & Gordon (1992) coefficients
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
    
    temp_scaled = 298.15 / temp_k
    
    # Natural logarithm of oxygen solubility
    ln_c_star = (A0 + A1 * temp_scaled + A2 * jnp.log(temp_scaled) + 
                 A3 * temp_scaled + A4 * temp_scaled**2 + A5 * temp_scaled**3 +
                 salinity * (B0 + B1 * temp_scaled + B2 * temp_scaled**2 + 
                           B3 * temp_scaled**3) +
                 C0 * salinity**2)
    
    # Convert from μmol/kg to mmol/m³
    # Assuming seawater density ≈ 1025 kg/m³
    c_star_umol_kg = jnp.exp(ln_c_star)  # μmol/kg
    c_star_mmol_m3 = c_star_umol_kg * 1.025 / 1000.0  # mmol/m³
    
    return c_star_mmol_m3

def atmospheric_reaeration_step(o2_conc: jnp.ndarray,
                              temp: jnp.ndarray,
                              salinity: jnp.ndarray,
                              wind_speed: jnp.ndarray,
                              depth: jnp.ndarray,
                              dt: float) -> jnp.ndarray:
    """
    Apply atmospheric reaeration to dissolved oxygen.
    
    Based on two-film theory:
    dO2/dt = k * (O2_sat - O2) / depth
    
    Args:
        o2_conc: Current O2 concentration [mmol/m³]
        temp: Temperature [°C]
        salinity: Salinity [PSU]
        wind_speed: Wind speed [m/s]
        depth: Water depth [m]
        dt: Time step [days]
    
    Returns:
        O2 concentration change [mmol/m³]
    """
    # Compute saturation concentration
    o2_sat = compute_oxygen_saturation(temp, salinity)
    
    # Compute gas transfer velocity
    k_gas = compute_gas_transfer_velocity(wind_speed, temp, gas='O2')  # m/day
    
    # Two-film theory: flux = k * (C_sat - C)
    # Change in concentration = flux / depth
    saturation_deficit = o2_sat - o2_conc
    
    # Reaeration rate [mmol/m³/day]
    reaeration_rate = k_gas * saturation_deficit / depth
    
    # Apply time step
    do2_dt_reaeration = reaeration_rate * dt
    
    return do2_dt_reaeration

def estimate_wind_speed_from_tidal_velocity(u_tidal: jnp.ndarray) -> jnp.ndarray:
    """
    Estimate wind speed from tidal velocity when wind data unavailable.
    
    This is a simplified approach - ideally wind should be provided as input.
    Assumes wind creates surface stress similar to tidal flow.
    
    Args:
        u_tidal: Tidal velocity [m/s]
    
    Returns:
        Estimated wind speed [m/s]
    """
    # Simple scaling relationship
    # Typical estuarine conditions: wind ~ 3-8 m/s
    baseline_wind = 5.0  # m/s
    
    # Add variability based on tidal velocity
    tidal_component = 2.0 * jnp.tanh(jnp.abs(u_tidal) / 0.5)
    
    wind_estimated = baseline_wind + tidal_component
    
    return jnp.clip(wind_estimated, 1.0, 15.0)

@jax.jit
def apply_atmospheric_reaeration(state: jnp.ndarray,
                               params: Dict[str, Any],
                               temp: jnp.ndarray,
                               salinity: jnp.ndarray, 
                               velocity: jnp.ndarray,
                               depth: jnp.ndarray,
                               dt: float) -> jnp.ndarray:
    """
    Apply atmospheric reaeration to the biogeochemical state.
    
    This is the main function to integrate into biogeochemistry.py
    
    Args:
        state: Biogeochemical state [n_species, n_grid]
        params: Model parameters
        temp: Temperature [°C] [n_grid]
        salinity: Salinity [PSU] [n_grid] 
        velocity: Tidal velocity [m/s] [n_grid]
        depth: Water depth [m] [n_grid]
        dt: Time step [days]
    
    Returns:
        Updated state with reaeration applied
    """
    # Extract oxygen (species index 7)
    o2_idx = 7
    o2_conc = state[o2_idx, :]
    
    # Get wind speed (from params or estimate from tidal velocity)
    if 'wind_speed' in params:
        wind_speed = jnp.full_like(depth, params['wind_speed'])
    else:
        # Estimate from tidal velocity
        wind_speed = estimate_wind_speed_from_tidal_velocity(velocity)
    
    # Apply reaeration
    do2_dt_reaeration = atmospheric_reaeration_step(
        o2_conc, temp, salinity, wind_speed, depth, dt
    )
    
    # Update state
    new_state = state.at[o2_idx, :].add(do2_dt_reaeration)
    
    return new_state

def validate_reaeration_parameters():
    """
    Validate reaeration implementation with known test cases.
    """
    print("🧪 Validating Atmospheric Reaeration Implementation")
    print("="*55)
    
    # Test case: Tropical estuary conditions
    temp_c = 28.0  # °C
    salinity = 15.0  # PSU
    wind_speed = 5.0  # m/s
    depth = 5.0  # m
    
    # Compute saturation
    o2_sat = compute_oxygen_saturation(
        jnp.array([temp_c]), jnp.array([salinity])
    )[0]
    
    print(f"🌡️  Temperature: {temp_c}°C")
    print(f"🧂 Salinity: {salinity} PSU") 
    print(f"💨 Wind speed: {wind_speed} m/s")
    print(f"📏 Depth: {depth} m")
    print(f"🫧 O2 saturation: {o2_sat:.2f} mmol/m³ ({o2_sat*32/1000:.2f} mg/L)")
    
    # Test different deficit scenarios
    deficits = [0.0, 0.2, 0.5, 0.8]  # Fraction of saturation
    
    print("\n🔄 Reaeration rates for different O2 deficits:")
    print("   O2%    Current    Rate      Time to 95%")
    print("   ---    -------    ----      -----------")
    
    for deficit in deficits:
        current_o2 = o2_sat * (1.0 - deficit)
        
        # Compute reaeration rate
        k_gas = compute_gas_transfer_velocity(
            jnp.array([wind_speed]), jnp.array([temp_c])
        )[0]
        
        reaeration_rate = k_gas * (o2_sat - current_o2) / depth
        
        # Time to 95% saturation (exponential approach)
        if reaeration_rate > 0:
            time_to_95 = -jnp.log(0.05) * depth / k_gas
        else:
            time_to_95 = float('inf')
        
        o2_percent = (1.0 - deficit) * 100
        print(f"   {o2_percent:3.0f}%    {current_o2:5.1f}      {reaeration_rate:4.1f}      {time_to_95:5.1f} days")
    
    print("\n✅ Reaeration validation complete")
    
    # Expected values check
    expected_k_gas = 2.0  # m/day for 5 m/s wind
    computed_k_gas = compute_gas_transfer_velocity(
        jnp.array([5.0]), jnp.array([25.0])
    )[0]
    
    if abs(computed_k_gas - expected_k_gas) < 1.0:
        print(f"✅ Gas transfer velocity reasonable: {computed_k_gas:.2f} m/day")
    else:
        print(f"⚠️  Gas transfer velocity check: {computed_k_gas:.2f} m/day (expected ~{expected_k_gas})")

if __name__ == "__main__":
    validate_reaeration_parameters()