"""
Core configuration constants for the JAX C-GEM model.
Contains only fundamental physical constants and species definitions.
"""
import jax.numpy as jnp
from enum import IntEnum

# Physical constants
G = 9.81  # Gravity acceleration [m/s²]
PI = jnp.pi
TOL = 1e-6   # Convergence criterion - More realistic for numerical precision
MAXITS = 50  # Maximum iteration steps - Prevent infinite loops
EPS = 1e-5

# Species enumeration (must match C code enum)
class Species(IntEnum):
    PHY1 = 0   # Siliceous phytoplankton
    PHY2 = 1   # Non-siliceous phytoplankton  
    SI = 2     # Silica
    NO3 = 3    # Nitrate
    NH4 = 4    # Ammonium
    PO4 = 5    # Phosphate
    PIP = 6    # Particulate inorganic phosphorus
    O2 = 7     # Oxygen
    TOC = 8    # Total organic carbon
    S = 9      # Salinity
    SPM = 10   # Suspended particulate matter
    DIC = 11   # Dissolved inorganic carbon
    AT = 12    # Total alkalinity
    HS = 13    # Hydrogen sulfide
    PH = 14    # pH
    ALKC = 15  # Carbonate alkalinity
    CO2 = 16   # Carbon dioxide

# Species list for easy iteration
SPECIES_NAMES = [
    'PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 
    'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2'
]

MAXV = len(SPECIES_NAMES)  # Maximum number of species

# Redfield ratios (from C code)
REDFIELD_SI = 15.0 / 106.0
REDFIELD_N = 16.0 / 106.0  
REDFIELD_P = 1.0 / 106.0

# Default biogeochemical parameters (will be overridden by calibration)
DEFAULT_BIO_PARAMS = {
    # Phytoplankton parameters - FIXED: More balanced rates for stable dynamics
    'mumax_dia': 1.39e-5,  # Maximum growth rate for diatoms [1/s] - reduced 4x for stability
    'mumax_ndia': 1.39e-5,  # Maximum growth rate for non-diatoms [1/s] - reduced 4x for stability
    'alpha_light': 4.11e-7,  # Light saturation parameter
    'resp_dia': 4.0e-8,  # Respiration rate for diatoms [1/s] - reduced 4x for stability
    'resp_ndia': 4.0e-8,  # Respiration rate for non-diatoms [1/s] - reduced 4x for stability
    'mort_dia': 3.6e-8,  # Mortality rate for diatoms [1/s] - reduced 4x for stability
    'mort_ndia': 3.6e-8,  # Mortality rate for non-diatoms [1/s] - reduced 4x for stability
    
    # Half-saturation constants
    'ks_no3': 12.13,  # Half-saturation for NO3 [mmol/m³]
    'ks_nh4': 80.9,  # Half-saturation for NH4 [mmol/m³]
    'ks_po4': 0.05,  # Half-saturation for PO4 [mmol/m³]
    'ks_si': 7.07,  # Half-saturation for Si [mmol/m³]
    'ks_o2_resp': 31.0,  # Half-saturation for O2 in respiration
    'ks_o2_degrad': 31.0,  # Half-saturation for O2 in degradation
    'ks_o2_nitrif': 51.25,  # Half-saturation for O2 in nitrification
    'ks_no3_denitrif': 10.07,  # Half-saturation for NO3 in denitrification
    'ks_toc_denitrif': 312.5,  # Half-saturation for TOC in denitrification
    
    # Inhibition constants
    'ki_nh4': 80.9,  # NH4 inhibition of NO3 uptake
    'ki_o2_anaerobic': 33.0,  # O2 inhibition of anaerobic processes
    'ki_o2_denitrif': 33.0,  # O2 inhibition of denitrification
    
    # Rate constants - FIXED: More realistic rates for stable oxygen dynamics
    'degrad_rate': 1.8e-6,  # Organic matter degradation rate [1/s] - reduced 100x for stability
    'nitrif_rate': 1.6e-6,  # Nitrification rate [1/s] - reduced 100x for stability  
    'denitrif_rate': 3.05e-6,  # Denitrification rate [1/s] - reduced 100x for stability
    
    # Temperature dependence (Q10 values)
    'q10_phyto': 1.067,  # Q10 for phytoplankton processes
    'q10_resp': 1.067,  # Q10 for respiration
    'q10_mort': 1.067,  # Q10 for mortality
    'q10_degrad': 2.0,  # Q10 for degradation
    'q10_nitrif': 1.08,  # Q10 for nitrification
    'q10_denitrif': 1.07,  # Q10 for denitrification
    
    # Stoichiometric ratios (Redfield + extensions)
    'n_to_c': REDFIELD_N,  # N:C ratio
    'p_to_c': REDFIELD_P,  # P:C ratio
    'si_to_c': REDFIELD_SI,  # Si:C ratio
    'o2_to_c_photo': 106.0/106.0,  # O2:C ratio in photosynthesis
    'o2_to_c_resp': 106.0/106.0,  # O2:C ratio in respiration
    'o2_to_c_degrad': 106.0/106.0,  # O2:C ratio in degradation
    'o2_to_n_nitrif': 2.0,  # O2:N ratio in nitrification
    
    # Other parameters
    'anaerobic_factor': 0.1,  # Relative rate of anaerobic vs aerobic degradation
}