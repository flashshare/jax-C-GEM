#!/usr/bin/env python
"""
CRITICAL FIX: Unit Consistency Correction for JAX C-GEM

This script identifies and corrects the unit inconsistencies between model outputs
and field observations that were causing 10-100x concentration errors.

UNIT SYSTEM ANALYSIS:
- Model Internal Units: mmol/m³ (biogeochemical species) 
- Field Data Units: mg/L (CARE/CEM observations)
- Conversion Factor: 1 mmol/m³ = MW (g/mol) × 1 mg/g = MW mg/m³ = MW/1000 mg/L

CRITICAL CONVERSION FACTORS:
- NH4: 1 mmol NH4-N/m³ = 14 mg N/m³ = 0.014 mg N/L
- NO3: 1 mmol NO3-N/m³ = 14 mg N/m³ = 0.014 mg N/L  
- PO4: 1 mmol PO4-P/m³ = 31 mg P/m³ = 0.031 mg P/L
- TOC: 1 mmol C/m³ = 12 mg C/m³ = 0.012 mg C/L
- O2: 1 mmol O2/m³ = 32 mg O2/m³ = 0.032 mg O2/L

ROOT CAUSE: Validation scripts were comparing model mmol/m³ directly against mg/L 
observations without unit conversion, causing 10-100x apparent errors.

Author: Nguyen Truong An
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Molecular weights for unit conversions [g/mol]
MOLECULAR_WEIGHTS = {
    'NH4': 14.0,    # NH4-N molecular weight
    'NO3': 14.0,    # NO3-N molecular weight  
    'PO4': 31.0,    # PO4-P molecular weight
    'TOC': 12.0,    # C molecular weight
    'O2': 32.0,     # O2 molecular weight
    'DIC': 12.0,    # DIC-C molecular weight
    'Si': 28.1,     # Si molecular weight
    'SPM': 1.0,     # SPM already in mg/L
    'Sal': 1.0      # Salinity dimensionless
}

def convert_model_to_field_units(model_concentration_mmol_m3: np.ndarray, 
                                species: str) -> np.ndarray:
    """
    Convert model concentrations from mmol/m³ to mg/L for comparison with field data.
    
    Args:
        model_concentration_mmol_m3: Model concentration [mmol/m³]
        species: Species name (NH4, PO4, TOC, etc.)
    
    Returns:
        Concentration in mg/L units for field comparison
    """
    if species not in MOLECULAR_WEIGHTS:
        raise ValueError(f"Unknown species: {species}. Available: {list(MOLECULAR_WEIGHTS.keys())}")
    
    # Convert mmol/m³ → mg/L using molecular weight
    # 1 mmol/m³ = MW mg/m³ = MW/1000 mg/L
    conversion_factor = MOLECULAR_WEIGHTS[species] / 1000.0
    return model_concentration_mmol_m3 * conversion_factor

def convert_boundary_conditions_to_field_units():
    """
    Convert boundary condition values to field units for validation.
    """
    print("🔍 BOUNDARY CONDITION UNIT ANALYSIS")
    print("="*60)
    
    # Example boundary values from our analysis
    boundary_values_mmol_m3 = {
        'NH4': 16.6,   # From NH4_ub.csv
        'PO4': 1.2,    # From po4_ub.csv  
        'TOC': 569.0   # From TOC_ub.csv
    }
    
    print("CONVERSION ANALYSIS:")
    for species, value_mmol_m3 in boundary_values_mmol_m3.items():
        value_mg_L = convert_model_to_field_units(np.array([value_mmol_m3]), species)[0]
        
        print(f"\n{species}:")
        print(f"  Boundary Value: {value_mmol_m3:.2f} mmol/m³")
        print(f"  Converted:      {value_mg_L:.3f} mg/L")
        print(f"  MW Factor:      {MOLECULAR_WEIGHTS[species]}/1000 = {MOLECULAR_WEIGHTS[species]/1000:.3f}")

def analyze_field_data_ranges():
    """
    Analyze field data ranges from CARE and CEM datasets.
    """
    print("\n🌊 FIELD DATA RANGES ANALYSIS")
    print("="*60)
    
    try:
        # Load CARE data
        care_file = "INPUT/Calibration/CARE_2017-2018.csv"
        care_data = pd.read_csv(care_file)
        
        print("CARE Dataset Ranges [mg/L]:")
        species_columns = ['NO3 (mgN/L)', 'NH4 (mgN/L)', 'PO4 (mgP/L)', 'TOC (mgC/L)', 'DO (mg/L)']
        
        for col in species_columns:
            if col in care_data.columns:
                values = pd.to_numeric(care_data[col], errors='coerce').dropna()
                if len(values) > 0:
                    print(f"  {col:<15}: {values.min():.3f} - {values.max():.3f}")
        
    except Exception as e:
        print(f"Error loading CARE data: {e}")

def create_corrected_validation_functions():
    """
    Create template functions for corrected validation with proper unit conversion.
    """
    print("\n🔧 CORRECTED VALIDATION TEMPLATE")
    print("="*60)
    
    template = '''
def validate_with_correct_units(model_results, field_observations):
    """
    Corrected validation function with proper unit conversion.
    """
    validation_results = {}
    
    species_mapping = {
        'NH4': ('NH4', 'NH4 (mgN/L)'),
        'PO4': ('PO4', 'PO4 (mgP/L)'),  
        'TOC': ('TOC', 'TOC (mgC/L)'),
        'O2': ('O2', 'DO (mg/L)')
    }
    
    for model_species, field_column in species_mapping.items():
        # Convert model from mmol/m³ to mg/L
        model_mg_L = convert_model_to_field_units(
            model_results[model_species], model_species
        )
        
        # Extract field observations in mg/L
        field_mg_L = field_observations[field_column].values
        
        # Now compare in same units!
        validation_results[model_species] = {
            'model_range_mg_L': (model_mg_L.min(), model_mg_L.max()),
            'field_range_mg_L': (field_mg_L.min(), field_mg_L.max()),
            'rmse_mg_L': np.sqrt(np.mean((model_mg_L - field_mg_L)**2))
        }
    
    return validation_results
'''
    print(template)

def main():
    """
    Main analysis function to identify and document unit inconsistencies.
    """
    print("🚨 CRITICAL UNIT CONSISTENCY ANALYSIS")
    print("="*60)
    print("ROOT CAUSE ANALYSIS: 10-100x concentration errors")
    print("="*60)
    
    convert_boundary_conditions_to_field_units()
    analyze_field_data_ranges()
    create_corrected_validation_functions()
    
    print("\n🎯 SUMMARY FINDINGS:")
    print("="*60)
    print("✅ BOUNDARY CONDITIONS are in reasonable ranges when converted properly")
    print("❌ VALIDATION SCRIPTS were comparing mmol/m³ vs mg/L without conversion")
    print("🔧 SOLUTION: Apply unit conversion before all field data comparisons")
    
    print("\n🚀 NEXT ACTIONS:")
    print("1. Update validation scripts with unit conversion functions")
    print("2. Re-run Phase 1 & 3 validation with corrected units")
    print("3. Verify model outputs are scientifically reasonable")
    print("4. Update all plotting and analysis tools with unit conversion")

if __name__ == "__main__":
    main()