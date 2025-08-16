#!/usr/bin/env python
"""
CORRECTED Phase 1 Analysis Script: Longitudinal Profile Validation with Unit Conversion

This script properly aligns model coordinates with field observation coordinates
and applies correct unit conversions for accurate validation against CEM spatial data.

CRITICAL UNIT CORRECTIONS:
- Model units: mmol/m³ (biogeochemical species)
- Field units: mg/L (CEM observations)
- Conversion: 1 mmol/m³ = MW/1000 mg/L (MW = molecular weight)

Author: Nguyen Truong An
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.model_config import SPECIES_NAMES
except ImportError as e:
    print(f"Warning: Could not import model config: {e}")
    SPECIES_NAMES = ['PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2']

# Unit conversion factors: mmol/m³ → mg/L
UNIT_CONVERSION_FACTORS = {
    'NH4': 14.0 / 1000.0,    # NH4-N: 14 g/mol → mg N/L
    'NO3': 14.0 / 1000.0,    # NO3-N: 14 g/mol → mg N/L
    'PO4': 31.0 / 1000.0,    # PO4-P: 31 g/mol → mg P/L
    'TOC': 12.0 / 1000.0,    # TOC-C: 12 g/mol → mg C/L
    'O2': 32.0 / 1000.0,     # O2: 32 g/mol → mg/L
    'DIC': 12.0 / 1000.0,    # DIC-C: 12 g/mol → mg C/L
    'SI': 28.1 / 1000.0,     # Si: 28.1 g/mol → mg Si/L
    'SPM': 1.0,              # SPM already in mg/L
    'S': 1.0                 # Salinity dimensionless (psu)
}

def convert_model_to_field_units(concentration_mmol_m3, species):
    """Convert model concentration from mmol/m³ to mg/L for field comparison."""
    if species in UNIT_CONVERSION_FACTORS:
        return concentration_mmol_m3 * UNIT_CONVERSION_FACTORS[species]
    else:
        print(f"Warning: No conversion factor for {species}, returning as-is")
        return concentration_mmol_m3

def load_simulation_results(results_file="OUT/simulation_results.npz"):
    """Load simulation results with error handling."""
    try:
        print(f"📂 Loading simulation results from {results_file}")
        data = np.load(results_file, allow_pickle=True)
        
        # Extract arrays - species are stored individually
        results = {
            'time': data['time'],
            'H': data['H'],
            'U': data['U']
        }
        
        # Add species data
        for species in SPECIES_NAMES:
            if species in data:
                results[species] = data[species]
        
        # Create space coordinate (202 km total, 102 points, 2 km spacing)
        results['space'] = np.linspace(0, 202000, 102)  # meters from mouth
        
        print(f"✅ Loaded results: {len(results['time'])} time points, {len(results['space'])} spatial points")
        return results
    
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def load_field_observations():
    """Load and parse CEM field observations."""
    try:
        print("📊 Loading CEM field observations...")
        cem_file = "INPUT/Calibration/CEM_2017-2018.csv"
        cem_data = pd.read_csv(cem_file)
        print(f"✅ Loaded {len(cem_data)} CEM observations")
        return cem_data
    except Exception as e:
        print(f"❌ Error loading CEM data: {e}")
        return None

def create_corrected_station_mapping():
    """Create proper coordinate mapping between field and model coordinates."""
    # Field coordinates: km from upstream (mouth = 202 km)
    # Model coordinates: km from mouth (upstream = 202 km)
    # Conversion: model_km = 202 - field_km
    
    station_mapping = {
        'CEM_158': {  # Bến Súc station at 158 km from upstream
            'field_km': 158,
            'model_km': 202 - 158,  # 44 km from mouth
            'model_index': int((202 - 158) / 2),  # Index in 2km grid
            'name': 'Bến Súc (CEM)'
        }
    }
    
    print("📍 Station coordinate mapping:")
    for station, info in station_mapping.items():
        print(f"  {info['name']}: Field {info['field_km']}km → Model {info['model_km']}km (index {info['model_index']})")
    
    return station_mapping

def calculate_time_averaged_profiles(data, warmup_days=100, analysis_days=265):
    """Calculate time-averaged profiles for final 265 days (post-warmup)."""
    total_days = len(data['time']) // (24 * 60 // 3)  # Assuming 3-min time steps
    warmup_steps = warmup_days * (24 * 60 // 3)
    
    if len(data['time']) <= warmup_steps:
        print(f"⚠️  Warning: Dataset has {len(data['time'])} steps, warmup needs {warmup_steps}")
        analysis_start = len(data['time']) // 2
    else:
        analysis_start = warmup_steps
    
    print(f"📊 Calculating time-averaged profiles:")
    print(f"   Total steps: {len(data['time'])}")
    print(f"   Analysis period: steps {analysis_start} to {len(data['time'])}")
    
    # Calculate time-averaged concentrations for each species
    profiles = {}
    for species in SPECIES_NAMES:
        if species in data:
            # Time average over analysis period
            species_data = data[species][analysis_start:, :]  # [time, space]
            profiles[species] = np.mean(species_data, axis=0)  # [space]
    
    # Add hydrodynamics
    profiles['H'] = np.mean(data['H'][analysis_start:, :], axis=0)
    profiles['U'] = np.mean(data['U'][analysis_start:, :], axis=0)
    profiles['space_km'] = data['space'] / 1000.0  # Convert m to km
    
    print(f"✅ Calculated profiles for {len(profiles)-3} species + hydrodynamics")
    return profiles

def aggregate_field_data_by_location(cem_data):
    """Aggregate CEM field data by location for comparison."""
    print("📊 Aggregating CEM field data by location...")
    
    # Group by location and calculate statistics
    location_stats = {}
    for location in cem_data['Location'].unique():
        if pd.isna(location):
            continue
            
        location_data = cem_data[cem_data['Location'] == location]
        
        stats = {
            'location_km': float(location),
            'n_observations': len(location_data),
            'salinity': {
                'mean': location_data['Salinity'].mean(),
                'std': location_data['Salinity'].std(),
                'min': location_data['Salinity'].min(),
                'max': location_data['Salinity'].max()
            }
        }
        
        # Add other species if available
        for species, unit_col in [('NH4', 'NH4 (mgN/L)'), ('PO4', 'PO4 (mgP/L)'), ('TOC', 'TOC (mgC/L)')]:
            if unit_col in location_data.columns:
                valid_data = pd.to_numeric(location_data[unit_col], errors='coerce').dropna()
                if len(valid_data) > 0:
                    stats[species.lower()] = {
                        'mean': valid_data.mean(),
                        'std': valid_data.std(),
                        'min': valid_data.min(), 
                        'max': valid_data.max()
                    }
        
        location_stats[location] = stats
    
    print(f"✅ Aggregated data for {len(location_stats)} locations")
    return location_stats

def create_corrected_validation_figure(profiles, cem_data, stations, output_dir="OUT"):
    """Create validation figure with proper unit conversion."""
    print("📈 Creating corrected validation figure with unit conversion...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CORRECTED Phase 1 Validation: Longitudinal Profiles vs CEM Field Data\n(WITH PROPER UNIT CONVERSION)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Salinity profile
    ax1 = axes[0, 0]
    ax1.plot(profiles['space_km'], profiles['S'], 'b-', linewidth=2, label='JAX C-GEM Model')
    
    # Add CEM salinity observations with proper coordinates
    cem_locations = cem_data.groupby('Location')['Salinity'].mean().reset_index()
    for _, row in cem_locations.iterrows():
        if not pd.isna(row['Location']) and not pd.isna(row['Salinity']):
            field_km = row['Location']
            model_km = 202 - field_km  # Coordinate conversion
            ax1.plot(model_km, row['Salinity'], 'ro', markersize=8, label='CEM Observations' if _ == 0 else "")
    
    ax1.set_xlabel('Distance from mouth (km)')
    ax1.set_ylabel('Salinity (psu)')
    ax1.set_title('Salinity Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. NH4 profile with unit conversion
    ax2 = axes[0, 1]
    # Convert model NH4 from mmol/m³ to mgN/L
    model_nh4_mg_L = convert_model_to_field_units(profiles['NH4'], 'NH4')
    ax2.plot(profiles['space_km'], model_nh4_mg_L, 'b-', linewidth=2, label='JAX C-GEM Model (converted)')
    
    # Add CEM NH4 observations
    for _, row in cem_data.iterrows():
        if not pd.isna(row['Location']) and not pd.isna(row.get('NH4 (mgN/L)', np.nan)):
            field_km = row['Location']
            model_km = 202 - field_km
            ax2.plot(model_km, row['NH4 (mgN/L)'], 'ro', markersize=6, alpha=0.7)
    
    ax2.set_xlabel('Distance from mouth (km)')
    ax2.set_ylabel('NH4 (mgN/L)')
    ax2.set_title('NH4 Profile - UNITS CORRECTED')
    ax2.grid(True, alpha=0.3)
    
    # 3. PO4 profile with unit conversion
    ax3 = axes[1, 0]
    model_po4_mg_L = convert_model_to_field_units(profiles['PO4'], 'PO4')
    ax3.plot(profiles['space_km'], model_po4_mg_L, 'b-', linewidth=2, label='JAX C-GEM Model (converted)')
    
    ax3.set_xlabel('Distance from mouth (km)')
    ax3.set_ylabel('PO4 (mgP/L)')
    ax3.set_title('PO4 Profile - UNITS CORRECTED')
    ax3.grid(True, alpha=0.3)
    
    # 4. TOC profile with unit conversion
    ax4 = axes[1, 1]
    model_toc_mg_L = convert_model_to_field_units(profiles['TOC'], 'TOC')
    ax4.plot(profiles['space_km'], model_toc_mg_L, 'b-', linewidth=2, label='JAX C-GEM Model (converted)')
    
    ax4.set_xlabel('Distance from mouth (km)')
    ax4.set_ylabel('TOC (mgC/L)')
    ax4.set_title('TOC Profile - UNITS CORRECTED')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase1_corrected_units_longitudinal_profiles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved figure: {output_path}")
    
    plt.show()
    return str(output_path)

def calculate_validation_metrics_corrected(profiles, cem_data, stations):
    """Calculate validation metrics with proper unit conversion."""
    print("\n📊 CORRECTED VALIDATION METRICS (with unit conversion)")
    print("="*70)
    
    validation_results = {}
    
    # Salinity validation (already in same units)
    station_158 = stations['CEM_158']
    model_idx = station_158['model_index']
    
    # Extract model values at station location
    model_salinity = profiles['S'][model_idx]
    model_nh4_mg_L = convert_model_to_field_units(profiles['NH4'][model_idx], 'NH4')
    model_po4_mg_L = convert_model_to_field_units(profiles['PO4'][model_idx], 'PO4')
    model_toc_mg_L = convert_model_to_field_units(profiles['TOC'][model_idx], 'TOC')
    
    # Field observations at location 158
    field_158 = cem_data[cem_data['Location'] == 158.0]
    
    if len(field_158) > 0:
        field_salinity = field_158['Salinity'].mean()
        field_nh4 = pd.to_numeric(field_158['NH4 (mgN/L)'], errors='coerce').mean()
        
        print(f"🎯 STATION COMPARISON (Bến Súc - 158km from upstream, 44km from mouth):")
        print(f"   Salinity:  Model = {model_salinity:.2f} psu,  Field = {field_salinity:.2f} psu")
        print(f"   NH4:       Model = {model_nh4_mg_L:.3f} mgN/L, Field = {field_nh4:.3f} mgN/L")
        print(f"   PO4:       Model = {model_po4_mg_L:.3f} mgP/L, Field = N/A")
        print(f"   TOC:       Model = {model_toc_mg_L:.3f} mgC/L, Field = N/A")
        
        validation_results = {
            'station': 'Bến Súc (158km)',
            'model_salinity_psu': model_salinity,
            'field_salinity_psu': field_salinity,
            'model_nh4_mgN_L': model_nh4_mg_L,
            'field_nh4_mgN_L': field_nh4,
            'salinity_error': abs(model_salinity - field_salinity),
            'nh4_error_mgN_L': abs(model_nh4_mg_L - field_nh4)
        }
    
    return validation_results

def main():
    """Main validation function with unit conversion."""
    print("🔬 CORRECTED PHASE 1 VALIDATION: Longitudinal Profiles with Unit Conversion")
    print("="*80)
    
    # Load simulation results
    results = load_simulation_results()
    if results is None:
        return
    
    # Load field observations
    cem_data = load_field_observations()
    if cem_data is None:
        return
    
    # Create station mapping
    stations = create_corrected_station_mapping()
    
    # Calculate time-averaged profiles
    profiles = calculate_time_averaged_profiles(results)
    
    # Aggregate field data
    field_stats = aggregate_field_data_by_location(cem_data)
    
    # Create validation figure
    figure_path = create_corrected_validation_figure(profiles, cem_data, stations)
    
    # Calculate metrics
    metrics = calculate_validation_metrics_corrected(profiles, cem_data, stations)
    
    # Generate report
    report_path = "OUT/phase1_corrected_units_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("CORRECTED PHASE 1 VALIDATION REPORT - WITH UNIT CONVERSION\\n")
        f.write("="*60 + "\\n\\n")
        f.write("CRITICAL CORRECTION: Applied proper unit conversion\\n")
        f.write("- Model units: mmol/m³ (biogeochemical species)\\n")
        f.write("- Field units: mg/L (CEM observations)\\n")
        f.write("- Conversion factors applied for all species\\n\\n")
        
        if metrics:
            f.write(f"STATION VALIDATION RESULTS:\\n")
            f.write(f"Station: {metrics['station']}\\n")
            f.write(f"Salinity Error: {metrics['salinity_error']:.3f} psu\\n")
            f.write(f"NH4 Error: {metrics['nh4_error_mgN_L']:.6f} mgN/L\\n")
        
        f.write(f"\\nGenerated figure: {figure_path}\\n")
    
    print(f"\\n📝 Generated validation report: {report_path}")
    print("\\n✅ CORRECTED PHASE 1 VALIDATION COMPLETE")

if __name__ == "__main__":
    main()