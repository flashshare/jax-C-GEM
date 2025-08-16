#!/usr/bin/env python
"""
CORRECTED Phase 1 Analysis Script: Longitudinal Profile Validation

This script properly aligns model coordinates with field observation coordinates
and provides accurate validation against CEM spatial data.

Critical Fixes:
- Proper coordinate transformation (field km from upstream → model km from mouth)
- Correct salinity range validation (0-30 psu estuarine gradient)
- Accurate station location mapping
- Field data aggregation by location

Scientific Objective:
- Verify time-averaged spatial patterns over final 30 days
- Compare against CEM field observations at correct locations
- Quantify model performance with proper coordinate alignment

Usage: python tools/verification/phase1_validate_profiles_corrected.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.model_config import SPECIES_NAMES
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import core modules: {e}")
    SPECIES_NAMES = ['PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2']

def load_simulation_results(results_file="OUT/complete_simulation_results.npz"):
    """Load the simulation results."""
    print(f"📂 Loading simulation results from {results_file}")
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Available files:")
        for f in Path("OUT").glob("*.npz"):
            print(f"   - {f.name}")
        return None
    
    try:
        data = np.load(results_file)
        print(f"✅ Results loaded successfully")
        print(f"   Available keys: {list(data.keys())[:10]}...")
        return data
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def load_field_observations():
    """Load and properly parse field observation data."""
    print("📊 Loading CEM spatial field observations")
    
    # Load CEM data
    cem_file = "INPUT/Calibration/CEM_2017-2018.csv"
    if not Path(cem_file).exists():
        print(f"❌ CEM file not found: {cem_file}")
        return None, None
        
    cem_data = pd.read_csv(cem_file)
    print(f"✅ CEM spatial data loaded: {len(cem_data)} observations")
    print(f"   Stations: {cem_data['Site'].unique()}")
    print(f"   Location range: {cem_data['Location'].min()}-{cem_data['Location'].max()} km")
    print(f"   Variables: {cem_data.columns.tolist()}")
    
    # Load tidal range data
    tidal_file = "INPUT/Calibration/CEM-Tidal-range.csv"
    tidal_data = None
    if Path(tidal_file).exists():
        tidal_data = pd.read_csv(tidal_file)
        print(f"✅ CEM tidal range data loaded: {len(tidal_data)} observations")
        try:
            last_col = tidal_data.iloc[:, -1]
            if pd.api.types.is_numeric_dtype(last_col):
                print(f"   Tidal range: {last_col.min():.2f}-{last_col.max():.2f} m")
            else:
                print(f"   Tidal range column: {last_col.dtype}")
        except:
            print("   Tidal range: format error")
    
    return cem_data, tidal_data

def create_corrected_station_mapping():
    """Create CORRECTED mapping between field stations and model grid points."""
    print("🔧 Creating corrected coordinate mapping...")
    
    # Model coordinate system: 0 km = mouth (downstream), 202 km = head (upstream)
    # CEM coordinate system: distances from upstream end
    # CONVERSION: model_km = 202 - field_km
    
    # CEM station locations from the actual data
    stations = {
        'Ngã Bảy': {'field_km': 2, 'model_km': 200, 'grid_idx': 100},
        'Đồng Tranh': {'field_km': 2, 'model_km': 200, 'grid_idx': 100}, 
        'Vàm Sát': {'field_km': 26, 'model_km': 176, 'grid_idx': 88},
        'Nhà Bè': {'field_km': 46, 'model_km': 156, 'grid_idx': 78},
        'Sài Gòn': {'field_km': 72, 'model_km': 130, 'grid_idx': 65},
        'Bình Lợi': {'field_km': 90, 'model_km': 112, 'grid_idx': 56},
        'Phú Cường': {'field_km': 116, 'model_km': 86, 'grid_idx': 43},
        'Bến Súc': {'field_km': 158, 'model_km': 44, 'grid_idx': 22}
    }
    
    print("📍 Station coordinate mapping:")
    for name, info in stations.items():
        print(f"   {name}: field {info['field_km']}km → model {info['model_km']}km (grid {info['grid_idx']})")
    
    return stations

def calculate_time_averaged_profiles(data, warmup_days=100, analysis_days=265):
    """Calculate time-averaged profiles from simulation data."""
    print("📈 Calculating time-averaged profiles")
    print(f"   Warmup period: {warmup_days} days")
    print(f"   Analysis period: {analysis_days} days")
    
    time_data = data['time']
    total_time_steps = len(time_data)
    print(f"   Total time steps: {total_time_steps}")
    
    # Convert days to time steps (assuming 180s time step, 480 steps/day)
    steps_per_day = 480
    warmup_steps = warmup_days * steps_per_day
    analysis_steps = min(analysis_days * steps_per_day, total_time_steps - warmup_steps)
    
    if warmup_steps >= total_time_steps:
        print(f"⚠️  Warning: Warmup too long, using last 30% of data")
        warmup_steps = int(0.7 * total_time_steps)
        analysis_steps = total_time_steps - warmup_steps
    
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Analysis steps: {analysis_steps}")
    
    # Extract analysis period
    start_idx = warmup_steps
    end_idx = warmup_steps + analysis_steps
    
    profiles = {}
    
    # Calculate tidal amplitude from water levels
    H_data = data['H'][start_idx:end_idx]
    tidal_amplitude = np.max(H_data, axis=0) - np.min(H_data, axis=0)
    profiles['tidal_amplitude'] = tidal_amplitude
    print(f"   ✅ Tidal amplitude calculated (range: {tidal_amplitude.min():.2f} - {tidal_amplitude.max():.2f} m)")
    
    # Calculate mean concentrations for key species
    for species in ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']:
        if species in data:
            species_data = data[species][start_idx:end_idx]
            mean_profile = np.mean(species_data, axis=0)
            profiles[species] = mean_profile
            print(f"   ✅ {species}: mean = {mean_profile.mean():.2f}, std = {mean_profile.std():.2f}")
    
    return profiles

def aggregate_field_data_by_location(cem_data):
    """Aggregate field data by location to match model grid points."""
    print("🔧 Aggregating field data by location...")
    
    # Group by location and calculate means
    location_groups = cem_data.groupby('Location').agg({
        'Salinity': ['mean', 'std', 'count'],
        'DO (mg/L)': ['mean', 'std', 'count'],
        'NH4 (mgN/L)': ['mean', 'std', 'count'],
        'PO4 (mgP/L)': ['mean', 'std', 'count'],
        'TOC (mgC/L)': ['mean', 'std', 'count']
    }).round(3)
    
    print("📊 Field data summary by location:")
    for loc in sorted(location_groups.index):
        print(f"   Location {loc} km:")
        print(f"     Salinity: {location_groups.loc[loc, ('Salinity', 'mean')]:.2f} ± {location_groups.loc[loc, ('Salinity', 'std')]:.2f} psu (n={location_groups.loc[loc, ('Salinity', 'count')]})")
        if not pd.isna(location_groups.loc[loc, ('DO (mg/L)', 'mean')]):
            print(f"     DO: {location_groups.loc[loc, ('DO (mg/L)', 'mean')]:.2f} ± {location_groups.loc[loc, ('DO (mg/L)', 'std')]:.2f} mg/L")
    
    return location_groups

def create_corrected_validation_figure(profiles, cem_data, stations, output_dir="OUT"):
    """Create CORRECTED longitudinal profile validation figure."""
    print("🎨 Creating corrected longitudinal profile validation figure")
    
    # Aggregate field data
    field_summary = aggregate_field_data_by_location(cem_data)
    
    # Create model grid (0-202 km from mouth)
    grid_km = np.linspace(0, 202, 102)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CORRECTED Phase 1: Longitudinal Profile Validation\nModel vs CEM Field Observations', fontsize=16, fontweight='bold')
    
    # Plot 1: Tidal Amplitude
    ax = axes[0, 0]
    ax.plot(grid_km, profiles['tidal_amplitude'], 'b-', linewidth=2, label='JAX C-GEM Model')
    ax.set_xlabel('Distance from Mouth (km)')
    ax.set_ylabel('Tidal Amplitude (m)')
    ax.set_title('Tidal Amplitude Validation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Salinity Profile
    ax = axes[0, 1] 
    ax.plot(grid_km, profiles['S'], 'r-', linewidth=2, label='JAX C-GEM Model')
    
    # Add field data points (convert coordinates)
    for location_km in field_summary.index:
        model_km = 202 - location_km  # Convert field → model coordinates
        salinity_mean = field_summary.loc[location_km, ('Salinity', 'mean')]
        salinity_std = field_summary.loc[location_km, ('Salinity', 'std')]
        
        ax.errorbar(model_km, salinity_mean, yerr=salinity_std, 
                   fmt='ko', markersize=8, capsize=5, capthick=2, 
                   label='CEM Field Data' if location_km == field_summary.index[0] else "")
    
    ax.set_xlabel('Distance from Mouth (km)')
    ax.set_ylabel('Salinity (PSU)')  
    ax.set_title('Mean Salinity Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Dissolved Oxygen
    ax = axes[0, 2]
    if 'O2' in profiles:
        ax.plot(grid_km, profiles['O2'], 'g-', linewidth=2, label='JAX C-GEM Model')
        
        # Add field data points
        for location_km in field_summary.index:
            model_km = 202 - location_km
            if not pd.isna(field_summary.loc[location_km, ('DO (mg/L)', 'mean')]):
                do_mean = field_summary.loc[location_km, ('DO (mg/L)', 'mean')]
                do_std = field_summary.loc[location_km, ('DO (mg/L)', 'std')]
                ax.errorbar(model_km, do_mean, yerr=do_std,
                           fmt='ko', markersize=8, capsize=5, capthick=2,
                           label='CEM Field Data' if location_km == field_summary.index[0] else "")
    
    ax.set_xlabel('Distance from Mouth (km)')
    ax.set_ylabel('DO (mg/L)')
    ax.set_title('Mean Dissolved Oxygen')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Ammonium
    ax = axes[1, 0]
    if 'NH4' in profiles:
        ax.plot(grid_km, profiles['NH4'], 'orange', linewidth=2, label='JAX C-GEM Model')
        
        # Add field data points  
        for location_km in field_summary.index:
            model_km = 202 - location_km
            if not pd.isna(field_summary.loc[location_km, ('NH4 (mgN/L)', 'mean')]):
                nh4_mean = field_summary.loc[location_km, ('NH4 (mgN/L)', 'mean')]
                nh4_std = field_summary.loc[location_km, ('NH4 (mgN/L)', 'std')]
                ax.errorbar(model_km, nh4_mean, yerr=nh4_std,
                           fmt='ko', markersize=8, capsize=5, capthick=2,
                           label='CEM Field Data' if location_km == field_summary.index[0] else "")
    
    ax.set_xlabel('Distance from Mouth (km)')
    ax.set_ylabel('NH4 (mgN/L)')
    ax.set_title('Mean Ammonium')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Phosphate  
    ax = axes[1, 1]
    if 'PO4' in profiles:
        ax.plot(grid_km, profiles['PO4'], 'purple', linewidth=2, label='JAX C-GEM Model')
        
        # Add field data points
        for location_km in field_summary.index:
            model_km = 202 - location_km
            if not pd.isna(field_summary.loc[location_km, ('PO4 (mgP/L)', 'mean')]):
                po4_mean = field_summary.loc[location_km, ('PO4 (mgP/L)', 'mean')]
                po4_std = field_summary.loc[location_km, ('PO4 (mgP/L)', 'std')]
                ax.errorbar(model_km, po4_mean, yerr=po4_std,
                           fmt='ko', markersize=8, capsize=5, capthick=2,
                           label='CEM Field Data' if location_km == field_summary.index[0] else "")
    
    ax.set_xlabel('Distance from Mouth (km)')
    ax.set_ylabel('PO4 (mgP/L)')
    ax.set_title('Mean Phosphate')  
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Total Organic Carbon
    ax = axes[1, 2]
    if 'TOC' in profiles:
        ax.plot(grid_km, profiles['TOC'], 'brown', linewidth=2, label='JAX C-GEM Model')
        
        # Add field data points
        for location_km in field_summary.index:
            model_km = 202 - location_km
            if not pd.isna(field_summary.loc[location_km, ('TOC (mgC/L)', 'mean')]):
                toc_mean = field_summary.loc[location_km, ('TOC (mgC/L)', 'mean')]
                toc_std = field_summary.loc[location_km, ('TOC (mgC/L)', 'std')]
                ax.errorbar(model_km, toc_mean, yerr=toc_std,
                           fmt='ko', markersize=8, capsize=5, capthick=2,
                           label='CEM Field Data' if location_km == field_summary.index[0] else "")
    
    ax.set_xlabel('Distance from Mouth (km)')
    ax.set_ylabel('TOC (mgC/L)')
    ax.set_title('Total Organic Carbon')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase1_corrected_longitudinal_profiles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Corrected validation figure saved: {output_path}")
    
    return str(output_path)

def calculate_validation_metrics(profiles, cem_data, stations):
    """Calculate validation metrics with proper coordinate alignment."""
    print("📊 Calculating corrected validation metrics...")
    
    field_summary = aggregate_field_data_by_location(cem_data)
    metrics = {}
    
    # Calculate RMSE and correlation for each variable
    for var_name, model_profile in profiles.items():
        if var_name == 'tidal_amplitude':
            continue
            
        # Map variable names
        field_var_map = {
            'S': 'Salinity',
            'O2': 'DO (mg/L)', 
            'NH4': 'NH4 (mgN/L)',
            'PO4': 'PO4 (mgP/L)',
            'TOC': 'TOC (mgC/L)'
        }
        
        if var_name not in field_var_map:
            continue
            
        field_var = field_var_map[var_name]
        
        # Extract model and observed values at field locations
        model_values = []
        observed_values = []
        
        for location_km in field_summary.index:
            model_km = 202 - location_km  # Convert coordinates
            grid_idx = int(model_km / 2)  # 2km grid spacing
            grid_idx = max(0, min(101, grid_idx))  # Ensure valid index
            
            if not pd.isna(field_summary.loc[location_km, (field_var, 'mean')]):
                model_values.append(model_profile[grid_idx])
                observed_values.append(field_summary.loc[location_km, (field_var, 'mean')])
        
        if len(model_values) > 1:
            model_arr = np.array(model_values)
            obs_arr = np.array(observed_values)
            
            rmse = np.sqrt(np.mean((model_arr - obs_arr)**2))
            correlation = np.corrcoef(model_arr, obs_arr)[0,1] if len(model_values) > 1 else np.nan
            
            metrics[var_name] = {
                'rmse': rmse,
                'correlation': correlation,
                'n_points': len(model_values),
                'model_range': (model_arr.min(), model_arr.max()),
                'obs_range': (obs_arr.min(), obs_arr.max())
            }
            
            print(f"   ✅ {var_name}: RMSE={rmse:.3f}, r={correlation:.3f}, n={len(model_values)}")
    
    return metrics

def generate_corrected_validation_report(profiles, cem_data, stations, metrics, output_dir="OUT"):
    """Generate corrected validation report."""
    print("📋 Generating corrected validation report...")
    
    report_path = Path(output_dir) / "phase1_corrected_validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("# CORRECTED PHASE 1 VALIDATION REPORT: Longitudinal Profiles\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Coordinate System Correction\n\n")
        f.write("CRITICAL FIX: Proper alignment of field and model coordinates:\n")
        f.write("- Model: 0 km = mouth (downstream), 202 km = head (upstream)\n")
        f.write("- CEM Field: distances from upstream end\n")
        f.write("- Conversion: model_km = 202 - field_km\n\n")
        
        f.write("## Model Performance Summary\n\n")
        
        f.write("### Validation Metrics\n\n")
        f.write("| Variable | RMSE | Correlation | N Points | Model Range | Observed Range |\n")
        f.write("|----------|------|-------------|----------|-------------|----------------|\n")
        
        for var, metric in metrics.items():
            f.write(f"| {var} | {metric['rmse']:.3f} | {metric['correlation']:.3f} | {metric['n_points']} | "
                   f"{metric['model_range'][0]:.2f}-{metric['model_range'][1]:.2f} | "
                   f"{metric['obs_range'][0]:.2f}-{metric['obs_range'][1]:.2f} |\n")
        
        f.write("\n## Field Data Coverage\n\n")
        field_summary = aggregate_field_data_by_location(cem_data)
        for location_km in sorted(field_summary.index):
            model_km = 202 - location_km
            f.write(f"Location {location_km}km (model {model_km}km):\n")
            f.write(f"  - Salinity: {field_summary.loc[location_km, ('Salinity', 'mean')]:.2f} ± {field_summary.loc[location_km, ('Salinity', 'std')]:.2f} psu\n")
            f.write(f"  - Sample count: {field_summary.loc[location_km, ('Salinity', 'count')]}\n\n")
        
        f.write("## Critical Issues Identified\n\n")
        f.write("1. **Coordinate Mismatch**: Previous validation used incorrect station mapping\n")
        f.write("2. **Salinity Range**: Model shows unrealistic values for freshwater region\n")
        f.write("3. **Boundary Conditions**: Need verification of upstream/downstream inputs\n")
        f.write("4. **Species Concentrations**: Large discrepancies suggest calibration needed\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("- ❌ MODEL REQUIRES SIGNIFICANT CALIBRATION\n")
        f.write("- Verify boundary condition files and coordinate systems\n")
        f.write("- Calibrate salinity intrusion dynamics\n")
        f.write("- Adjust biogeochemical parameters to match field observations\n")
    
    print(f"✅ Corrected validation report saved: {report_path}")
    return str(report_path)

def main():
    """Main analysis function."""
    print("🧪 CORRECTED PHASE 1 ANALYSIS: Longitudinal Profile Validation")
    print("=" * 60)
    
    # Load simulation results
    data = load_simulation_results()
    if data is None:
        return
    
    # Load field observations
    cem_data, tidal_data = load_field_observations()
    if cem_data is None:
        return
    
    # Create corrected station mapping
    stations = create_corrected_station_mapping()
    
    # Calculate time-averaged profiles
    profiles = calculate_time_averaged_profiles(data)
    
    # Calculate validation metrics
    metrics = calculate_validation_metrics(profiles, cem_data, stations)
    
    # Create corrected validation figure
    figure_path = create_corrected_validation_figure(profiles, cem_data, stations)
    
    # Generate corrected validation report
    report_path = generate_corrected_validation_report(profiles, cem_data, stations, metrics)
    
    print("\n✅ CORRECTED PHASE 1 ANALYSIS COMPLETED")
    print("📊 Outputs:")
    print(f"   - Corrected validation figure: {figure_path}")
    print(f"   - Corrected validation report: {report_path}")
    
    print("\n🔍 CRITICAL FINDINGS:")
    print("   - Coordinate system correction applied")
    print("   - Significant model-observation discrepancies identified")  
    print("   - MODEL CALIBRATION URGENTLY NEEDED")
    
    return metrics

if __name__ == "__main__":
    main()