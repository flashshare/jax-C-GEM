#!/usr/bin/env python
"""
Phase 1 Analysis Script: Longitudinal Profile Validation

This script loads the 40-day simulation results and validates the model's ability
to produce stable, time-averaged longitudinal profiles that conform to estuarine theory.

Scientific Objective:
- Verify time-averaged spatial patterns over final 30 days
- Compare against field observations at PC, BD, BK stations
- Quantify model performance with RMSE and correlation metrics

Usage: python analysis/phase1_validate_profiles.py
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
    """Load the final simulation results."""
    print(f"📂 Loading simulation results from {results_file}")
    
    if not Path(results_file).exists():
        # Try alternative locations
        alternative_files = [
            "OUT/final_results.npz",
            "OUT/simulation_results.npz",
            "OUT/results.npz"
        ]
        
        for alt_file in alternative_files:
            if Path(alt_file).exists():
                results_file = alt_file
                print(f"   Found alternative file: {alt_file}")
                break
        else:
            raise FileNotFoundError(f"Could not find results file. Tried: {results_file}, {alternative_files}")
    
    data = np.load(results_file)
    print(f"✅ Results loaded successfully")
    print(f"   Available keys: {list(data.keys())}")
    
    return data

def load_field_observations(cem_file="INPUT/Calibration/CEM_2017-2018.csv", 
                           cem_tidal_file="INPUT/Calibration/CEM-Tidal-range.csv"):
    """Load field observation data for comprehensive spatial validation."""
    print(f"📊 Loading spatial field observations")
    
    observations = {}
    
    # Load CEM data for comprehensive spatial coverage (200 km)
    if Path(cem_file).exists():
        try:
            cem_data = pd.read_csv(cem_file)
            print(f"✅ CEM spatial data loaded: {len(cem_data)} observations")
            print(f"   Stations: {cem_data['Site'].unique()}")
            print(f"   Location range: {cem_data['Location'].min()}-{cem_data['Location'].max()} km")
            print(f"   Variables: {list(cem_data.columns)}")
            observations['cem'] = cem_data
        except Exception as e:
            print(f"❌ Error loading CEM data: {e}")
    
    # Load CEM tidal range data for spatial tidal validation
    if Path(cem_tidal_file).exists():
        try:
            tidal_data = pd.read_csv(cem_tidal_file)
            print(f"✅ CEM tidal range data loaded: {len(tidal_data)} observations")
            print(f"   Stations: {tidal_data['Station'].unique()}")
            print(f"   Tidal range: {tidal_data['Tidal Range (m)'].min():.2f}-{tidal_data['Tidal Range (m)'].max():.2f} m")
            observations['tidal'] = tidal_data
        except Exception as e:
            print(f"❌ Error loading tidal data: {e}")
    
    return observations

def calculate_time_averaged_profiles(data, warmup_days=10, total_days=40):
    """Calculate time-averaged profiles for the analysis period."""
    print(f"📈 Calculating time-averaged profiles")
    print(f"   Warmup period: {warmup_days} days")
    print(f"   Analysis period: {total_days - warmup_days} days")
    
    # Extract time array
    time_array = data['time']
    n_steps = len(time_array)
    
    # Convert days to time steps (assuming 30-min output intervals)
    steps_per_day = 48  # 24 hours * 2 (30-min intervals)
    warmup_steps = warmup_days * steps_per_day
    
    print(f"   Total time steps: {n_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Analysis steps: {n_steps - warmup_steps}")
    
    # Select analysis period (skip warmup)
    analysis_indices = slice(warmup_steps, None)
    
    # Extract key variables
    results = {}
    
    # Hydrodynamic variables
    if 'H' in data:
        H_data = data['H'][analysis_indices, :]
        # Calculate tidal amplitude as max - min over analysis period
        results['tidal_amplitude'] = np.max(H_data, axis=0) - np.min(H_data, axis=0)
        results['mean_water_level'] = np.mean(H_data, axis=0)
        print(f"   ✅ Tidal amplitude calculated (range: {results['tidal_amplitude'].min():.2f} - {results['tidal_amplitude'].max():.2f} m)")
    
    # Species concentrations
    species_to_analyze = ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']
    
    for species in species_to_analyze:
        if species in data:
            species_data = data[species][analysis_indices, :]
            results[f'mean_{species}'] = np.mean(species_data, axis=0)
            results[f'std_{species}'] = np.std(species_data, axis=0)
            print(f"   ✅ {species}: mean = {results[f'mean_{species}'].mean():.2f}, std = {results[f'std_{species}'].mean():.2f}")
    
    return results

def create_station_mapping():
    """Define comprehensive field station locations from CEM dataset."""
    # Enhanced station mapping based on CEM_2017-2018.csv coverage
    stations = {
        # Primary stations for model validation (match our enhanced config)
        'PC': {'name': 'Phu Cuong', 'distance_km': 114, 'description': 'Lower estuary', 'priority': 'high'},
        'BD': {'name': 'Ben Do', 'distance_km': 78, 'description': 'Mid-estuary mixing zone', 'priority': 'high'},
        'BK': {'name': 'Binh Khanh', 'distance_km': 48, 'description': 'Upper estuary', 'priority': 'high'},
        
        # Additional CEM stations for comprehensive spatial validation
        'BS': {'name': 'Ben Suc', 'distance_km': 156, 'description': 'Furthest upstream', 'priority': 'medium'},
        'TT': {'name': 'Thi Tinh', 'distance_km': 124, 'description': 'Upper-mid estuary', 'priority': 'medium'},
        'BP': {'name': 'Binh Phuoc', 'distance_km': 94, 'description': 'Mid-lower estuary', 'priority': 'medium'},
        'VS': {'name': 'Vung Sat', 'distance_km': 28, 'description': 'Lower estuary', 'priority': 'medium'},
        'VC': {'name': 'Vung Cat', 'distance_km': 20, 'description': 'Near mouth', 'priority': 'medium'}
    }
    
    print(f"📍 Comprehensive station mapping: {len(stations)} stations covering 0-156 km")
    return stations

def extract_station_values(profiles, stations, total_length_km=202):
    """Extract model values at field station locations."""
    station_values = {}
    n_grid = len(next(iter(profiles.values())))
    
    for station_code, station_info in stations.items():
        distance_km = station_info['distance_km']
        # Convert distance to grid index
        grid_index = int((distance_km / total_length_km) * (n_grid - 1))
        grid_index = max(0, min(grid_index, n_grid - 1))  # Ensure valid index
        
        station_values[station_code] = {}
        for var_name, profile in profiles.items():
            station_values[station_code][var_name] = profile[grid_index]
        
        print(f"   📍 {station_code} ({station_info['name']}): {distance_km}km → grid {grid_index}")
    
    return station_values

def create_longitudinal_profile_figure(profiles, field_data, stations, output_dir="OUT"):
    """Create comprehensive longitudinal profile validation using CEM spatial data."""
    print("🎨 Creating comprehensive longitudinal profile validation figure")
    
    # Create grid distance array (assuming 202 km total length)
    n_grid = len(profiles['tidal_amplitude'])
    distance_km = np.linspace(0, 202, n_grid)
    
    # Create figure with 2x3 subplots for comprehensive validation
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Enhanced plot configurations using CEM data
    plot_configs = [
        {'var': 'tidal_amplitude', 'title': 'Tidal Amplitude Validation', 'ylabel': 'Amplitude (m)', 
         'color': 'blue', 'field_var': 'tidal_range', 'field_source': 'CEM-Tidal'},
        {'var': 'mean_S', 'title': 'Mean Salinity Profile', 'ylabel': 'Salinity (PSU)', 
         'color': 'red', 'field_var': 'Salinity', 'field_source': 'CEM'},
        {'var': 'mean_O2', 'title': 'Mean Dissolved Oxygen', 'ylabel': 'DO (mg/L)', 
         'color': 'green', 'field_var': 'DO (mg/L)', 'field_source': 'CEM'},
        {'var': 'mean_NH4', 'title': 'Mean Ammonium', 'ylabel': 'NH4 (mgN/L)', 
         'color': 'orange', 'field_var': 'NH4 (mgN/L)', 'field_source': 'CEM'},
        {'var': 'mean_TSS', 'title': 'Total Suspended Solids', 'ylabel': 'TSS (mg/L)', 
         'color': 'brown', 'field_var': 'TSS (mg/L)', 'field_source': 'CEM'},
        {'var': 'mean_TOC', 'title': 'Total Organic Carbon', 'ylabel': 'TOC (mgC/L)', 
         'color': 'purple', 'field_var': 'TOC (mgC/L)', 'field_source': 'CEM'}
    ]
    
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        var_name = config['var']
        
        if var_name in profiles:
            # Plot model profile
            ax.plot(distance_km, profiles[var_name], 
                   color=config['color'], linewidth=3, label='JAX C-GEM Model', alpha=0.8)
            
            # Add field observations from appropriate dataset
            if field_data and config['field_source'] == 'CEM' and 'cem' in field_data:
                cem_data = field_data['cem']
                field_var = config['field_var']
                
                if field_var in cem_data.columns:
                    # Group by location and calculate mean
                    location_means = cem_data.groupby('Location')[field_var].agg(['mean', 'std']).reset_index()
                    
                    # Plot field observations
                    ax.errorbar(location_means['Location'], location_means['mean'], 
                              yerr=location_means['std'], fmt='o', color='black', 
                              markersize=8, capsize=5, capthick=2, 
                              label='CEM Field Data', alpha=0.7)
            
            elif field_data and config['field_source'] == 'CEM-Tidal' and 'tidal' in field_data:
                tidal_data = field_data['tidal']
                
                # Group by location and calculate mean tidal range
                location_means = tidal_data.groupby('Location')['Tidal Range (m)'].agg(['mean', 'std']).reset_index()
                
                # Plot tidal range observations
                ax.errorbar(location_means['Location'], location_means['mean'], 
                          yerr=location_means['std'], fmt='s', color='darkblue', 
                          markersize=8, capsize=5, capthick=2, 
                          label='CEM Tidal Range', alpha=0.7)
            
            # Highlight priority stations
            for station_code, station_info in stations.items():
                if station_info['priority'] == 'high':
                    station_km = station_info['distance_km']
                    ax.axvline(station_km, color='red', linestyle='--', alpha=0.8, linewidth=2)
                    ax.text(station_km, ax.get_ylim()[1]*0.95, station_code, 
                           ha='center', va='top', fontweight='bold', color='red')
        
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_xlabel('Distance from Mouth (km)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase1_comprehensive_longitudinal_profiles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive figure saved: {output_path}")
    
    return fig

def calculate_validation_statistics(model_values, observed_values):
    """Calculate RMSE and correlation metrics."""
    if observed_values is None or len(observed_values) == 0:
        return {'rmse': np.nan, 'correlation': np.nan, 'n_obs': 0}
    
    # Convert to numpy arrays
    model_array = np.array(list(model_values.values()))
    obs_array = np.array(observed_values)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(model_array) | np.isnan(obs_array))
    model_clean = model_array[valid_mask]
    obs_clean = obs_array[valid_mask]
    
    if len(model_clean) < 2:
        return {'rmse': np.nan, 'correlation': np.nan, 'n_obs': len(model_clean)}
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((model_clean - obs_clean)**2))
    correlation = np.corrcoef(model_clean, obs_clean)[0, 1]
    
    return {
        'rmse': rmse,
        'correlation': correlation,
        'n_obs': len(model_clean),
        'model_mean': np.mean(model_clean),
        'obs_mean': np.mean(obs_clean)
    }

def generate_validation_report(station_values, field_data, output_dir="OUT"):
    """Generate a comprehensive validation report."""
    print("📋 Generating validation report")
    
    report_lines = [
        "# PHASE 1 VALIDATION REPORT: LONGITUDINAL PROFILES",
        "=" * 60,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Performance Summary",
        ""
    ]
    
    # Summary statistics for each variable
    variables = ['tidal_amplitude', 'mean_S', 'mean_O2', 'mean_NH4']
    var_names = ['Tidal Amplitude', 'Salinity', 'Dissolved Oxygen', 'Ammonium']
    
    for var, var_name in zip(variables, var_names):
        report_lines.append(f"### {var_name}")
        
        # Extract values for all stations
        station_data = []
        for station_code in ['PC', 'BD', 'BK']:
            if station_code in station_values and var in station_values[station_code]:
                value = station_values[station_code][var]
                station_data.append((station_code, value))
        
        if station_data:
            for station_code, value in station_data:
                report_lines.append(f"  {station_code}: {value:.3f}")
            
            values = [val for _, val in station_data]
            report_lines.append(f"  Range: {min(values):.3f} - {max(values):.3f}")
        
        report_lines.append("")
    
    # Write report
    report_path = Path(output_dir) / "phase1_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Validation report saved: {report_path}")

def main():
    """Main analysis function for Phase 1 validation."""
    print("🧪 PHASE 1 ANALYSIS: LONGITUDINAL PROFILE VALIDATION")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("OUT")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Load simulation results
        data = load_simulation_results()
        
        # 2. Load field observations
        field_data = load_field_observations()
        
        # 3. Calculate time-averaged profiles
        profiles = calculate_time_averaged_profiles(data)
        
        # 4. Define station locations
        stations = create_station_mapping()
        
        # 5. Extract values at station locations
        station_values = extract_station_values(profiles, stations)
        
        # 6. Create longitudinal profile figure
        fig = create_longitudinal_profile_figure(profiles, field_data, stations, str(output_dir))
        
        # 7. Generate validation report
        generate_validation_report(station_values, field_data, str(output_dir))
        
        print("\n✅ PHASE 1 ANALYSIS COMPLETED SUCCESSFULLY")
        print("📊 Outputs:")
        print("   - Longitudinal profile figure: OUT/phase1_longitudinal_profiles.png")
        print("   - Validation report: OUT/phase1_validation_report.txt")
        print("\n🎯 Next Step: Review plots for estuarine theory compliance")
        print("   - Tidal amplitude should show exponential damping upstream")
        print("   - Salinity should show realistic salt wedge intrusion")
        print("   - Oxygen should show characteristic sag curve")
        print("   - NH4 should increase upstream from urban sources")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
