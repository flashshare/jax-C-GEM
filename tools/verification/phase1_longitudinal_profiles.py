#!/usr/bin/env python
"""
Phase 1: Longitudinal Profile Validation

This script validates model results against CEM spatial data (2017-2018) covering
locations from 2km to 158km from the estuarine mouth.

Key Features:
- Proper unit conversion from model mmol/m³ to field mg/L units
- Time-averaged profiles over simulation period
- Station-based analysis at CEM measurement locations
- Statistical validation metrics (RMSE, R², MAPE)
- Publication-quality visualization

Field Data: CEM 2017-2018.csv (318 observations across 6 stations)
Species: NH4, NO3, PO4, TOC, DO, Salinity, TSS

Author: Nguyen Truong An
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Unit conversion factors: mmol/m³ → mg/L
UNIT_CONVERSION_FACTORS = {
    'NH4': 14.0 / 1000.0,    # mmol NH4-N/m³ → mg N/L
    'NO3': 14.0 / 1000.0,    # mmol NO3-N/m³ → mg N/L
    'PO4': 31.0 / 1000.0,    # mmol PO4-P/m³ → mg P/L
    'TOC': 12.0 / 1000.0,    # mmol C/m³ → mg C/L
    'O2': 32.0 / 1000.0,     # mmol O2/m³ → mg O2/L
    'S': 1.0,                # Salinity (dimensionless)
    'SPM': 1.0,              # mg/L (already correct units)
}

def load_model_results(results_dir="OUT"):
    """Load model results in NPZ format (preferred) or CSV fallback."""
    npz_file = Path(results_dir) / "simulation_results.npz"
    
    if npz_file.exists():
        print(f"📊 Loading model results from {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        
        # Create spatial grid from model config
        # EL = 202000 m, DELXI = 2000 m → M = 102 cells, locations from 1-201 km
        M = 102
        locations = np.linspace(1, 201, M)  # km from mouth
        
        # Extract time array - use 'time' key
        time_array = data['time'] if 'time' in data else np.arange(data['NH4'].shape[0])
        
        # Extract water levels if available (for tidal amplitude calculation)
        water_levels = None
        if 'H' in data:
            water_levels = data['H']  # Shape should be (time, space)
            print(f"   ✓ Water levels: {water_levels.shape}")
        
        # Extract species concentrations
        model_data = {}
        for species in UNIT_CONVERSION_FACTORS.keys():
            if species in data:
                model_data[species] = data[species]
                print(f"   ✓ {species}: {model_data[species].shape}")
        
        return {
            'locations': locations,
            'time_array': time_array,
            'species_data': model_data,
            'water_levels': water_levels
        }
    else:
        raise FileNotFoundError(f"Model results not found in {results_dir}")

def load_cem_observations():
    """Load CEM field observations."""
    cem_file = Path("INPUT/Calibration/CEM_2017-2018.csv")
    
    if not cem_file.exists():
        raise FileNotFoundError(f"CEM data not found: {cem_file}")
    
    print(f"📋 Loading CEM observations from {cem_file}")
    cem_data = pd.read_csv(cem_file)
    
    # Parse dates
    cem_data['Date'] = pd.to_datetime(cem_data['Date'], format='%m/%d/%Y')
    
    # Define species mapping (CEM columns → model species)
    species_mapping = {
        'NH4 (mgN/L)': 'NH4',
        'PO4 (mgP/L)': 'PO4', 
        'TOC (mgC/L)': 'TOC',
        'DO (mg/L)': 'O2',
        'Salinity': 'S',
        'TSS (mg/L)': 'SPM'
    }
    
    # Process observations by location
    cem_processed = []
    for _, row in cem_data.iterrows():
        if pd.notna(row['Location']):
            for cem_col, species in species_mapping.items():
                if cem_col in cem_data.columns and pd.notna(row[cem_col]):
                    cem_processed.append({
                        'location': row['Location'],
                        'species': species,
                        'concentration': row[cem_col],
                        'date': row['Date'],
                        'site': row['Site']
                    })
    
    cem_df = pd.DataFrame(cem_processed)
    print(f"   ✓ Processed {len(cem_df)} observations")
    print(f"   ✓ Locations: {sorted(cem_df['location'].unique())}")
    print(f"   ✓ Species: {sorted(cem_df['species'].unique())}")
    
    return cem_df

def calculate_time_averaged_profiles(model_results, warmup_days=100):
    """Calculate time-averaged spatial profiles from model results."""
    print(f"📈 Calculating time-averaged profiles (excluding {warmup_days} warmup days)")
    
    locations = model_results['locations']
    time_array = model_results['time_array']
    
    # Skip warmup period
    warmup_steps = warmup_days * 24  # hours
    if len(time_array) > warmup_steps:
        analysis_start = warmup_steps
        print(f"   ✓ Using data from step {analysis_start} to {len(time_array)}")
    else:
        analysis_start = 0
        print(f"   ⚠️  Short simulation: using all {len(time_array)} steps")
    
    profiles = {}
    for species, data in model_results['species_data'].items():
        if species in UNIT_CONVERSION_FACTORS:
            # Time average over analysis period
            avg_profile = np.mean(data[analysis_start:, :], axis=0)
            
            # Convert units: mmol/m³ → mg/L
            conversion_factor = UNIT_CONVERSION_FACTORS[species]
            profiles[species] = avg_profile * conversion_factor
            
            print(f"   ✓ {species}: {avg_profile.min():.3f} - {avg_profile.max():.3f} mmol/m³")
            print(f"     → {profiles[species].min():.3f} - {profiles[species].max():.3f} mg/L")
    
    return {'locations': locations, 'profiles': profiles}

def calculate_tidal_amplitude_profile(model_results, warmup_days=100):
    """Calculate tidal amplitude profile from mouth to upstream.
    
    Args:
        model_results: Dictionary containing time series data
        warmup_days: Days to exclude from analysis for model spinup
        
    Returns:
        Dictionary with locations and tidal_amplitude arrays, or None if no water level data
    """
    print(f"🌊 Calculating tidal amplitude profile")
    
    # Check if water levels are available in the model results
    if 'water_levels' not in model_results or model_results['water_levels'] is None:
        print("   ⚠️ Water level data not available in model results")
        return None
    
    locations = model_results['locations']
    time_array = model_results['time_array']
    water_levels = model_results['water_levels']  # Shape: (time, space)
    
    # Skip warmup period
    warmup_steps = warmup_days * 24  # hours
    if len(time_array) > warmup_steps:
        analysis_start = warmup_steps
        print(f"   ✓ Using data from step {analysis_start} to {len(time_array)}")
    else:
        analysis_start = 0
        print(f"   ⚠️  Short simulation: using all {len(time_array)} steps")
    
    print(f"   Water levels shape: {water_levels.shape}")
    
    if water_levels.shape[0] > analysis_start:
        # Use steady-state period for tidal analysis
        steady_water_levels = water_levels[analysis_start:]
        
        # Calculate tidal range (max - min) at each location
        tidal_amplitude = np.max(steady_water_levels, axis=0) - np.min(steady_water_levels, axis=0)
        
        print(f"   ✓ Tidal amplitude range: {tidal_amplitude.min():.3f} - {tidal_amplitude.max():.3f} m")
        
        return {
            'locations': locations,
            'tidal_amplitude': tidal_amplitude
        }
    
    return None

def aggregate_cem_by_location(cem_data):
    """Aggregate CEM observations by location and species."""
    print("📊 Aggregating CEM data by location")
    
    cem_aggregated = cem_data.groupby(['location', 'species'])['concentration'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    print(f"   ✓ {len(cem_aggregated)} location-species combinations")
    
    return cem_aggregated

def interpolate_model_to_stations(model_profiles, station_locations):
    """Interpolate model profiles to field station locations."""
    model_locations = model_profiles['locations']
    interpolated = {}
    
    for species, profile in model_profiles['profiles'].items():
        interpolated[species] = np.interp(station_locations, model_locations, profile)
    
    return interpolated

def calculate_validation_metrics(model_values, field_values):
    """Calculate validation metrics between model and field data."""
    # Filter valid data points
    valid_mask = ~(np.isnan(model_values) | np.isnan(field_values))
    
    if np.sum(valid_mask) < 2:
        return {
            'rmse': np.nan,
            'r2': np.nan,
            'mape': np.nan,
            'mean_error': np.nan,
            'n_points': 0
        }
    
    model_valid = model_values[valid_mask]
    field_valid = field_values[valid_mask]
    
    # RMSE
    rmse = np.sqrt(np.mean((model_valid - field_valid) ** 2))
    
    # R² - Use numpy correlation coefficient instead
    try:
        correlation_matrix = np.corrcoef(model_valid, field_valid)
        r2 = correlation_matrix[0, 1] ** 2
        if np.isnan(r2):
            r2 = 0.0
    except:
        r2 = np.nan
    
    # MAPE (%)
    mape = np.mean(np.abs((field_valid - model_valid) / field_valid)) * 100
    
    # Mean error
    mean_error = np.mean(model_valid - field_valid)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'mean_error': mean_error,
        'n_points': np.sum(valid_mask)
    }

def create_validation_figure(model_profiles, cem_aggregated, interpolated_model, 
                           tidal_amplitude=None, output_dir="OUT"):
    """Create comprehensive validation figure."""
    print("🎨 Creating longitudinal profile validation figure")
    
    # Get common species
    model_species = set(model_profiles['profiles'].keys())
    field_species = set(cem_aggregated['species'].unique())
    common_species = model_species.intersection(field_species)
    
    if not common_species:
        print("   ⚠️  No common species found for validation")
        return
    
    # Create figure with space for tidal amplitude if available
    has_tidal = tidal_amplitude is not None
    n_total_panels = len(common_species) + (1 if has_tidal else 0)
    
    # Adjust subplot layout for optimal arrangement
    if n_total_panels <= 4:
        ncols = 2
        nrows = (n_total_panels + 1) // 2
    elif n_total_panels <= 6:
        ncols = 3
        nrows = 2
    else:
        ncols = 3
        nrows = (n_total_panels + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten() if n_total_panels > 1 else [axes]
    
    validation_results = {}
    
    for i, species in enumerate(sorted(common_species)):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        # Plot model profile
        locations = model_profiles['locations']
        model_profile = model_profiles['profiles'][species]
        ax.plot(locations, model_profile, 'b-', linewidth=2, label='Model', alpha=0.8)
        
        # Plot field observations
        species_data = cem_aggregated[cem_aggregated['species'] == species]
        if not species_data.empty:
            ax.errorbar(species_data['location'], species_data['mean'], 
                       yerr=species_data['std'], fmt='ro', markersize=8,
                       capsize=5, capthick=2, label='CEM Observations', alpha=0.8)
        
        # Calculate validation metrics
        station_locs = species_data['location'].values
        station_model = interpolated_model.get(species, np.array([]))
        station_field = species_data['mean'].values
        
        if len(station_locs) > 0 and len(station_model) > 0:
            # Interpolate model to exact station locations
            model_at_stations = np.interp(station_locs, locations, model_profile)
            metrics = calculate_validation_metrics(model_at_stations, station_field)
            validation_results[species] = metrics
            
            # Add metrics to plot
            ax.text(0.05, 0.95, f'R² = {metrics["r2"]:.3f}\nRMSE = {metrics["rmse"]:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Distance from Mouth (km)')
        ax.set_ylabel(f'{species} Concentration (mg/L)')
        ax.set_title(f'{species} Longitudinal Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable limits
        if not species_data.empty:
            ax.set_xlim(0, max(locations.max(), species_data['location'].max()) * 1.1)
    
    # Add tidal amplitude panel if data is available
    if has_tidal and len(common_species) < len(axes):
        tidal_ax = axes[len(common_species)]
        tidal_locations = tidal_amplitude['locations']
        tidal_range = tidal_amplitude['tidal_amplitude']
        
        # Plot tidal amplitude profile
        tidal_ax.plot(tidal_locations, tidal_range, 'b-', linewidth=2.5, 
                     label='Model Tidal Amplitude', alpha=0.8)
        
        # Formatting
        tidal_ax.set_xlabel('Distance from Mouth (km)')
        tidal_ax.set_ylabel('Tidal Amplitude (m)')
        tidal_ax.set_title('Tidal Amplitude Profile (Mouth → Upstream)')
        tidal_ax.legend()
        tidal_ax.grid(True, alpha=0.3)
        tidal_ax.set_xlim(0, tidal_locations.max() * 1.05)
        
        # Add informative text
        max_tidal = np.max(tidal_range)
        min_tidal = np.min(tidal_range)
        tidal_ax.text(0.05, 0.95, 
                     f'Max: {max_tidal:.2f} m\nMin: {min_tidal:.2f} m\nDecay: {max_tidal/min_tidal:.1f}x', 
                     transform=tidal_ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Remove empty subplots
    total_used = len(common_species) + (1 if has_tidal else 0)
    for i in range(total_used, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase1_longitudinal_profiles.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Figure saved: {output_path}")
    
    # Print validation summary
    print("\n📊 VALIDATION SUMMARY - Longitudinal Profiles")
    print("=" * 60)
    for species, metrics in validation_results.items():
        print(f"{species:>6}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}, "
              f"MAPE = {metrics['mape']:.1f}%, n = {metrics['n_points']}")
    
    return validation_results

def main():
    """Main validation workflow."""
    print("🔬 Phase 1: Longitudinal Profile Validation")
    print("=" * 50)
    
    try:
        # Load data
        model_results = load_model_results()
        cem_data = load_cem_observations()
        
        # Calculate time-averaged profiles
        model_profiles = calculate_time_averaged_profiles(model_results)
        
        # Calculate tidal amplitude profile
        tidal_amplitude = calculate_tidal_amplitude_profile(model_results)
        
        # Aggregate field data
        cem_aggregated = aggregate_cem_by_location(cem_data)
        
        # Interpolate model to station locations
        station_locations = cem_aggregated['location'].unique()
        interpolated_model = interpolate_model_to_stations(model_profiles, station_locations)
        
        # Create validation figure with tidal amplitude
        validation_results = create_validation_figure(model_profiles, cem_aggregated, 
                                                    interpolated_model, tidal_amplitude)
        
        print("\n✅ Phase 1 validation completed successfully!")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Error in Phase 1 validation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()