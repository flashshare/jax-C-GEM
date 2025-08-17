#!/usr/bin/env python
"""
Phase 2: Tidal Dynamics Validation

This script validates model hydrodynamics against SIHYMECC tidal range observations
at three key stations: PC (86km), BD (130km), BK (156km) from the estuarine mouth.

Key Features:
- Tidal range analysis at 3 stations along the estuary
- Statistical comparison of modeled vs observed tidal ranges
- Spatial tidal amplification patterns
- Time series validation with field measurements

Field Data: SIHYMECC_Tidal-range2017-2018.csv (43 measurement days)
Stations: PC=86km, BD=130km, BK=156km from mouth

Author: Nguyen Truong An
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Station locations (km from mouth)
TIDAL_STATIONS = {
    'PC': 86,   # Phú Cường
    'BD': 130,  # Bến Đình  
    'BK': 156   # Bến Kéo
}

def load_model_hydrodynamics(results_dir="OUT"):
    """Load model hydrodynamic results."""
    npz_file = Path(results_dir) / "simulation_results.npz"
    
    if npz_file.exists():
        print(f"🌊 Loading model hydrodynamics from {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        
        # Create spatial grid from model config
        # EL = 202000 m, DELXI = 2000 m → M = 102 cells, locations from 1-201 km
        M = 102
        locations = np.linspace(1, 201, M)  # km from mouth
        
        # Extract water levels and time array
        if 'H' in data and 'time' in data:
            time_array = data['time']
            water_levels = data['H']  # Shape: (n_times, n_cells)
            
            print(f"   ✓ Water levels: {water_levels.shape}")
            print(f"   ✓ Locations: {locations.min():.1f} - {locations.max():.1f} km")
            print(f"   ✓ Time range: {len(time_array)} hourly steps")
            
            return {
                'locations': locations,
                'time_array': time_array,
                'water_levels': water_levels
            }
        else:
            raise KeyError("Required hydrodynamic data (H, time) not found in results file")
    else:
        raise FileNotFoundError(f"Model results not found in {results_dir}")

def load_sihymecc_observations():
    """Load SIHYMECC tidal range observations."""
    sihymecc_file = Path("INPUT/Calibration/SIHYMECC_Tidal-range2017-2018.csv")
    
    if not sihymecc_file.exists():
        raise FileNotFoundError(f"SIHYMECC data not found: {sihymecc_file}")
    
    print(f"📊 Loading SIHYMECC tidal data from {sihymecc_file}")
    tidal_data = pd.read_csv(sihymecc_file)
    
    # Parse dates
    tidal_data['Date'] = pd.to_datetime(tidal_data['Day'], format='%m/%d/%Y')
    
    # Melt to long format
    tidal_long = tidal_data.melt(
        id_vars=['Date'], 
        value_vars=['PC', 'BD', 'BK'],
        var_name='Station', 
        value_name='TidalRange'
    ).dropna()
    
    # Add station locations
    tidal_long['Location'] = tidal_long['Station'].map(TIDAL_STATIONS)
    
    print(f"   ✓ Loaded {len(tidal_long)} tidal range observations")
    print(f"   ✓ Date range: {tidal_long['Date'].min()} to {tidal_long['Date'].max()}")
    print(f"   ✓ Stations: {sorted(tidal_long['Station'].unique())}")
    
    return tidal_long

def calculate_model_tidal_ranges(model_hydro, warmup_days=100):
    """Calculate daily tidal ranges from model water levels."""
    print(f"📈 Calculating model tidal ranges (excluding {warmup_days} warmup days)")
    
    locations = model_hydro['locations']
    time_array = model_hydro['time_array']
    water_levels = model_hydro['water_levels']
    
    # Skip warmup period
    warmup_steps = warmup_days * 24  # hours
    if len(time_array) > warmup_steps:
        analysis_start = warmup_steps
        water_levels_analysis = water_levels[analysis_start:, :]
        time_analysis = time_array[analysis_start:]
        print(f"   ✓ Using data from step {analysis_start} to {len(time_array)}")
    else:
        water_levels_analysis = water_levels
        time_analysis = time_array
        analysis_start = 0
        print(f"   ⚠️  Short simulation: using all {len(time_array)} steps")
    
    # Calculate daily tidal ranges
    n_days = len(time_analysis) // 24
    daily_ranges = []
    dates = []
    
    for day in range(n_days):
        day_start = day * 24
        day_end = (day + 1) * 24
        
        if day_end <= len(time_analysis):
            daily_levels = water_levels_analysis[day_start:day_end, :]
            daily_range = np.max(daily_levels, axis=0) - np.min(daily_levels, axis=0)
            daily_ranges.append(daily_range)
            
            # Calculate date (assuming hourly time steps starting from day 0)
            base_date = datetime(2017, 1, 1) + timedelta(days=warmup_days//24 + day)
            dates.append(base_date)
    
    if daily_ranges:
        daily_ranges_array = np.array(daily_ranges)  # Shape: (n_days, n_cells)
        print(f"   ✓ Calculated tidal ranges for {len(daily_ranges)} days")
        print(f"   ✓ Range statistics: {daily_ranges_array.mean():.2f} ± {daily_ranges_array.std():.2f} m")
        
        return {
            'locations': locations,
            'dates': dates,
            'tidal_ranges': daily_ranges_array
        }
    else:
        raise ValueError("No daily tidal ranges could be calculated")

def extract_model_at_stations(model_ranges, station_locations):
    """Extract model tidal ranges at specific station locations."""
    print("🎯 Extracting model data at station locations")
    
    model_locs = model_ranges['locations']
    dates = model_ranges['dates']
    ranges_grid = model_ranges['tidal_ranges']
    
    station_data = []
    
    for station, location in station_locations.items():
        # Interpolate to exact station location
        station_ranges = []
        
        for day_ranges in ranges_grid:
            interpolated_range = np.interp(location, model_locs, day_ranges)
            station_ranges.append(interpolated_range)
        
        for date, tidal_range in zip(dates, station_ranges):
            station_data.append({
                'Date': date,
                'Station': station,
                'Location': location,
                'TidalRange': tidal_range,
                'Source': 'Model'
            })
    
    model_df = pd.DataFrame(station_data)
    print(f"   ✓ Extracted {len(model_df)} model predictions at {len(station_locations)} stations")
    
    return model_df

def align_model_field_data(model_df, field_df):
    """Align model and field data by date and station."""
    print("🔗 Aligning model and field data")
    
    # Combine datasets
    model_df['Source'] = 'Model'
    field_df['Source'] = 'Field'
    
    # Ensure same date format
    model_df['Date'] = pd.to_datetime(model_df['Date'])
    field_df['Date'] = pd.to_datetime(field_df['Date'])
    
    # Combine and find overlapping dates
    combined = pd.concat([
        model_df[['Date', 'Station', 'TidalRange', 'Source']],
        field_df[['Date', 'Station', 'TidalRange', 'Source']]
    ])
    
    # Find dates with both model and field data
    date_station_counts = combined.groupby(['Date', 'Station']).size()
    valid_pairs = date_station_counts[date_station_counts == 2].index
    
    aligned_data = []
    for date, station in valid_pairs:
        model_val = model_df[(model_df['Date'] == date) & (model_df['Station'] == station)]['TidalRange'].iloc[0]
        field_val = field_df[(field_df['Date'] == date) & (field_df['Station'] == station)]['TidalRange'].iloc[0]
        
        aligned_data.append({
            'Date': date,
            'Station': station,
            'Location': TIDAL_STATIONS[station],
            'Model': model_val,
            'Field': field_val
        })
    
    aligned_df = pd.DataFrame(aligned_data)
    print(f"   ✓ Found {len(aligned_df)} matching model-field pairs")
    print(f"   ✓ Stations: {sorted(aligned_df['Station'].unique())}")
    
    return aligned_df

def calculate_tidal_validation_metrics(aligned_df):
    """Calculate validation metrics for tidal range predictions."""
    print("📊 Calculating tidal validation metrics")
    
    metrics_by_station = {}
    
    for station in aligned_df['Station'].unique():
        station_data = aligned_df[aligned_df['Station'] == station]
        
        if len(station_data) > 1:
            model_vals = station_data['Model'].values
            field_vals = station_data['Field'].values
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((model_vals - field_vals) ** 2))
            mae = np.mean(np.abs(model_vals - field_vals))
            mean_error = np.mean(model_vals - field_vals)
            
            # R²
            try:
                correlation_matrix = np.corrcoef(model_vals, field_vals)
                r2 = correlation_matrix[0, 1] ** 2
                if np.isnan(r2):
                    r2 = 0.0
            except:
                r2 = np.nan
            
            # Relative error
            rel_error = np.mean(np.abs(model_vals - field_vals) / field_vals) * 100
            
            metrics_by_station[station] = {
                'n_points': len(station_data),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_error': mean_error,
                'relative_error': rel_error,
                'model_mean': np.mean(model_vals),
                'field_mean': np.mean(field_vals)
            }
    
    return metrics_by_station

def create_tidal_validation_figure(aligned_df, metrics, output_dir="OUT"):
    """Create comprehensive tidal validation figure."""
    print("🎨 Creating tidal dynamics validation figure")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Time series comparison
    ax1 = axes[0, 0]
    stations = sorted(aligned_df['Station'].unique())
    colors = ['blue', 'green', 'red']
    
    for i, station in enumerate(stations):
        station_data = aligned_df[aligned_df['Station'] == station].sort_values('Date')
        ax1.plot(station_data['Date'], station_data['Model'], 
                color=colors[i], linestyle='-', label=f'{station} Model', alpha=0.7)
        ax1.scatter(station_data['Date'], station_data['Field'], 
                   color=colors[i], marker='o', label=f'{station} Observed', alpha=0.8)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Tidal Range (m)')
    ax1.set_title('Tidal Range Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model vs Field scatter
    ax2 = axes[0, 1]
    for i, station in enumerate(stations):
        station_data = aligned_df[aligned_df['Station'] == station]
        ax2.scatter(station_data['Field'], station_data['Model'], 
                   color=colors[i], label=station, alpha=0.7, s=50)
    
    # 1:1 line
    min_val = min(aligned_df['Field'].min(), aligned_df['Model'].min())
    max_val = max(aligned_df['Field'].max(), aligned_df['Model'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 Line')
    
    ax2.set_xlabel('Observed Tidal Range (m)')
    ax2.set_ylabel('Modeled Tidal Range (m)')
    ax2.set_title('Model vs Observations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Spatial tidal amplification
    ax3 = axes[1, 0]
    station_means = aligned_df.groupby(['Station', 'Location']).agg({
        'Model': 'mean',
        'Field': 'mean'
    }).reset_index()
    
    ax3.plot(station_means['Location'], station_means['Model'], 
            'b-o', label='Model', linewidth=2, markersize=8)
    ax3.plot(station_means['Location'], station_means['Field'], 
            'r-s', label='Observed', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Distance from Mouth (km)')
    ax3.set_ylabel('Mean Tidal Range (m)')
    ax3.set_title('Spatial Tidal Amplification')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation metrics bar chart
    ax4 = axes[1, 1]
    metrics_df = pd.DataFrame(metrics).T
    x_pos = np.arange(len(stations))
    
    bars1 = ax4.bar(x_pos - 0.2, metrics_df['rmse'], 0.4, label='RMSE (m)', alpha=0.7)
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x_pos + 0.2, metrics_df['r2'], 0.4, label='R²', alpha=0.7, color='orange')
    
    ax4.set_xlabel('Station')
    ax4.set_ylabel('RMSE (m)', color='blue')
    ax4_twin.set_ylabel('R²', color='orange')
    ax4.set_title('Validation Metrics by Station')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stations)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase2_tidal_dynamics.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Figure saved: {output_path}")
    
    # Print metrics summary
    print("\n📊 VALIDATION SUMMARY - Tidal Dynamics")
    print("=" * 60)
    print(f"{'Station':<8} {'n':>3} {'RMSE':>6} {'R²':>6} {'RelErr%':>7} {'ModelMean':>9} {'FieldMean':>9}")
    print("-" * 60)
    for station, vals in metrics.items():
        print(f"{station:<8} {vals['n_points']:>3} {vals['rmse']:>6.3f} {vals['r2']:>6.3f} "
              f"{vals['relative_error']:>6.1f}% {vals['model_mean']:>8.2f} {vals['field_mean']:>8.2f}")
    
    return output_path

def main():
    """Main tidal validation workflow."""
    print("🌊 Phase 2: Tidal Dynamics Validation")
    print("=" * 50)
    
    try:
        # Load data
        model_hydro = load_model_hydrodynamics()
        field_data = load_sihymecc_observations()
        
        # Calculate model tidal ranges
        model_ranges = calculate_model_tidal_ranges(model_hydro)
        
        # Extract at station locations
        model_at_stations = extract_model_at_stations(model_ranges, TIDAL_STATIONS)
        
        # Align model and field data
        aligned_data = align_model_field_data(model_at_stations, field_data)
        
        if aligned_data.empty:
            print("⚠️  No overlapping dates found between model and field data")
            return None
        
        # Calculate validation metrics
        metrics = calculate_tidal_validation_metrics(aligned_data)
        
        # Create validation figure
        figure_path = create_tidal_validation_figure(aligned_data, metrics)
        
        print("\n✅ Phase 2 validation completed successfully!")
        
        return {'aligned_data': aligned_data, 'metrics': metrics, 'figure': figure_path}
        
    except Exception as e:
        print(f"❌ Error in Phase 2 validation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()