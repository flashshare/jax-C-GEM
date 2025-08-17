#!/usr/bin/env python
"""
Phase 3: Seasonal Cycles Validation

This script validates model temporal dynamics against CARE long-term monitoring
data at Phú Cường station (116km from estuarine mouth).

Key Features:
- Monthly seasonal cycle validation
- Proper unit conversion from model mmol/m³ to field mg/L
- Statistical validation of temporal patterns
- Dry/wet season analysis following original C-GEM methodology

Field Data: CARE_2017-2018.csv (146 observations at PC station, 116km from mouth)
Species: NH4, NO3, PO4, TOC, DO, Salinity
Seasons: Dry=[12,1,2,3,4,5], Wet=[6,7,8,9,10,11]

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

# Unit conversion factors: mmol/m³ → mg/L
UNIT_CONVERSION_FACTORS = {
    'NH4': 14.0 / 1000.0,    # mmol NH4-N/m³ → mg N/L
    'NO3': 14.0 / 1000.0,    # mmol NO3-N/m³ → mg N/L
    'PO4': 31.0 / 1000.0,    # mmol PO4-P/m³ → mg P/L
    'TOC': 12.0 / 1000.0,    # mmol C/m³ → mg C/L
    'O2': 32.0 / 1000.0,     # mmol O2/m³ → mg O2/L
    'S': 1.0,                # Salinity (dimensionless)
}

# PC station location
PC_STATION_LOCATION = 116  # km from mouth

def load_model_results(results_dir="OUT"):
    """Load model results and extract time series at PC station."""
    npz_file = Path(results_dir) / "simulation_results.npz"
    
    if npz_file.exists():
        print(f"📊 Loading model results from {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        
        # Create spatial grid from model config
        # EL = 202000 m, DELXI = 2000 m → M = 102 cells, locations from 1-201 km
        M = 102
        locations = np.linspace(1, 201, M)  # km from mouth
        
        # Extract time array
        time_array = data['time'] if 'time' in data else np.arange(data['NH4'].shape[0])
        
        # Find PC station index
        pc_index = np.argmin(np.abs(locations - PC_STATION_LOCATION))
        actual_location = locations[pc_index]
        print(f"   ✓ PC station: {PC_STATION_LOCATION}km → model cell {pc_index} ({actual_location:.1f}km)")
        
        # Extract species time series at PC
        model_data = {}
        for species in UNIT_CONVERSION_FACTORS.keys():
            if species in data:
                # Extract time series at PC station
                time_series = data[species][:, pc_index]
                
                # Convert units: mmol/m³ → mg/L
                conversion_factor = UNIT_CONVERSION_FACTORS[species]
                model_data[species] = time_series * conversion_factor
                
                print(f"   ✓ {species}: {len(time_series)} time steps, "
                      f"range {time_series.min():.3f}-{time_series.max():.3f} mmol/m³")
                print(f"     → {model_data[species].min():.3f}-{model_data[species].max():.3f} mg/L")
        
        return {
            'time_array': time_array,
            'pc_location': actual_location,
            'species_data': model_data
        }
    else:
        raise FileNotFoundError(f"Model results not found in {results_dir}")

def load_care_observations():
    """Load CARE field observations at PC station."""
    care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
    
    if not care_file.exists():
        raise FileNotFoundError(f"CARE data not found: {care_file}")
    
    print(f"📋 Loading CARE observations from {care_file}")
    care_data = pd.read_csv(care_file)
    
    # Parse dates
    care_data['Date'] = pd.to_datetime(care_data['Date'], format='%d-%b-%y')
    
    # Filter for PC station only
    pc_data = care_data[care_data['Site'] == 'PC'].copy()
    
    # Define species mapping (CARE columns → model species)
    species_mapping = {
        'NH4 (mgN/L)': 'NH4',
        'NO3 (mgN/L)': 'NO3',
        'PO4 (mgP/L)': 'PO4',
        'TOC (mgC/L)': 'TOC',
        'DO (mg/L)': 'O2',
        'Salinity': 'S'
    }
    
    # Reshape to long format
    care_processed = []
    for _, row in pc_data.iterrows():
        for care_col, species in species_mapping.items():
            if care_col in care_data.columns and pd.notna(row[care_col]):
                care_processed.append({
                    'date': row['Date'],
                    'species': species,
                    'concentration': row[care_col],
                    'month': row['Date'].month,
                    'year': row['Date'].year
                })
    
    care_df = pd.DataFrame(care_processed)
    
    # Add seasonal classification (following original C-GEM methodology)
    care_df['season'] = 'Wet'
    care_df.loc[care_df['month'].isin([12, 1, 2, 3, 4, 5]), 'season'] = 'Dry'
    
    print(f"   ✓ Processed {len(care_df)} observations")
    print(f"   ✓ Date range: {care_df['date'].min()} to {care_df['date'].max()}")
    print(f"   ✓ Species: {sorted(care_df['species'].unique())}")
    print(f"   ✓ Seasonal breakdown: Dry={len(care_df[care_df['season']=='Dry'])}, "
          f"Wet={len(care_df[care_df['season']=='Wet'])}")
    
    return care_df

def calculate_model_monthly_cycles(model_results, warmup_days=100):
    """Calculate monthly cycles from model time series."""
    print(f"📈 Calculating model monthly cycles (excluding {warmup_days} warmup days)")
    
    time_array = model_results['time_array']
    species_data = model_results['species_data']
    
    # Skip warmup period
    warmup_steps = warmup_days * 24  # hours
    if len(time_array) > warmup_steps:
        analysis_start = warmup_steps
        print(f"   ✓ Using data from step {analysis_start} to {len(time_array)}")
    else:
        analysis_start = 0
        print(f"   ⚠️  Short simulation: using all {len(time_array)} steps")
    
    # Create date range for model data (assuming 2017 start)
    start_date = datetime(2017, 1, 1) + timedelta(hours=analysis_start)
    model_dates = pd.date_range(start=start_date, periods=len(time_array) - analysis_start, freq='H')
    
    # Calculate monthly averages
    monthly_model = {}
    for species, data in species_data.items():
        species_series = data[analysis_start:]
        
        # Create DataFrame for easier processing
        df = pd.DataFrame({
            'date': model_dates,
            'concentration': species_series
        })
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Calculate monthly means
        monthly_means = df.groupby(['year', 'month'])['concentration'].mean().reset_index()
        monthly_means['species'] = species
        
        # Add seasonal classification
        monthly_means['season'] = 'Wet'
        monthly_means.loc[monthly_means['month'].isin([12, 1, 2, 3, 4, 5]), 'season'] = 'Dry'
        
        monthly_model[species] = monthly_means
        
        print(f"   ✓ {species}: {len(monthly_means)} monthly averages")
    
    return monthly_model

def align_monthly_data(model_monthly, care_df):
    """Align model and field monthly data."""
    print("🔗 Aligning model and field monthly data")
    
    aligned_data = []
    
    # Group CARE data by year, month, species
    care_monthly = care_df.groupby(['year', 'month', 'species'])['concentration'].mean().reset_index()
    
    for species in model_monthly.keys():
        model_data = model_monthly[species]
        care_species = care_monthly[care_monthly['species'] == species]
        
        for _, care_row in care_species.iterrows():
            # Find matching model data
            model_match = model_data[
                (model_data['year'] == care_row['year']) & 
                (model_data['month'] == care_row['month'])
            ]
            
            if not model_match.empty:
                aligned_data.append({
                    'species': species,
                    'year': care_row['year'],
                    'month': care_row['month'],
                    'season': model_match.iloc[0]['season'],
                    'model': model_match.iloc[0]['concentration'],
                    'field': care_row['concentration']
                })
    
    aligned_df = pd.DataFrame(aligned_data)
    
    print(f"   ✓ Aligned {len(aligned_df)} model-field pairs")
    print(f"   ✓ Species: {sorted(aligned_df['species'].unique())}")
    print(f"   ✓ Time range: {aligned_df['year'].min()}-{aligned_df['month'].min()} to "
          f"{aligned_df['year'].max()}-{aligned_df['month'].max()}")
    
    return aligned_df

def calculate_seasonal_validation_metrics(aligned_df):
    """Calculate validation metrics by species and season."""
    print("📊 Calculating seasonal validation metrics")
    
    metrics = {}
    
    for species in aligned_df['species'].unique():
        species_data = aligned_df[aligned_df['species'] == species]
        
        metrics[species] = {}
        
        # Overall metrics
        if len(species_data) > 1:
            model_vals = species_data['model'].values
            field_vals = species_data['field'].values
            
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
            
            metrics[species]['overall'] = {
                'n_points': len(species_data),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_error': mean_error,
                'relative_error': rel_error
            }
        
        # Seasonal metrics
        for season in ['Dry', 'Wet']:
            season_data = species_data[species_data['season'] == season]
            
            if len(season_data) > 1:
                model_vals = season_data['model'].values
                field_vals = season_data['field'].values
                
                rmse = np.sqrt(np.mean((model_vals - field_vals) ** 2))
                mae = np.mean(np.abs(model_vals - field_vals))
                
                try:
                    correlation_matrix = np.corrcoef(model_vals, field_vals)
                    r2 = correlation_matrix[0, 1] ** 2
                    if np.isnan(r2):
                        r2 = 0.0
                except:
                    r2 = np.nan
                
                metrics[species][season.lower()] = {
                    'n_points': len(season_data),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
    
    return metrics

def create_seasonal_validation_figure(aligned_df, metrics, output_dir="OUT"):
    """Create comprehensive seasonal validation figure."""
    print("🎨 Creating seasonal cycles validation figure")
    
    # Get common species
    species_list = sorted(aligned_df['species'].unique())
    n_species = len(species_list)
    
    if n_species == 0:
        print("   ⚠️  No species data available for plotting")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, species in enumerate(species_list):
        if i >= len(axes):
            break
        
        ax = axes[i]
        species_data = aligned_df[aligned_df['species'] == species]
        
        if not species_data.empty:
            # Monthly time series
            monthly_data = species_data.groupby('month').agg({
                'model': 'mean',
                'field': 'mean'
            }).reset_index()
            
            ax.plot(monthly_data['month'], monthly_data['model'], 
                   'b-o', label='Model', linewidth=2, markersize=6)
            ax.plot(monthly_data['month'], monthly_data['field'], 
                   'r-s', label='CARE Obs', linewidth=2, markersize=6)
            
            # Add seasonal background
            dry_months = [12, 1, 2, 3, 4, 5]
            wet_months = [6, 7, 8, 9, 10, 11]
            
            y_min, y_max = ax.get_ylim()
            for month in dry_months:
                ax.axvspan(month-0.5, month+0.5, alpha=0.1, color='orange', label='Dry' if month == 12 else "")
            for month in wet_months:
                ax.axvspan(month-0.5, month+0.5, alpha=0.1, color='blue', label='Wet' if month == 6 else "")
            
            # Add validation metrics
            if species in metrics and 'overall' in metrics[species]:
                overall_metrics = metrics[species]['overall']
                ax.text(0.05, 0.95, 
                       f'R² = {overall_metrics["r2"]:.3f}\nRMSE = {overall_metrics["rmse"]:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Month')
        ax.set_ylabel(f'{species} (mg/L)')
        ax.set_title(f'{species} Seasonal Cycle at PC Station')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 13))
        
        # Set reasonable y-limits
        if not species_data.empty:
            all_vals = np.concatenate([species_data['model'].values, species_data['field'].values])
            ax.set_ylim(0, np.percentile(all_vals, 95) * 1.1)
    
    # Remove empty subplots
    for i in range(len(species_list), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase3_seasonal_cycles.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Figure saved: {output_path}")
    
    # Print validation summary
    print("\n📊 VALIDATION SUMMARY - Seasonal Cycles at PC Station")
    print("=" * 70)
    print(f"{'Species':<8} {'n':>3} {'RMSE':>6} {'R²':>6} {'RelErr%':>7} {'Dry-R²':>7} {'Wet-R²':>7}")
    print("-" * 70)
    
    for species, species_metrics in metrics.items():
        if 'overall' in species_metrics:
            overall = species_metrics['overall']
            dry_r2 = species_metrics.get('dry', {}).get('r2', np.nan)
            wet_r2 = species_metrics.get('wet', {}).get('r2', np.nan)
            
            print(f"{species:<8} {overall['n_points']:>3} {overall['rmse']:>6.3f} {overall['r2']:>6.3f} "
                  f"{overall['relative_error']:>6.1f}% {dry_r2:>6.3f} {wet_r2:>6.3f}")
    
    return output_path

def main():
    """Main seasonal validation workflow."""
    print("📅 Phase 3: Seasonal Cycles Validation")
    print("=" * 50)
    
    try:
        # Load data
        model_results = load_model_results()
        care_data = load_care_observations()
        
        # Calculate model monthly cycles
        model_monthly = calculate_model_monthly_cycles(model_results)
        
        # Align model and field data
        aligned_data = align_monthly_data(model_monthly, care_data)
        
        if aligned_data.empty:
            print("⚠️  No overlapping data found between model and field observations")
            return None
        
        # Calculate validation metrics
        metrics = calculate_seasonal_validation_metrics(aligned_data)
        
        # Create validation figure
        figure_path = create_seasonal_validation_figure(aligned_data, metrics)
        
        print("\n✅ Phase 3 validation completed successfully!")
        
        return {'aligned_data': aligned_data, 'metrics': metrics, 'figure': figure_path}
        
    except Exception as e:
        print(f"❌ Error in Phase 3 validation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()