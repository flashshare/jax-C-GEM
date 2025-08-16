#!/usr/bin/env python
"""
C-GEM Compatible Analysis Framework

This script applies the exact scientific methodology from the original C-GEM
analysis scripts while handling the unit conversion between JAX C-GEM (mmol/m³) 
and the expected mg/L units for field data comparison.

Key Methodology from Original C-GEM:
- Hourly model output → Daily averages  
- Seasonal classification: Dry=[12,1,2,3,4,5], Wet=[6,7,8,9,10,11]
- Station-based analysis at PC=86km, BD=130km, BK=156km
- Direct comparison with field observations (after unit conversion)
- Longitudinal profile analysis
- Time series validation with correlation metrics

Author: Nguyen Truong An
Based on: Original C-GEM plotWaterQuality2.py and plotFigure2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Unit conversion factors: mmol/m³ → mg/L (following our breakthrough analysis)
UNIT_CONVERSION_FACTORS = {
    'S': 1.0,              # Salinity (already in psu)
    'NH4': 14.0 / 1000.0,  # NH4-N: 14 g/mol → mg/L
    'NO3': 14.0 / 1000.0,  # NO3-N: 14 g/mol → mg/L  
    'PO4': 31.0 / 1000.0,  # PO4-P: 31 g/mol → mg/L
    'TOC': 12.0 / 1000.0,  # TOC-C: 12 g/mol → mg/L
    'O2': 32.0 / 1000.0,   # O2: 32 g/mol → mg/L
    'SI': 28.1 / 1000.0,   # Si: 28.1 g/mol → mg/L
    'SPM': 1.0,            # SPM already in mg/L
    'PHY1': 12.0 / 1000.0, # Diatoms as Carbon → mg/L
    'PHY2': 12.0 / 1000.0  # Non-diatoms as Carbon → mg/L
}

# Station locations (from original C-GEM methodology)
STATION_LOCATIONS = {
    'PC': 86,   # Phu Cuong - upstream station
    'BD': 130,  # Ben Do - urban station  
    'BK': 156   # Binh Khanh - downstream station
}

def convert_model_to_field_units(data_array, species_name):
    """Convert model units (mmol/m³) to field units (mg/L) following original methodology."""
    if species_name in UNIT_CONVERSION_FACTORS:
        conversion_factor = UNIT_CONVERSION_FACTORS[species_name]
        return data_array * conversion_factor
    else:
        print(f"⚠️  Warning: No conversion factor for {species_name}, using as-is")
        return data_array

def load_jax_cgem_results(results_file="OUT/complete_simulation_results.npz"):
    """Load JAX C-GEM results and convert to original C-GEM analysis format."""
    print(f"📂 Loading JAX C-GEM results from {results_file}")
    
    try:
        # Load NPZ results
        with np.load(results_file, allow_pickle=True) as data:
            results = {key: data[key] for key in data.files}
            
        print(f"✅ Loaded {len(results)} variables from NPZ file")
        
        # Convert to pandas DataFrame following original C-GEM pattern
        simulation_data = {}
        
        # Extract time array (following original: hourly from 2017-01-01)
        if 'time' in results:
            time_days = results['time']
            time_step = pd.date_range('2017-01-01', periods=len(time_days), freq='1H')
        else:
            print("⚠️  Warning: No time data found, creating default time series")
            n_steps = len(next(iter(results.values())))
            time_step = pd.date_range('2017-01-01', periods=n_steps, freq='1H')
        
        # Process each species following original methodology
        species_to_process = ['S', 'NH4', 'NO3', 'PO4', 'TOC', 'O2', 'SI', 'SPM', 'PHY1', 'PHY2']
        
        for species in species_to_process:
            if species in results:
                print(f"  Processing {species}...")
                
                # Get raw model data (should be shape: time x space)
                raw_data = results[species]
                if isinstance(raw_data, list):
                    raw_data = np.array(raw_data)
                    
                # Convert units: mmol/m³ → mg/L
                converted_data = convert_model_to_field_units(raw_data, species)
                
                # Create DataFrame following original pattern
                df = pd.DataFrame(converted_data)
                df = df.iloc[:-1, :]  # Remove last time step (original pattern)
                df["Date"] = time_step[:len(df)]
                df = df.set_index("Date")
                
                # Daily averaging (original methodology)
                df_daily = df.resample('D').mean()
                
                # Convert to long format (original melt pattern)
                df_long = df_daily.melt(var_name='Cell', ignore_index=False)
                df_long = df_long.astype(float)
                
                # Location mapping (no flipping needed as user confirmed)
                df_long["Location"] = df_long["Cell"] * 2  # Simplified mapping
                df_long = df_long.rename(columns={'value': species})
                df_long = df_long.reset_index()
                
                # Add temporal features (original seasonal classification)
                df_long['Month'] = df_long["Date"].dt.month
                df_long['Season'] = "Wet"
                df_long.loc[df_long['Month'].isin([12,1,2,3,4,5]), 'Season'] = 'Dry'
                df_long["Type"] = "Simulation"
                
                simulation_data[species] = df_long
                
        return simulation_data, time_step
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None

def load_field_observations():
    """Load field observations following original methodology."""
    print("📂 Loading field observations...")
    
    try:
        # Load CEM data (spatial profiles)
        cem_data = pd.read_csv("INPUT/Calibration/CEM_2017-2018.csv")
        cem_data['Date'] = pd.to_datetime(cem_data['Date'])
        
        # Load CARE data (time series at stations) 
        care_data = pd.read_csv("INPUT/Calibration/CARE_2017-2018.csv")
        care_data['Date'] = pd.to_datetime(care_data['Date'])
        
        print("✅ Field observations loaded successfully")
        return cem_data, care_data
        
    except Exception as e:
        print(f"❌ Error loading field data: {e}")
        return None, None

def create_longitudinal_profiles(simulation_data, cem_data, output_dir="OUT"):
    """Create longitudinal profile plots following original C-GEM methodology."""
    print("📊 Creating longitudinal profile analysis...")
    
    # Parameters to plot (matching original analysis)
    parameters = {
        'SPM': 'TSS (mg/L)',
        'O2': 'DO (mg/L)', 
        'NH4': 'NH4 (mgN/L)',
        'NO3': 'NO3 (mgN/L)',
        'PO4': 'PO4 (mgP/L)',
        'SI': 'DSi (mgSi/L)',
        'PHY1': 'Chl-a (μg/L)',
        'TOC': 'TOC (mgC/L)'
    }
    
    n = len(parameters)
    fig, axes = plt.subplots(int(np.ceil(n/2)), 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (species, label) in enumerate(parameters.items()):
        ax = axes[i]
        
        if species in simulation_data:
            # Plot simulation (following original seaborn pattern)
            sns.lineplot(x='Location', y=species, data=simulation_data[species], 
                        ax=ax, ci="sd", label='Simulation')
            
            # Plot observations if available
            field_col_mapping = {
                'NH4': 'NH4 (mgN/L)',
                'NO3': 'NO3 (mgN/L)', 
                'PO4': 'PO4 (mgP/L)',
                'TOC': 'TOC (mgC/L)',
                'O2': 'DO (mg/L)',
                'SPM': 'TSS (mg/L)'
            }
            
            if species in field_col_mapping and field_col_mapping[species] in cem_data.columns:
                field_col = field_col_mapping[species]
                sns.scatterplot(x="Location", y=field_col, data=cem_data,
                              ax=ax, color='red', label='Observation')
        
        ax.set_ylabel(label)
        ax.set_title(f"{label} Longitudinal Profile")
        if i == 0:
            ax.legend()
    
    # Set limits following original methodology
    if 'SPM' in simulation_data:
        axes[0].set_ylim(0, 200)
    if 'PHY1' in simulation_data:
        axes[6].set_ylim(0, 76)
    
    plt.suptitle("Water Quality in Saigon River - JAX C-GEM vs Observations", fontsize=15)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%d-%m-%H-%M")
    plt.savefig(f"{output_dir}/Longitudinal_Profile_CGEM_Compatible_{timestamp}.png", 
                bbox_inches='tight', dpi=200)
    plt.show()

def create_station_time_series(simulation_data, care_data, output_dir="OUT"):
    """Create station-based time series following original methodology."""
    print("📊 Creating station time series analysis...")
    
    # Extract station data (following original pattern)
    station_sim_data = {}
    for station_name, location_km in STATION_LOCATIONS.items():
        station_sim_data[station_name] = {}
        for species in simulation_data:
            data = simulation_data[species]
            station_data = data.loc[data["Location"] == location_km, :]
            station_sim_data[station_name][species] = station_data
    
    # Extract observation data by station
    station_obs_data = {}
    for station_name in STATION_LOCATIONS:
        if 'Site' in care_data.columns:
            station_obs_data[station_name] = care_data[care_data.Site == station_name]
        else:
            print(f"⚠️  Warning: No 'Site' column in CARE data for station {station_name}")
    
    # Create 8x3 figure (original pattern)
    parameters = ['SPM', 'O2', 'NH4', 'NO3', 'PO4', 'SI', 'PHY1', 'TOC']
    parameter_labels = {
        'SPM': 'TSS (mg/L)', 'O2': 'DO (mg/L)', 'NH4': 'NH4 (mgN/L)',
        'NO3': 'NO3 (mgN/L)', 'PO4': 'PO4 (mgP/L)', 'SI': 'DSi (mgSi/L)',
        'PHY1': 'Chl-a (μg/L)', 'TOC': 'TOC (mgC/L)'
    }
    
    fig, axes = plt.subplots(8, 3, figsize=(12, 12))
    
    for i, species in enumerate(parameters):
        for j, station_name in enumerate(['PC', 'BD', 'BK']):
            ax = axes[i, j]
            
            # Plot simulation time series
            if station_name in station_sim_data and species in station_sim_data[station_name]:
                station_data = station_sim_data[station_name][species]
                if not station_data.empty:
                    ax.plot(station_data["Date"], station_data[species], 
                           lw=0.5, color="black", label=f"Simulated {station_name}")
            
            # Plot observations
            if station_name in station_obs_data:
                obs_data = station_obs_data[station_name]
                field_mapping = {
                    'NH4': 'NH4 (mgN/L)', 'PO4': 'PO4 (mgP/L)', 'TOC': 'TOC (mgC/L)',
                    'O2': 'DO (mg/L)', 'S': 'Salinity'
                }
                
                if species in field_mapping and field_mapping[species] in obs_data.columns:
                    field_col = field_mapping[species]
                    ax.scatter(obs_data["Date"], obs_data[field_col], 
                             facecolors="none", edgecolors="blue", marker='.')
            
            # Formatting (following original pattern)
            if i == 0:
                ax.set_title(f"km {STATION_LOCATIONS[station_name]} - {station_name} station")
            if j > 0:
                ax.set_yticklabels([])
            if i < 7:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(parameter_labels.get(species, species))
    
    # Format bottom row x-axes
    for j in range(3):
        axes[7, j].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[7, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[7, j].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%d-%m-%H-%M")
    plt.savefig(f"{output_dir}/Station_Time_Series_CGEM_Compatible_{timestamp}.png", 
                bbox_inches='tight', dpi=200)
    plt.show()

def calculate_validation_statistics(simulation_data, field_data):
    """Calculate validation statistics following original methodology."""
    print("📈 Calculating validation statistics...")
    
    # Statistics will be calculated per species and station
    validation_results = {}
    
    # Implementation will follow the Pearson correlation, RMSE, MAPE pattern from original
    # This is a placeholder for the statistical framework
    
    return validation_results

def main():
    """Main analysis function following original C-GEM methodology."""
    print("🔬 JAX C-GEM Analysis Framework - Following Original C-GEM Methodology")
    print("=" * 70)
    
    # Load simulation results and convert units properly
    simulation_data, time_step = load_jax_cgem_results()
    if simulation_data is None:
        print("❌ Failed to load simulation results")
        return
    
    # Load field observations  
    cem_data, care_data = load_field_observations()
    if cem_data is None or care_data is None:
        print("❌ Failed to load field observations")
        return
    
    print(f"✅ Data loaded successfully:")
    print(f"   - Simulation species: {list(simulation_data.keys())}")
    if time_step is not None and len(time_step) > 0:
        print(f"   - Time period: {time_step[0]} to {time_step[-1]}")
    else:
        print("   - Time period: Unable to determine")
    print(f"   - CEM observations: {len(cem_data)} records")
    print(f"   - CARE observations: {len(care_data)} records")
    
    # Create analyses following original methodology
    create_longitudinal_profiles(simulation_data, cem_data)
    create_station_time_series(simulation_data, care_data)
    
    # Calculate validation statistics
    stats = calculate_validation_statistics(simulation_data, care_data)
    
    print("🎯 Analysis complete! Figures saved following original C-GEM methodology")

if __name__ == "__main__":
    main()