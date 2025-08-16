#!/usr/bin/env python
"""
CORRECTED Phase 3 Analysis Script: Long-Term Seasonal Validation

This script properly analyzes seasonal biogeochemical cycles against CARE field observations
with correct temporal alignment and station mapping.

Critical Fixes:
- Proper temporal alignment with 2017 calendar year
- Correct station coordinate mapping (CARE: PC=86km, BD=130km, BK=156km)
- Monthly aggregation with proper date parsing
- Realistic seasonal pattern validation

Scientific Objective:
- Compare monthly-averaged model outputs against CARE 2017 observations
- Quantify seasonal cycle accuracy at PC, BD, BK stations
- Validate long-term biogeochemical dynamics

Usage: python tools/verification/phase3_validate_seasonal_cycle_corrected.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.model_config import SPECIES_NAMES
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import core modules: {e}")
    SPECIES_NAMES = ['PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2']

def load_simulation_results(results_file="OUT/complete_simulation_results.npz"):
    """Load the 365-day simulation results."""
    print(f"📂 Loading 365-day simulation results from {results_file}")
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    try:
        data = np.load(results_file)
        print(f"✅ Results loaded successfully")
        print(f"   Available keys: {list(data.keys())[:10]}...")
        print(f"   Time range: {data['time'][0]:.1f} - {data['time'][-1]:.1f} days")
        print(f"   Grid points: {data['H'].shape[1]}")
        return data
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def load_field_observations(field_file="INPUT/Calibration/CARE_2017-2018.csv"):
    """Load and properly parse CARE field observations."""
    print(f"📊 Loading CARE field observations from {field_file}")
    
    if not Path(field_file).exists():
        print(f"❌ CARE file not found: {field_file}")
        return None
    
    try:
        field_data = pd.read_csv(field_file)
        print(f"✅ Field data loaded: {len(field_data)} observations")
        print(f"   Columns: {field_data.columns.tolist()}")
        
        # Parse dates properly
        field_data['Date'] = pd.to_datetime(field_data['Date'], format='%d-%b-%y')
        field_data['month'] = field_data['Date'].dt.month
        field_data['year'] = field_data['Date'].dt.year
        
        print(f"   Date range: {field_data['Date'].min()} to {field_data['Date'].max()}")
        print(f"   Sites: {field_data['Site'].unique()}")
        print(f"   Locations: {sorted(field_data['Location'].unique())} km")
        
        return field_data
        
    except Exception as e:
        print(f"❌ Error loading field data: {e}")
        return None

def define_corrected_station_locations():
    """Define CORRECTED station locations for CARE data."""
    print("🔧 Defining corrected CARE station locations...")
    
    # CARE stations with proper coordinate conversion
    # Field coordinates are from upstream, model coordinates are from mouth
    stations = {
        'PC': {
            'name': 'Phu Cuong',
            'field_km': 86,     # From upstream (CARE data)
            'model_km': 202 - 86, # = 116 km from mouth
            'grid_idx': int((202 - 86) / 2)  # Grid index
        },
        'BD': {
            'name': 'Ben Do', 
            'field_km': 130,    # From upstream
            'model_km': 202 - 130, # = 72 km from mouth  
            'grid_idx': int((202 - 130) / 2)
        },
        'BK': {
            'name': 'Binh Khanh',
            'field_km': 156,    # From upstream
            'model_km': 202 - 156, # = 46 km from mouth
            'grid_idx': int((202 - 156) / 2)
        }
    }
    
    print("📍 CARE station mapping:")
    for code, info in stations.items():
        print(f"   {code} ({info['name']}): field {info['field_km']}km → model {info['model_km']}km (grid {info['grid_idx']})")
    
    return stations

def extract_model_seasonal_cycles(data, stations):
    """Extract seasonal cycles from model data at CARE station locations."""
    print("📈 Extracting model seasonal cycles at CARE stations...")
    
    # Define simulation time mapping
    # Assume simulation starts 2017-01-01 after 100-day warmup
    time_seconds = data['time']
    warmup_days = 100
    
    # Convert simulation time to calendar dates
    start_date = datetime(2017, 1, 1)
    simulation_dates = []
    monthly_data = {station: {var: [] for var in ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']} for station in stations.keys()}
    monthly_dates = []
    
    # Process time series data
    for i, t_sec in enumerate(time_seconds):
        t_days = t_sec / (24 * 3600)  # Convert to days
        if t_days < warmup_days:
            continue  # Skip warmup period
            
        # Calculate calendar date
        sim_date = start_date + timedelta(days=t_days - warmup_days)
        simulation_dates.append(sim_date)
        
        # Extract data at station locations
        for station_code, station_info in stations.items():
            grid_idx = station_info['grid_idx']
            
            # Extract species concentrations
            if 'S' in data:
                monthly_data[station_code]['S'].append(data['S'][i, grid_idx])
            if 'O2' in data: 
                monthly_data[station_code]['O2'].append(data['O2'][i, grid_idx])
            if 'NH4' in data:
                monthly_data[station_code]['NH4'].append(data['NH4'][i, grid_idx])
            if 'NO3' in data:
                monthly_data[station_code]['NO3'].append(data['NO3'][i, grid_idx])
            if 'PO4' in data:
                monthly_data[station_code]['PO4'].append(data['PO4'][i, grid_idx])  
            if 'TOC' in data:
                monthly_data[station_code]['TOC'].append(data['TOC'][i, grid_idx])
    
    # Convert to monthly averages
    print("📅 Calculating monthly averages...")
    df = pd.DataFrame({'date': simulation_dates})
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    monthly_model = {}
    for station_code in stations.keys():
        monthly_model[station_code] = {}
        station_df = df.copy()
        
        for var in ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']:
            if len(monthly_data[station_code][var]) == len(station_df):
                station_df[var] = monthly_data[station_code][var]
                # Group by month and calculate means
                monthly_means = station_df.groupby('month')[var].mean()
                monthly_model[station_code][var] = monthly_means
                print(f"   ✅ {station_code} {var}: {len(monthly_means)} monthly values")
    
    return monthly_model

def extract_field_seasonal_cycles(field_data, stations):
    """Extract seasonal cycles from CARE field observations."""
    print("📊 Extracting field seasonal cycles from CARE data...")
    
    # Variable name mapping
    field_var_map = {
        'S': 'Salinity',
        'O2': 'DO (mg/L)', 
        'NH4': 'NH4 (mgN/L)',
        'NO3': 'NO3 (mgN/L)',
        'PO4': 'PO4 (mgP/L)',
        'TOC': 'TOC (mgC/L)'
    }
    
    monthly_field = {}
    for station_code in stations.keys():
        monthly_field[station_code] = {}
        station_data = field_data[field_data['Site'] == station_code]
        
        if len(station_data) == 0:
            print(f"   ⚠️  No data for station {station_code}")
            continue
        
        print(f"   📍 {station_code}: {len(station_data)} observations")
        
        for model_var, field_var in field_var_map.items():
            if field_var in station_data.columns:
                # Group by month and calculate means
                monthly_means = station_data.groupby('month')[field_var].mean()
                monthly_field[station_code][model_var] = monthly_means
                print(f"     ✅ {model_var}: {len(monthly_means)} monthly values")
            else:
                print(f"     ❌ {field_var} not found in field data")
    
    return monthly_field

def calculate_seasonal_validation_metrics(monthly_model, monthly_field, stations):
    """Calculate validation metrics for seasonal cycles."""
    print("📊 Calculating seasonal validation metrics...")
    
    metrics = {}
    for station_code in stations.keys():
        metrics[station_code] = {}
        
        if station_code not in monthly_model or station_code not in monthly_field:
            continue
            
        for var in ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']:
            if var in monthly_model[station_code] and var in monthly_field[station_code]:
                model_data = monthly_model[station_code][var]
                field_data = monthly_field[station_code][var]
                
                # Find common months
                common_months = model_data.index.intersection(field_data.index)
                if len(common_months) > 1:
                    model_vals = model_data.loc[common_months].values
                    field_vals = field_data.loc[common_months].values
                    
                    rmse = np.sqrt(np.mean((model_vals - field_vals)**2))
                    correlation = np.corrcoef(model_vals, field_vals)[0,1] if len(common_months) > 1 else np.nan
                    
                    metrics[station_code][var] = {
                        'rmse': rmse,
                        'correlation': correlation,
                        'n_months': len(common_months),
                        'model_mean': model_vals.mean(),
                        'field_mean': field_vals.mean()
                    }
                    
                    print(f"   ✅ {station_code} {var}: RMSE={rmse:.3f}, r={correlation:.3f}, n={len(common_months)}")
    
    return metrics

def create_corrected_seasonal_validation_figure(monthly_model, monthly_field, stations, metrics, output_dir="OUT"):
    """Create CORRECTED seasonal validation figure."""
    print("🎨 Creating corrected seasonal validation figure...")
    
    # Create figure with subplots for each variable
    variables = ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fig.suptitle('CORRECTED Phase 3: Seasonal Cycle Validation\nModel vs CARE 2017 Observations', 
                fontsize=16, fontweight='bold')
    
    colors = {'PC': 'blue', 'BD': 'red', 'BK': 'green'}
    months = np.arange(1, 13)
    
    for i, var in enumerate(variables):
        ax = axes[i]
        
        # Plot model data (lines)
        for station_code, color in colors.items():
            if (station_code in monthly_model and 
                var in monthly_model[station_code]):
                model_data = monthly_model[station_code][var]
                ax.plot(model_data.index, model_data.values, 
                       color=color, linewidth=2, linestyle='-',
                       label=f'{station_code} Model')
        
        # Plot field data (points)  
        for station_code, color in colors.items():
            if (station_code in monthly_field and
                var in monthly_field[station_code]):
                field_data = monthly_field[station_code][var]
                ax.scatter(field_data.index, field_data.values,
                          color=color, s=100, marker='o', 
                          label=f'{station_code} Obs')
        
        # Formatting
        ax.set_xlabel('Month')
        ax.set_ylabel(f'{var}')
        ax.set_title(f'{var} Seasonal Cycle')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(1, 12)
        ax.set_xticks(months)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase3_corrected_seasonal_validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Corrected seasonal validation figure saved: {output_path}")
    
    return str(output_path)

def generate_corrected_seasonal_report(monthly_model, monthly_field, stations, metrics, output_dir="OUT"):
    """Generate corrected seasonal validation report."""
    print("📋 Generating corrected seasonal validation report...")
    
    report_path = Path(output_dir) / "phase3_corrected_validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("# CORRECTED PHASE 3 VALIDATION REPORT: Seasonal Cycles\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Coordinate and Temporal Correction\n\n")
        f.write("CRITICAL FIXES:\n")
        f.write("- Proper station coordinate mapping (field km → model km)\n")
        f.write("- Correct temporal alignment with 2017 calendar year\n") 
        f.write("- Monthly aggregation with proper date parsing\n\n")
        
        f.write("## Station Mapping\n\n")
        for code, info in stations.items():
            f.write(f"- {code} ({info['name']}): {info['field_km']}km → {info['model_km']}km (grid {info['grid_idx']})\n")
        f.write("\n")
        
        f.write("## Validation Metrics Summary\n\n")
        f.write("| Station | Variable | RMSE | Correlation | N Months | Model Mean | Field Mean |\n")
        f.write("|---------|----------|------|-------------|----------|------------|------------|\n")
        
        for station_code in stations.keys():
            if station_code in metrics:
                for var, metric in metrics[station_code].items():
                    f.write(f"| {station_code} | {var} | {metric['rmse']:.3f} | "
                           f"{metric['correlation']:.3f} | {metric['n_months']} | "
                           f"{metric['model_mean']:.2f} | {metric['field_mean']:.2f} |\n")
        
        f.write("\n## Critical Assessment\n\n")
        
        # Count successful validations
        total_comparisons = 0
        good_correlations = 0
        
        for station_code in metrics:
            for var, metric in metrics[station_code].items():
                total_comparisons += 1
                if not np.isnan(metric['correlation']) and metric['correlation'] > 0.5:
                    good_correlations += 1
        
        success_rate = (good_correlations / total_comparisons * 100) if total_comparisons > 0 else 0
        
        f.write(f"### Overall Performance\n")
        f.write(f"- Total comparisons: {total_comparisons}\n") 
        f.write(f"- Good correlations (r > 0.5): {good_correlations}\n")
        f.write(f"- Success rate: {success_rate:.1f}%\n\n")
        
        if success_rate < 50:
            f.write("❌ **MODEL VALIDATION FAILED**\n\n")
            f.write("Critical Issues:\n")
            f.write("- Poor correlation with field observations\n")
            f.write("- Large RMSE values indicate systematic bias\n") 
            f.write("- Seasonal patterns do not match observations\n\n")
            
            f.write("## Immediate Actions Required\n\n")
            f.write("1. **Boundary Condition Review**: Check upstream/downstream inputs\n")
            f.write("2. **Parameter Calibration**: Biogeochemical parameters need optimization\n")
            f.write("3. **Temporal Dynamics**: Verify seasonal forcing data\n")
            f.write("4. **Spatial Resolution**: Consider higher resolution near stations\n")
        else:
            f.write("✅ **MODEL SHOWS REASONABLE PERFORMANCE**\n\n")
    
    print(f"✅ Corrected seasonal validation report saved: {report_path}")
    return str(report_path)

def main():
    """Main analysis function."""
    print("🧬 CORRECTED PHASE 3 ANALYSIS: Long-Term Seasonal Validation")
    print("=" * 60)
    
    # Load simulation results
    data = load_simulation_results()
    if data is None:
        return
    
    # Load field observations  
    field_data = load_field_observations()
    if field_data is None:
        return
    
    # Define corrected station locations
    stations = define_corrected_station_locations()
    
    # Extract seasonal cycles from model data
    monthly_model = extract_model_seasonal_cycles(data, stations)
    
    # Extract seasonal cycles from field data
    monthly_field = extract_field_seasonal_cycles(field_data, stations)
    
    # Calculate validation metrics
    metrics = calculate_seasonal_validation_metrics(monthly_model, monthly_field, stations)
    
    # Create corrected validation figure
    figure_path = create_corrected_seasonal_validation_figure(monthly_model, monthly_field, stations, metrics)
    
    # Generate corrected validation report  
    report_path = generate_corrected_seasonal_report(monthly_model, monthly_field, stations, metrics)
    
    print("\n✅ CORRECTED PHASE 3 ANALYSIS COMPLETED")
    print("📊 Outputs:")
    print(f"   - Corrected seasonal figure: {figure_path}")
    print(f"   - Corrected validation report: {report_path}")
    
    # Assessment
    total_comparisons = sum(len(metrics.get(s, {})) for s in stations.keys())
    if total_comparisons == 0:
        print("\n❌ CRITICAL ERROR: No valid comparisons possible")
        print("   - Check data temporal alignment")
        print("   - Verify station coordinate mapping")
    else:
        print(f"\n📊 Validation completed: {total_comparisons} comparisons")
        print("🔍 See detailed report for performance assessment")
    
    return metrics

if __name__ == "__main__":
    main()