#!/usr/bin/env python
"""
Tidal Overestimation Diagnostic Tool

This script investigates the critical tidal overestimation problem:
- Model predicts 5.6-7.5m tidal ranges
- Field observations show 2.1-3.2m tidal ranges  
- Factor of 2-3x overestimation indicates physics implementation errors

Systematic Investigation:
1. Manning friction coefficient validation
2. Cross-sectional geometry verification
3. Boundary tidal forcing validation
4. Wave celerity and energy dissipation analysis

Author: Nguyen Truong An
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import signal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def load_model_hydrodynamics():
    """Load model hydrodynamic results"""
    print("📂 Loading model hydrodynamic results...")
    
    # Try NPZ format first
    npz_file = "OUT/complete_simulation_results.npz"
    if Path(npz_file).exists():
        data = np.load(npz_file)
        H = data['H']  # [time, space]
        U = data['U']  # [time, space] 
        print(f"✅ Loaded NPZ: H{H.shape}, U{U.shape}")
        return H, U
    
    # Fallback to CSV
    h_file = "OUT/Hydrodynamics/H.csv"
    u_file = "OUT/Hydrodynamics/U.csv"
    
    if not Path(h_file).exists():
        raise FileNotFoundError(f"❌ No hydrodynamic data found at {h_file}")
    
    H = np.loadtxt(h_file, delimiter=',')
    U = np.loadtxt(u_file, delimiter=',')
    print(f"✅ Loaded CSV: H{H.shape}, U{U.shape}")
    
    return H, U

def load_field_tidal_data():
    """Load SIHYMECC field tidal observations"""
    field_file = "INPUT/Calibration/SIHYMECC_Tidal-range2017-2018.csv"
    
    if not Path(field_file).exists():
        print(f"⚠️ No field data at {field_file}")
        return None
    
    df = pd.read_csv(field_file)
    print(f"✅ Loaded field tidal data: {len(df)} observations")
    print(f"   Stations: {df.columns[1:].tolist()}")
    return df

def calculate_model_tidal_ranges(H, warmup_days=10):
    """Calculate tidal ranges from model water level data"""
    print("🌊 Calculating model tidal ranges...")
    
    # Skip warmup period
    n_timesteps_per_day = H.shape[0] // 35  # Approximate
    skip_steps = warmup_days * n_timesteps_per_day
    H_analysis = H[skip_steps:, :]
    
    print(f"   Analysis period: {H_analysis.shape[0]} timesteps, {H.shape[1]} grid points")
    
    # Calculate tidal ranges for each grid point
    tidal_ranges = np.zeros(H.shape[1])
    
    for i in range(H.shape[1]):
        h_series = H_analysis[:, i]
        # Simple max-min approach (could be enhanced with tidal analysis)
        tidal_range = np.max(h_series) - np.min(h_series)
        tidal_ranges[i] = tidal_range
    
    return tidal_ranges

def analyze_manning_friction():
    """Analyze Manning friction coefficient effects"""
    print("🔧 Analyzing Manning friction coefficient...")
    
    # Load model configuration to get current Manning's n
    config_file = "config/model_config.txt"
    manning_n = 0.025  # Default assumption
    
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            for line in f:
                if 'MANNING' in line.upper() or 'FRICTION' in line.upper():
                    print(f"   Found friction parameter: {line.strip()}")
    
    print(f"   Current Manning's n: {manning_n}")
    print(f"   Typical estuarine values: 0.020-0.035")
    
    # Calculate what Manning's n would be needed to reduce tidal amplitude
    # Theoretical relationship: amplitude ∝ 1/friction
    current_overestimation = 2.5  # Average factor
    suggested_manning = manning_n * current_overestimation
    
    print(f"   Suggested Manning's n to reduce amplitude: {suggested_manning:.3f}")
    
    return manning_n, suggested_manning

def analyze_geometry_effects():
    """Analyze cross-sectional geometry effects on tidal amplification"""
    print("🏞️ Analyzing geometry effects...")
    
    # Load geometry if available
    geom_file = "INPUT/Geometry/Geometry.csv"
    if Path(geom_file).exists():
        geom = pd.read_csv(geom_file)
        print(f"✅ Loaded geometry: {len(geom)} cross-sections")
        
        # Basic geometry analysis
        if 'Width' in geom.columns and 'Depth' in geom.columns:
            widths = geom['Width'].values
            depths = geom['Depth'].values
            areas = widths * depths
            
            print(f"   Width range: {widths.min():.0f} - {widths.max():.0f} m")
            print(f"   Depth range: {depths.min():.1f} - {depths.max():.1f} m")
            print(f"   Area range: {areas.min():.0f} - {areas.max():.0f} m²")
            
            # Check for unrealistic geometry that could cause amplification
            convergence_ratio = widths[0] / widths[-1]  # Mouth to head
            print(f"   Convergence ratio: {convergence_ratio:.1f}")
            
            if convergence_ratio > 10:
                print("   ⚠️ HIGH CONVERGENCE may cause excessive tidal amplification")
            
            return geom
    else:
        print("   ⚠️ No geometry file found")
        return None

def analyze_wave_celerity(H, U):
    """Analyze wave celerity and energy dissipation"""
    print("🌊 Analyzing wave celerity and energy...")
    
    # Calculate average depth
    mean_depth = np.mean(H, axis=0)
    
    # Theoretical shallow water wave speed: c = sqrt(g*h)
    g = 9.81
    theoretical_celerity = np.sqrt(g * mean_depth)
    
    print(f"   Mean depth range: {mean_depth.min():.1f} - {mean_depth.max():.1f} m")
    print(f"   Theoretical wave speed: {theoretical_celerity.min():.1f} - {theoretical_celerity.max():.1f} m/s")
    
    # Estimate actual wave speed from model (simple approach)
    # This would need more sophisticated tidal analysis for accuracy
    dx = 2000.0  # Grid spacing in meters
    dt = 180.0   # Time step in seconds
    
    print(f"   Grid resolution: Δx={dx:.0f}m, Δt={dt:.0f}s")
    print(f"   CFL number: {theoretical_celerity.max() * dt / dx:.2f} (should be < 1.0)")
    
    return theoretical_celerity

def compare_with_field_data(model_ranges, field_df):
    """Compare model tidal ranges with field observations"""
    print("📊 Comparing with field observations...")
    
    if field_df is None:
        print("   ⚠️ No field data available for comparison")
        return
    
    # Station locations (km from mouth)
    stations = {'PC': 86/2, 'BD': 130/2, 'BK': 156/2}  # Convert to grid indices (rough)
    
    for station, grid_idx in stations.items():
        if station in field_df.columns:
            field_mean = field_df[station].mean()
            if grid_idx < len(model_ranges):
                model_value = model_ranges[int(grid_idx)]
                ratio = model_value / field_mean if field_mean > 0 else np.nan
                
                print(f"   {station}: Field={field_mean:.1f}m, Model={model_value:.1f}m, Ratio={ratio:.1f}x")
            else:
                print(f"   {station}: Grid index {grid_idx} out of range")

def create_diagnostic_plots(H, U, tidal_ranges, geom=None):
    """Create comprehensive diagnostic plots"""
    print("🎨 Creating diagnostic plots...")
    
    # Create distance grid
    dx = 2000.0  # meters
    distance_km = np.arange(H.shape[1]) * dx / 1000.0
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Tidal range along estuary
    axes[0,0].plot(distance_km, tidal_ranges, 'b-', linewidth=2, label='Model')
    axes[0,0].axhline(y=2.6, color='r', linestyle='--', label='PC Field (2.6m)')
    axes[0,0].axhline(y=3.2, color='g', linestyle='--', label='BD Field (3.2m)')
    axes[0,0].axhline(y=2.1, color='orange', linestyle='--', label='BK Field (2.1m)')
    axes[0,0].set_xlabel('Distance from Mouth (km)')
    axes[0,0].set_ylabel('Tidal Range (m)')
    axes[0,0].set_title('Tidal Range Comparison: Model vs Field')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Water level time series at key stations
    station_indices = [43, 65, 78]  # Approximate PC, BD, BK
    time_hours = np.arange(H.shape[0]) * 180 / 3600  # Convert to hours
    
    for i, idx in enumerate(station_indices):
        if idx < H.shape[1]:
            axes[0,1].plot(time_hours[-200:], H[-200:, idx], 
                          label=f'Station {idx} (≈{distance_km[idx]:.0f}km)')
    
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].set_ylabel('Water Level (m)')
    axes[0,1].set_title('Water Level Time Series (Last 200 steps)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Velocity patterns
    mean_velocity = np.mean(np.abs(U), axis=0)
    max_velocity = np.max(np.abs(U), axis=0)
    
    axes[1,0].plot(distance_km, mean_velocity, 'b-', label='Mean |Velocity|')
    axes[1,0].plot(distance_km, max_velocity, 'r--', label='Max |Velocity|')
    axes[1,0].set_xlabel('Distance from Mouth (km)')
    axes[1,0].set_ylabel('Velocity (m/s)')
    axes[1,0].set_title('Velocity Distribution Along Estuary')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Geometry (if available)
    if geom is not None and len(geom) == len(distance_km):
        if 'Width' in geom.columns:
            axes[1,1].plot(distance_km, geom['Width'], 'g-', label='Width')
        if 'Depth' in geom.columns:
            ax_depth = axes[1,1].twinx()
            ax_depth.plot(distance_km, geom['Depth'], 'b-', label='Depth')
            ax_depth.set_ylabel('Depth (m)', color='b')
        axes[1,1].set_xlabel('Distance from Mouth (km)')
        axes[1,1].set_ylabel('Width (m)', color='g')
        axes[1,1].set_title('Estuarine Geometry')
    else:
        axes[1,1].text(0.5, 0.5, 'Geometry data\nnot available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Geometry (Not Available)')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "OUT/tidal_overestimation_diagnosis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved diagnostic plot: {output_file}")
    
    return fig

def generate_diagnostic_report(manning_n, suggested_manning, field_df=None):
    """Generate comprehensive diagnostic report"""
    print("📋 Generating diagnostic report...")
    
    report = [
        "TIDAL OVERESTIMATION DIAGNOSTIC REPORT",
        "=" * 50,
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "PROBLEM SUMMARY:",
        "- Model tidal ranges: 5.6-7.5m",
        "- Field observations: 2.1-3.2m", 
        "- Overestimation factor: 2-3x",
        "",
        "ROOT CAUSE ANALYSIS:",
        "",
        "1. MANNING FRICTION COEFFICIENT:",
        f"   Current Manning's n: {manning_n:.3f}",
        f"   Typical estuarine range: 0.020-0.035",
        f"   Suggested n for amplitude reduction: {suggested_manning:.3f}",
        "",
        "2. GEOMETRY EFFECTS:",
        "   - Check cross-sectional areas for realism",
        "   - High convergence ratio may amplify tides",
        "   - Verify bathymetry against field surveys",
        "",
        "3. BOUNDARY FORCING:",
        "   - Verify tidal amplitude at mouth (should match field)",
        "   - Check phase relationships",
        "   - Validate harmonic constituents",
        "",
        "RECOMMENDED FIXES:",
        "",
        "PRIORITY 1: Increase Manning friction coefficient",
        f"   - Change Manning's n from {manning_n:.3f} to {suggested_manning:.3f}",
        "   - Test sensitivity to friction values",
        "",
        "PRIORITY 2: Validate geometry",
        "   - Compare cross-sections with field surveys", 
        "   - Check for unrealistic convergence",
        "   - Verify depth-area relationships",
        "",
        "PRIORITY 3: Boundary condition verification",
        "   - Confirm mouth tidal amplitude matches observations",
        "   - Check upstream discharge consistency",
        "",
        "VALIDATION TARGETS:",
        "- Tidal range ratio: 0.8-1.2 (currently 2-3x)",
        "- Phase relationships: Within ±30 minutes",
        "- Energy dissipation: Consistent with friction",
    ]
    
    # Save report
    report_file = "OUT/tidal_overestimation_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Saved diagnostic report: {report_file}")
    
    return '\n'.join(report)

def main():
    """Main diagnostic function"""
    print("🔍 TIDAL OVERESTIMATION DIAGNOSIS")
    print("=" * 50)
    
    try:
        # Load data
        H, U = load_model_hydrodynamics()
        field_df = load_field_tidal_data()
        
        # Calculate tidal ranges
        tidal_ranges = calculate_model_tidal_ranges(H)
        print(f"📊 Model tidal range: {tidal_ranges.min():.1f} - {tidal_ranges.max():.1f} m")
        
        # Analyze potential causes
        manning_n, suggested_manning = analyze_manning_friction()
        geom = analyze_geometry_effects()
        theoretical_celerity = analyze_wave_celerity(H, U)
        
        # Compare with field data
        compare_with_field_data(tidal_ranges, field_df)
        
        # Create plots and report
        fig = create_diagnostic_plots(H, U, tidal_ranges, geom)
        report = generate_diagnostic_report(manning_n, suggested_manning, field_df)
        
        print("\n" + "=" * 50)
        print("🎯 DIAGNOSIS COMPLETE")
        print("=" * 50)
        print("\nKEY FINDINGS:")
        print(f"1. Tidal overestimation factor: {tidal_ranges.max()/2.5:.1f}x average")
        print(f"2. Current Manning's n: {manning_n:.3f}")
        print(f"3. Suggested Manning's n: {suggested_manning:.3f}")
        print("\nFILES GENERATED:")
        print("- OUT/tidal_overestimation_diagnosis.png")
        print("- OUT/tidal_overestimation_report.txt")
        print("\n🔧 NEXT STEP: Modify Manning coefficient and re-run simulation")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()