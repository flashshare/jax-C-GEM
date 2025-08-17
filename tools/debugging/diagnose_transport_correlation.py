#!/usr/bin/env python
"""
Transport Correlation Diagnostic Tool

This script investigates why most species have poor spatial correlations
while O₂ shows good correlation (R²=0.792 vs others 0.125-0.367).

Key Investigation Areas:
1. Boundary condition values for each species
2. Dispersion coefficient magnitude and realism  
3. Transport-only runs (no biogeochemistry)
4. Mass conservation analysis
5. Why O₂ works while others fail

Author: Nguyen Truong An
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Unit conversion factors: mmol/m³ → mg/L (same as verification scripts)
UNIT_CONVERSION_FACTORS = {
    'NH4': 14.0 / 1000.0,    # mmol N/m³ → mg N/L
    'NO3': 14.0 / 1000.0,    # mmol N/m³ → mg N/L
    'PO4': 31.0 / 1000.0,    # mmol P/m³ → mg P/L
    'TOC': 12.0 / 1000.0,    # mmol C/m³ → mg C/L
    'O2': 32.0 / 1000.0,     # mmol O₂/m³ → mg O₂/L
    'S': 1.0,                # Already in correct units
    'SPM': 1.0,              # Already in correct units
}

def load_simulation_results():
    """Load model results"""
    print("📂 Loading simulation results...")
    
    npz_file = "OUT/complete_simulation_results.npz"
    if Path(npz_file).exists():
        data = np.load(npz_file)
        species_data = {}
        
        for species in ['NH4', 'NO3', 'PO4', 'TOC', 'O2', 'S', 'SPM']:
            if species in data:
                species_data[species] = data[species]
                print(f"✅ Loaded {species}: {species_data[species].shape}")
            elif species == 'TOC' and 'TOC' not in data:
                # Sometimes stored as other names
                for alt_name in ['OC', 'OM', 'POC']:
                    if alt_name in data:
                        species_data[species] = data[alt_name]
                        print(f"✅ Loaded {species} (as {alt_name}): {species_data[species].shape}")
                        break
        
        return species_data
    
    print("❌ No simulation results found")
    return {}

def load_cem_observations():
    """Load CEM field observations for comparison"""
    print("📊 Loading CEM field observations...")
    
    cem_file = "INPUT/Calibration/CEM_2017-2018.csv"
    if not Path(cem_file).exists():
        print(f"❌ CEM file not found: {cem_file}")
        return None
    
    cem_df = pd.read_csv(cem_file)
    print(f"✅ Loaded CEM data: {len(cem_df)} observations")
    print(f"   Available species: {[col for col in cem_df.columns if col not in ['Date', 'Station', 'Distance_km']]}")
    
    return cem_df

def analyze_boundary_conditions():
    """Analyze boundary condition values for each species"""
    print("🔍 Analyzing boundary conditions...")
    
    # Check boundary condition files
    ub_dir = Path("INPUT/Boundary/UB")  # Upstream
    lb_dir = Path("INPUT/Boundary/LB")  # Downstream
    
    boundary_analysis = {}
    
    for species in ['NH4', 'NO3', 'PO4', 'TOC', 'O2', 'S', 'SPM']:
        analysis = {'species': species}
        
        # Upstream boundary
        ub_file = ub_dir / f"{species.upper()}_ub.csv"
        if ub_file.exists():
            try:
                ub_data = pd.read_csv(ub_file)
                if len(ub_data.columns) > 1:
                    values = ub_data.iloc[:, 1].values  # Second column typically contains values
                    analysis['upstream_mean'] = np.mean(values)
                    analysis['upstream_std'] = np.std(values)
                    analysis['upstream_range'] = [np.min(values), np.max(values)]
                else:
                    analysis['upstream_error'] = "Invalid file format"
            except Exception as e:
                analysis['upstream_error'] = str(e)
        else:
            analysis['upstream_error'] = "File not found"
        
        # Downstream boundary
        lb_file = lb_dir / f"{species.upper()}_lb.csv"
        if lb_file.exists():
            try:
                lb_data = pd.read_csv(lb_file)
                if len(lb_data.columns) > 1:
                    values = lb_data.iloc[:, 1].values
                    analysis['downstream_mean'] = np.mean(values)
                    analysis['downstream_std'] = np.std(values)
                    analysis['downstream_range'] = [np.min(values), np.max(values)]
                else:
                    analysis['downstream_error'] = "Invalid file format"
            except Exception as e:
                analysis['downstream_error'] = str(e)
        else:
            analysis['downstream_error'] = "File not found"
        
        boundary_analysis[species] = analysis
    
    return boundary_analysis

def calculate_dispersion_coefficients():
    """Analyze dispersion coefficient magnitudes"""
    print("🌊 Analyzing dispersion coefficients...")
    
    # This would need to be extracted from the model internals
    # For now, provide typical estuarine ranges for comparison
    
    typical_dispersion = {
        'Conservative (Salt)': [50, 500],    # m²/s
        'Nutrients': [10, 200],              # m²/s  
        'Particulates': [5, 100],            # m²/s
        'Dissolved gases': [20, 300],        # m²/s
    }
    
    print("   Typical estuarine dispersion coefficients:")
    for substance, range_vals in typical_dispersion.items():
        print(f"   {substance}: {range_vals[0]}-{range_vals[1]} m²/s")
    
    return typical_dispersion

def compare_species_correlations(species_data, cem_df):
    """Compare spatial correlations for different species"""
    print("📈 Comparing species correlations...")
    
    if cem_df is None:
        print("   ⚠️ No field data for comparison")
        return {}
    
    correlations = {}
    
    # Calculate mean longitudinal profiles (time-averaged)
    warmup_fraction = 0.3
    
    for species, data in species_data.items():
        if len(data.shape) != 2:
            continue
            
        # Skip warmup period and calculate mean
        skip_steps = int(data.shape[0] * warmup_fraction)
        mean_profile = np.mean(data[skip_steps:, :], axis=0)
        
        # Convert units to match field data
        if species in UNIT_CONVERSION_FACTORS:
            mean_profile = mean_profile * UNIT_CONVERSION_FACTORS[species]
        
        # Compare with CEM data at corresponding locations
        if species in cem_df.columns:
            # Get CEM spatial data
            cem_species = cem_df.groupby('Distance_km')[species].mean()
            
            # Interpolate model results to CEM locations
            model_distances = np.arange(len(mean_profile)) * 2  # 2km spacing
            
            model_interp = []
            cem_values = []
            
            for dist_km, cem_val in cem_species.items():
                if not np.isnan(cem_val) and dist_km <= model_distances[-1]:
                    model_idx = int(dist_km / 2)  # Convert km to index
                    if model_idx < len(mean_profile):
                        model_interp.append(mean_profile[model_idx])
                        cem_values.append(cem_val)
            
            if len(model_interp) > 2:  # Need at least 3 points for correlation
                r2 = pearsonr(model_interp, cem_values)[0] ** 2
                rmse = np.sqrt(np.mean((np.array(model_interp) - np.array(cem_values))**2))
                
                correlations[species] = {
                    'r2': r2,
                    'rmse': rmse,
                    'n_points': len(model_interp),
                    'model_range': [min(model_interp), max(model_interp)],
                    'field_range': [min(cem_values), max(cem_values)]
                }
                
                print(f"   {species}: R²={r2:.3f}, RMSE={rmse:.2f}, n={len(model_interp)}")
            else:
                print(f"   {species}: Insufficient data points for correlation")
        else:
            print(f"   {species}: Not found in CEM data")
    
    return correlations

def analyze_o2_success(species_data, correlations):
    """Analyze why O₂ has better correlation than other species"""
    print("🔍 Analyzing O₂ success pattern...")
    
    if 'O2' not in correlations:
        print("   ⚠️ O₂ correlation data not available")
        return
    
    o2_r2 = correlations['O2']['r2']
    print(f"   O₂ R² = {o2_r2:.3f} (Good performance)")
    
    # Compare with other species
    other_species = [s for s in correlations.keys() if s != 'O2']
    print(f"   Other species R²:")
    
    for species in other_species:
        r2 = correlations[species]['r2']
        ratio = o2_r2 / r2 if r2 > 0 else np.inf
        print(f"     {species}: R²={r2:.3f} ({ratio:.1f}x worse than O₂)")
    
    # Analyze O₂ characteristics
    if 'O2' in species_data:
        o2_data = species_data['O2']
        
        # Calculate spatial gradients
        mean_profile = np.mean(o2_data, axis=0)
        gradient_strength = np.std(mean_profile) / np.mean(mean_profile)
        
        print(f"   O₂ gradient strength: {gradient_strength:.3f}")
        print(f"   O₂ range: {mean_profile.min():.2f} - {mean_profile.max():.2f}")
    
    # Potential reasons for O₂ success
    print("   Potential reasons for O₂ success:")
    print("   1. Boundary conditions may be more realistic")
    print("   2. Reaeration provides strong source term")
    print("   3. Solubility equilibrium helps stabilize")
    print("   4. Less sensitive to transport errors")

def check_mass_conservation(species_data):
    """Check mass conservation for each species"""
    print("⚖️ Checking mass conservation...")
    
    conservation_results = {}
    
    for species, data in species_data.items():
        if len(data.shape) != 2:
            continue
        
        # Calculate total mass at each time step
        # Assume uniform grid spacing and cross-sectional area for simplicity
        dx = 2000.0  # Grid spacing in meters
        
        total_mass = np.sum(data, axis=1) * dx  # Simple integration
        
        # Check mass change rate
        mass_change = total_mass[-1] - total_mass[0]
        relative_change = abs(mass_change) / total_mass[0] if total_mass[0] != 0 else np.inf
        
        # Check for oscillations in mass
        mass_std = np.std(np.diff(total_mass))
        mass_trend = np.polyfit(range(len(total_mass)), total_mass, 1)[0]
        
        conservation_results[species] = {
            'initial_mass': total_mass[0],
            'final_mass': total_mass[-1],
            'relative_change': relative_change,
            'mass_std': mass_std,
            'trend': mass_trend
        }
        
        print(f"   {species}: Rel. change={relative_change:.4f}, Trend={mass_trend:.2e}")
    
    return conservation_results

def create_diagnostic_plots(species_data, correlations, boundary_analysis):
    """Create comprehensive diagnostic plots"""
    print("🎨 Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Species correlations
    if correlations:
        species_list = list(correlations.keys())
        r2_values = [correlations[s]['r2'] for s in species_list]
        
        bars = axes[0,0].bar(species_list, r2_values)
        axes[0,0].axhline(y=0.6, color='r', linestyle='--', label='Target (R²>0.6)')
        axes[0,0].axhline(y=correlations.get('O2', {}).get('r2', 0), color='g', 
                         linestyle='--', label=f'O₂ Performance')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Species Spatial Correlation Performance')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Color code bars
        for bar, r2 in zip(bars, r2_values):
            if r2 >= 0.6:
                bar.set_color('green')
            elif r2 >= 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    # Plot 2: Boundary condition comparison
    if boundary_analysis:
        species_with_bc = []
        upstream_means = []
        downstream_means = []
        
        for species, analysis in boundary_analysis.items():
            if 'upstream_mean' in analysis and 'downstream_mean' in analysis:
                species_with_bc.append(species)
                upstream_means.append(analysis['upstream_mean'])
                downstream_means.append(analysis['downstream_mean'])
        
        if species_with_bc:
            x = range(len(species_with_bc))
            width = 0.35
            
            axes[0,1].bar([i - width/2 for i in x], upstream_means, width, 
                         label='Upstream', alpha=0.8)
            axes[0,1].bar([i + width/2 for i in x], downstream_means, width, 
                         label='Downstream', alpha=0.8)
            
            axes[0,1].set_xlabel('Species')
            axes[0,1].set_ylabel('Concentration')
            axes[0,1].set_title('Boundary Condition Values')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(species_with_bc)
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Longitudinal profiles comparison
    distance_km = np.arange(102) * 2  # 2km spacing
    
    for i, (species, data) in enumerate(list(species_data.items())[:4]):  # First 4 species
        if len(data.shape) == 2:
            mean_profile = np.mean(data, axis=0)
            axes[1,0].plot(distance_km, mean_profile, label=f'{species}')
    
    axes[1,0].set_xlabel('Distance from Mouth (km)')
    axes[1,0].set_ylabel('Concentration')
    axes[1,0].set_title('Longitudinal Profiles (Model)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Performance summary
    if correlations:
        performance_data = []
        labels = []
        colors = []
        
        for species, data in correlations.items():
            performance_data.append([data['r2'], data['rmse']/max(1, data['rmse'])])  # Normalized
            labels.append(species)
            if data['r2'] >= 0.6:
                colors.append('green')
            elif data['r2'] >= 0.3:
                colors.append('orange')
            else:
                colors.append('red')
        
        performance_array = np.array(performance_data)
        scatter = axes[1,1].scatter(performance_array[:, 0], performance_array[:, 1], 
                                  c=colors, s=100, alpha=0.7)
        
        for i, label in enumerate(labels):
            axes[1,1].annotate(label, (performance_array[i, 0], performance_array[i, 1]),
                              xytext=(5, 5), textcoords='offset points')
        
        axes[1,1].set_xlabel('R² Score')
        axes[1,1].set_ylabel('Normalized RMSE')
        axes[1,1].set_title('Species Performance Summary')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = "OUT/transport_correlation_diagnosis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved diagnostic plot: {output_file}")

def generate_diagnostic_report(correlations, boundary_analysis, conservation_results):
    """Generate comprehensive diagnostic report"""
    print("📋 Generating diagnostic report...")
    
    report = [
        "TRANSPORT CORRELATION DIAGNOSTIC REPORT",
        "=" * 55,
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "PROBLEM SUMMARY:",
        "- Most species show poor spatial correlation (R² = 0.125-0.367)",
        "- O₂ shows good correlation (R² = 0.792)",
        "- Need to understand why O₂ works while others fail",
        "",
        "SPECIES PERFORMANCE ANALYSIS:",
    ]
    
    if correlations:
        for species, data in correlations.items():
            status = "GOOD" if data['r2'] >= 0.6 else "FAIR" if data['r2'] >= 0.3 else "POOR"
            report.append(f"   {species}: R²={data['r2']:.3f}, RMSE={data['rmse']:.2f} - {status}")
    
    report.extend([
        "",
        "BOUNDARY CONDITION ANALYSIS:",
    ])
    
    if boundary_analysis:
        for species, analysis in boundary_analysis.items():
            report.append(f"   {species}:")
            if 'upstream_mean' in analysis:
                report.append(f"     Upstream: {analysis['upstream_mean']:.2f} ± {analysis.get('upstream_std', 0):.2f}")
            if 'downstream_mean' in analysis:
                report.append(f"     Downstream: {analysis['downstream_mean']:.2f} ± {analysis.get('downstream_std', 0):.2f}")
            if 'upstream_error' in analysis:
                report.append(f"     Upstream ERROR: {analysis['upstream_error']}")
            if 'downstream_error' in analysis:
                report.append(f"     Downstream ERROR: {analysis['downstream_error']}")
    
    report.extend([
        "",
        "MASS CONSERVATION ANALYSIS:",
    ])
    
    if conservation_results:
        for species, data in conservation_results.items():
            status = "GOOD" if data['relative_change'] < 0.01 else "POOR"
            report.append(f"   {species}: Rel. change={data['relative_change']:.4f} - {status}")
    
    report.extend([
        "",
        "ROOT CAUSE ANALYSIS:",
        "",
        "1. O₂ SUCCESS FACTORS:",
        "   - May have more realistic boundary conditions",
        "   - Reaeration provides stabilizing source term",
        "   - Gas exchange equilibrium helps",
        "   - Less sensitive to transport numerical errors",
        "",
        "2. OTHER SPECIES FAILURE FACTORS:",
        "   - Boundary condition values may be unrealistic",
        "   - Dispersion coefficients may be inappropriate",
        "   - Mass conservation violations",
        "   - Strong sensitivity to transport errors",
        "",
        "RECOMMENDED FIXES:",
        "",
        "PRIORITY 1: Boundary condition validation",
        "   - Verify upstream/downstream values against literature",
        "   - Check for missing or corrupted boundary files",
        "   - Compare boundary gradients with field observations",
        "",
        "PRIORITY 2: Dispersion coefficient validation",
        "   - Check if dispersion values are within estuarine range (10-1000 m²/s)",
        "   - Verify Elder's formula implementation",
        "   - Test sensitivity to dispersion magnitude",
        "",
        "PRIORITY 3: Mass conservation fixes",
        "   - Identify species with mass violations",
        "   - Check numerical solver conservation properties",
        "   - Verify boundary flux calculations",
        "",
        "PRIORITY 4: Learn from O₂ success",
        "   - Extract O₂ boundary condition patterns",
        "   - Apply O₂ success factors to other species",
        "   - Test other species with O₂-like boundary conditions",
        "",
        "VALIDATION TARGETS:",
        "- All species R² > 0.6 (currently only O₂ achieves this)",
        "- Mass conservation error < 1% for all species",
        "- Realistic boundary condition gradients",
        "- Dispersion coefficients within 10-1000 m²/s range",
    ])
    
    # Save report
    report_file = "OUT/transport_correlation_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Saved diagnostic report: {report_file}")
    
    return '\n'.join(report)

def main():
    """Main diagnostic function"""
    print("🔍 TRANSPORT CORRELATION DIAGNOSIS")
    print("=" * 50)
    
    try:
        # Load data
        species_data = load_simulation_results()
        cem_df = load_cem_observations()
        
        if not species_data:
            print("❌ No species data available for analysis")
            return
        
        # Analyze boundary conditions
        boundary_analysis = analyze_boundary_conditions()
        
        # Analyze dispersion coefficients
        dispersion_analysis = calculate_dispersion_coefficients()
        
        # Compare correlations
        correlations = compare_species_correlations(species_data, cem_df)
        
        # Analyze O₂ success
        analyze_o2_success(species_data, correlations)
        
        # Check mass conservation
        conservation_results = check_mass_conservation(species_data)
        
        # Create diagnostic plots and report
        create_diagnostic_plots(species_data, correlations, boundary_analysis)
        report = generate_diagnostic_report(correlations, boundary_analysis, conservation_results)
        
        print("\n" + "=" * 50)
        print("🎯 DIAGNOSIS COMPLETE")
        print("=" * 50)
        
        if correlations:
            good_species = [s for s, d in correlations.items() if d['r2'] >= 0.6]
            poor_species = [s for s, d in correlations.items() if d['r2'] < 0.3]
            
            print(f"\nGOOD PERFORMERS (R² ≥ 0.6): {good_species}")
            print(f"POOR PERFORMERS (R² < 0.3): {poor_species}")
        
        print("\nFILES GENERATED:")
        print("- OUT/transport_correlation_diagnosis.png")
        print("- OUT/transport_correlation_report.txt")
        
        print("\n🔧 NEXT STEPS:")
        print("1. Fix boundary condition files for poor-performing species")
        print("2. Validate mass conservation")
        print("3. Apply O₂ success patterns to other species")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    main()