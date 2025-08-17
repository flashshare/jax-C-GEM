"""
Simple Physics Validation for JAX C-GEM

A streamlined validation script that creates one comprehensive plot showing:
1. Longitudinal profiles: Salinity, tidal amplitude, velocity amplitude, water quality
   - Compares simulated vs theoretical profiles (salt intrusion, dilution theory)
2. Tidal dynamics: Variation of key parameters over tidal cycles at selected stations
3. Diagnostic assessment: Identifies issues in transport, hydrodynamics, biogeochemistry

This provides automatic post-simulation validation with theoretical comparisons
and debug diagnostics to assess model physics performance.

Author: Nguyen Truong An
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_simulation_results(results_file="OUT/complete_simulation_results.npz"):
    """Load simulation results from NPZ file."""
    
    if isinstance(results_file, str):
        results_file = Path(results_file)
    
    if not results_file.exists():
        # Try alternative locations
        alt_files = [
            Path("OUT/simulation_results.npz"),
            Path("OUT/complete_simulation_results.npz")
        ]
        for alt_file in alt_files:
            if alt_file.exists():
                results_file = alt_file
                break
        else:
            raise FileNotFoundError(f"No simulation results found. Tried: {results_file} and alternatives")
    
    print(f"📂 Loading results from: {results_file}")
    
    # Load NPZ data
    npz_data = np.load(results_file)
    
    # Extract key variables
    results = {
        'time': npz_data['time'],
        'H': npz_data['H'],        # Water levels
        'U': npz_data['U'],        # Velocities  
        'grid_km': np.arange(npz_data['H'].shape[1]) * 2.0,  # Grid in km (2km spacing)
    }
    
    # Load available species
    species_names = ['S', 'O2', 'NH4', 'NO3', 'PO4', 'TOC']  # Key species for validation
    for species in species_names:
        if species in npz_data:
            results[species] = npz_data[species]
    
    print(f"  ✅ Loaded {len(results['time'])} time steps, {len(results['grid_km'])} grid points")
    print(f"  📊 Available species: {[s for s in species_names if s in results]}")
    
    return results

def calculate_theoretical_salt_intrusion(grid_km, freshwater_discharge=250.0, 
                                        mouth_width=3000.0, mouth_depth=15.0,
                                        tidal_amplitude=4.43, dispersion_coeff=100.0):
    """
    Calculate theoretical salt intrusion length using Savenije (2005) equations.
    
    Parameters:
    - freshwater_discharge: River discharge [m³/s]
    - mouth_width: Estuary width at mouth [m]
    - mouth_depth: Average depth at mouth [m]
    - tidal_amplitude: Tidal amplitude [m]
    - dispersion_coeff: Longitudinal dispersion coefficient [m²/s]
    
    Returns theoretical salinity profile based on exponential decay.
    """
    
    # Savenije salt intrusion length formula (simplified)
    # L_s = K * sqrt(K * A / Q) where K is dispersion, A is cross-section, Q is discharge
    cross_section_mouth = mouth_width * mouth_depth  # m²
    
    # Theoretical salt intrusion length [km]
    salt_intrusion_length = (dispersion_coeff * np.sqrt(dispersion_coeff * cross_section_mouth / freshwater_discharge)) / 1000.0
    
    # Limit to reasonable range for Saigon River (typically 40-120 km)
    salt_intrusion_length = np.clip(salt_intrusion_length, 40, 120)
    
    # Create theoretical salinity profile with exponential decay
    # S(x) = S_mouth * exp(-x / L_s)
    salinity_mouth = 30.0  # Typical mouth salinity
    theoretical_salinity = salinity_mouth * np.exp(-grid_km / salt_intrusion_length)
    
    return theoretical_salinity, salt_intrusion_length

def calculate_theoretical_water_quality(grid_km, species_name, model_config, data_loader):
    """
    Calculate realistic theoretical water quality profiles based on actual model inputs.
    
    Uses real boundary conditions and tributary data to create theoretical dilution profiles
    that represent what the simulation SHOULD produce under conservative mixing.
    """
    try:
        # Get actual upstream and downstream boundary conditions
        # Use middle of simulation time for representative values
        mid_time = 15 * 24 * 3600  # 15 days in seconds
        
        # Get boundary conditions
        upstream_bc = data_loader.get_boundary_conditions(mid_time).get('Upstream', {})
        downstream_bc = data_loader.get_boundary_conditions(mid_time).get('Downstream', {})
        
        # Get tributary inputs
        tributary_data = data_loader.get_tributary_inputs(mid_time)
        
        # Get forcing data for discharge
        forcing = data_loader.get_forcing_data(mid_time)
        discharge = forcing.get('upstream_discharge', model_config.get('Q_AVAIL', 250.0))
        
        # Map species names to boundary condition keys and unit conversions
        species_info = {
            'NH4': {'key': 'NH4', 'conversion': 0.014},  # mmol/m³ → mg/L
            'NO3': {'key': 'NO3', 'conversion': 0.014},  # mmol/m³ → mg/L
            'PO4': {'key': 'PO4', 'conversion': 0.031},  # mmol/m³ → mg/L
            'TOC': {'key': 'TOC', 'conversion': 0.012},  # mmol/m³ → mg/L
            'O2': {'key': 'O2', 'conversion': 0.032},    # mmol/m³ → mg/L
            'S': {'key': 'Sal', 'conversion': 1.0}       # psu (no conversion)
        }
        
        if species_name not in species_info:
            # Default dilution profile for unmapped species
            return np.linspace(5.0, 1.0, len(grid_km))
        
        bc_key = species_info[species_name]['key']
        conversion_factor = species_info[species_name]['conversion']
        
        # Get actual concentrations from boundary conditions (in mmol/m³)
        upstream_conc_raw = upstream_bc.get(bc_key, 1.0)
        downstream_conc_raw = downstream_bc.get(bc_key, 0.0)
        
        # Convert to mg/L for comparison with model results
        upstream_conc = upstream_conc_raw * conversion_factor
        downstream_conc = downstream_conc_raw * conversion_factor
        
        # For salinity, use more realistic values (already in psu)
        if species_name == 'S':
            upstream_conc = 0.5  # Fresh water
            downstream_conc = 32.0  # Seawater
        
        print(f"  📊 {species_name}: Upstream={upstream_conc:.2f}, Downstream={downstream_conc:.2f} mg/L")
        
        # Calculate conservative mixing profile
        concentrations = np.zeros_like(grid_km)
        L_total = np.max(grid_km)  # Total length
        
        for i, distance in enumerate(grid_km):
            # Simple linear mixing fraction (0 at mouth, 1 at upstream)
            mixing_fraction = distance / L_total
            
            # Conservative mixing
            base_conc = downstream_conc + mixing_fraction * (upstream_conc - downstream_conc)
            
            # Add tributary contributions (convert tributary data too)
            tributary_addition = 0.0
            for trib_name, trib_data_dict in tributary_data.items():
                if bc_key in trib_data_dict:
                    # Get tributary location from config
                    trib_location_km = 62  # Default - should be read from config
                    if 'Dongnai' in trib_name:
                        trib_location_km = 62  # ~31 * 2km = 62km
                    elif 'Canal' in trib_name:
                        trib_location_km = 74  # ~37 * 2km = 74km
                    
                    # Add tributary influence downstream of input point
                    if distance <= trib_location_km:
                        # Exponential decay downstream
                        decay_length = 20.0  # km
                        decay_factor = np.exp(-(trib_location_km - distance) / decay_length)
                        trib_conc_raw = trib_data_dict[bc_key]
                        trib_conc = trib_conc_raw * conversion_factor  # Convert to mg/L
                        tributary_addition += trib_conc * 0.1 * decay_factor  # 10% contribution
            
            concentrations[i] = base_conc + tributary_addition
        
        return concentrations
        
    except Exception as e:
        print(f"  ⚠️  Error calculating theoretical {species_name}: {e}")
        # Fallback to simple linear profile
        return np.linspace(5.0, 1.0, len(grid_km))

def assess_physics_performance(profiles, theoretical_profiles, dynamics):
    """
    Assess model physics performance and provide diagnostic feedback.
    
    Returns diagnostic assessment of:
    1. Hydrodynamics quality
    2. Transport/dispersion accuracy  
    3. Biogeochemical realism
    4. Overall model performance
    """
    
    diagnostics = {
        'hydrodynamics': {'status': 'UNKNOWN', 'issues': [], 'score': 0.0},
        'transport': {'status': 'UNKNOWN', 'issues': [], 'score': 0.0},
        'biogeochemistry': {'status': 'UNKNOWN', 'issues': [], 'score': 0.0},
        'overall': {'status': 'UNKNOWN', 'score': 0.0, 'recommendations': []}
    }
    
    # === HYDRODYNAMICS ASSESSMENT ===
    hydro_issues = []
    hydro_score = 1.0
    
    # Check tidal amplitude range
    if 'tidal_amplitude' in profiles:
        tidal_range = np.max(profiles['tidal_amplitude']) - np.min(profiles['tidal_amplitude'])
        if tidal_range < 2.0:
            hydro_issues.append("Tidal range too small (<2m) - check boundary conditions")
            hydro_score -= 0.3
        elif tidal_range > 12.0:
            hydro_issues.append("Tidal range too large (>12m) - check tidal forcing")
            hydro_score -= 0.2
    
    # Check velocity realism
    if 'velocity_amplitude' in profiles:
        max_vel = np.max(profiles['velocity_amplitude'])
        if max_vel < 0.5:
            hydro_issues.append("Maximum velocities too low (<0.5 m/s) - weak tidal forcing")
            hydro_score -= 0.4
        elif max_vel > 4.0:
            hydro_issues.append("Maximum velocities too high (>4.0 m/s) - check channel geometry")
            hydro_score -= 0.2
    
    # Check flow reversals in dynamics
    if dynamics and 'velocity' in dynamics:
        velocity_data = dynamics['velocity']
        flow_reversal = np.any(velocity_data < 0) and np.any(velocity_data > 0)
        if not flow_reversal:
            hydro_issues.append("No tidal flow reversals detected - check tidal dynamics")
            hydro_score -= 0.5
    
    diagnostics['hydrodynamics']['issues'] = hydro_issues
    diagnostics['hydrodynamics']['score'] = max(0, hydro_score)
    diagnostics['hydrodynamics']['status'] = 'GOOD' if hydro_score > 0.7 else 'ISSUES' if hydro_score > 0.4 else 'POOR'
    
    # === TRANSPORT/SALINITY ASSESSMENT ===
    transport_issues = []
    transport_score = 1.0
    
    # Compare simulated vs theoretical salinity intrusion (use 'S' for salinity)
    if 'S' in profiles and 'S' in theoretical_profiles:
        sim_sal = profiles['S']
        theo_sal = theoretical_profiles['S']
        
        # Find salt intrusion lengths (0.5 salinity threshold for freshwater/salt boundary)
        sim_intrusion_idx = np.where(sim_sal > 0.5)[0]
        theo_intrusion_idx = np.where(theo_sal > 0.5)[0]
        
        sim_intrusion_km = profiles['grid_km'][sim_intrusion_idx[-1]] if len(sim_intrusion_idx) > 0 else 0
        theo_intrusion_km = profiles['grid_km'][theo_intrusion_idx[-1]] if len(theo_intrusion_idx) > 0 else 0
        
        # Compare intrusion lengths
        if abs(sim_intrusion_km - theo_intrusion_km) > 30:  # >30km difference
            transport_issues.append(f"Salt intrusion mismatch: simulated={sim_intrusion_km:.1f}km vs theoretical={theo_intrusion_km:.1f}km")
            transport_score -= 0.4
            
        # Check salinity gradient smoothness
        sal_gradient = np.gradient(sim_sal)
        if np.any(np.abs(sal_gradient) > 1.0):  # Very steep gradients
            transport_issues.append("Salinity gradients too steep - check dispersion coefficients")
            transport_score -= 0.3
    
    diagnostics['transport']['issues'] = transport_issues
    diagnostics['transport']['score'] = max(0, transport_score)
    diagnostics['transport']['status'] = 'GOOD' if transport_score > 0.7 else 'ISSUES' if transport_score > 0.4 else 'POOR'
    
    # === BIOGEOCHEMISTRY ASSESSMENT ===
    bio_issues = []
    bio_score = 1.0
    
    # Check water quality species against theoretical profiles
    wq_species = ['NH4', 'NO3', 'PO4', 'TOC', 'O2']
    for species in wq_species:
        if species in profiles and species in theoretical_profiles:
            sim_profile = profiles[species]
            theo_profile = theoretical_profiles[species]
            
            # Calculate relative error
            rel_error = np.mean(np.abs(sim_profile - theo_profile) / (theo_profile + 0.01))
            
            if rel_error > 2.0:  # >200% error
                bio_issues.append(f"{species} profile deviates significantly from dilution theory (error: {rel_error:.1f})")
                bio_score -= 0.2
                
            # Check for unrealistic values
            if np.any(sim_profile < 0):
                bio_issues.append(f"{species} has negative concentrations - check species bounds")
                bio_score -= 0.3
                
            if species == 'O2' and np.any(sim_profile > 15):  # O2 > 15 mg/L unrealistic
                bio_issues.append(f"{species} concentrations too high (>15 mg/L) - check reaeration")
                bio_score -= 0.2
    
    diagnostics['biogeochemistry']['issues'] = bio_issues
    diagnostics['biogeochemistry']['score'] = max(0, bio_score)
    diagnostics['biogeochemistry']['status'] = 'GOOD' if bio_score > 0.7 else 'ISSUES' if bio_score > 0.4 else 'POOR'
    
    # === OVERALL ASSESSMENT ===
    overall_score = (diagnostics['hydrodynamics']['score'] + 
                    diagnostics['transport']['score'] + 
                    diagnostics['biogeochemistry']['score']) / 3.0
                    
    diagnostics['overall']['score'] = overall_score
    
    if overall_score > 0.8:
        diagnostics['overall']['status'] = 'EXCELLENT'
        diagnostics['overall']['recommendations'] = ["Model physics are performing well", "Ready for field data validation"]
    elif overall_score > 0.6:
        diagnostics['overall']['status'] = 'GOOD'  
        diagnostics['overall']['recommendations'] = ["Minor physics issues detected", "Consider parameter fine-tuning"]
    elif overall_score > 0.4:
        diagnostics['overall']['status'] = 'ISSUES'
        diagnostics['overall']['recommendations'] = ["Significant physics issues", "Review transport and biogeochemical parameters"]
    else:
        diagnostics['overall']['status'] = 'POOR'
        diagnostics['overall']['recommendations'] = ["Major physics problems detected", "Check fundamental model setup"]
    
    return diagnostics

def calculate_longitudinal_profiles(results, warmup_fraction=0.3):
    """Calculate time-averaged longitudinal profiles after warmup period."""
    
    print("📊 Calculating time-averaged longitudinal profiles...")
    
    # Determine warmup cutoff
    total_steps = len(results['time'])
    warmup_steps = int(total_steps * warmup_fraction)
    
    profiles = {
        'grid_km': results['grid_km']
    }
    
    # Hydrodynamic profiles
    H_post_warmup = results['H'][warmup_steps:]
    U_post_warmup = results['U'][warmup_steps:]
    
    profiles['tidal_amplitude'] = np.std(H_post_warmup, axis=0)
    profiles['velocity_amplitude'] = np.std(U_post_warmup, axis=0)
    
    # Species profiles (convert to mg/L)
    unit_conversions = {
        'S': 1.0, 'O2': 0.032, 'NH4': 0.014, 'NO3': 0.014, 
        'PO4': 0.031, 'TOC': 0.012, 'SPM': 1.0
    }
    
    for species, conversion in unit_conversions.items():
        if species in results:
            species_data = results[species][warmup_steps:]
            profiles[species] = np.mean(species_data, axis=0) * conversion
    
    print(f"  ✅ Profiles calculated over {len(H_post_warmup)} time steps")
    print(f"  📍 Grid extent: {profiles['grid_km'][0]:.0f} to {profiles['grid_km'][-1]:.0f} km")
    
    return profiles

def calculate_theoretical_profiles(grid_km, model_config, data_loader):
    """Calculate all theoretical profiles for comparison using actual model inputs."""
    
    print("🧮 Calculating theoretical profiles using real model data...")
    
    theoretical_profiles = {
        'grid_km': grid_km
    }
    
    # Get actual discharge from model
    freshwater_discharge = model_config.get('Q_AVAIL', 250.0)
    
    # Theoretical salinity intrusion
    theo_sal, intrusion_length = calculate_theoretical_salt_intrusion(grid_km, freshwater_discharge)
    theoretical_profiles['salinity'] = theo_sal
    theoretical_profiles['salt_intrusion_length'] = intrusion_length
    
    # Theoretical water quality profiles using real boundary conditions
    wq_species = ['NH4', 'NO3', 'PO4', 'TOC', 'O2']
    
    print("  📊 Using real boundary conditions for theoretical profiles:")
    for species in wq_species:
        theoretical_profiles[species] = calculate_theoretical_water_quality(grid_km, species, model_config, data_loader)
    
    print(f"  ✅ Theoretical salt intrusion length: {intrusion_length:.1f} km")
    print("  ✅ Water quality dilution profiles calculated from model inputs")
    
    return theoretical_profiles

def extract_tidal_dynamics(results, station_locations_km=[20, 50, 100, 150], n_cycles=3):
    """Extract tidal dynamics at specific stations."""
    
    # Skip warmup and take only requested cycles
    start_idx = int(len(results['time']) * 0.3)
    
    # Estimate tidal period (assume ~12.5 hours = 45000 seconds)
    dt = results['time'][1] - results['time'][0] if len(results['time']) > 1 else 180
    tidal_period_steps = int(45000 / dt)  # ~12.5 hours in time steps
    end_idx = min(start_idx + n_cycles * tidal_period_steps, len(results['time']))
    
    # Extract time series
    time_subset = results['time'][start_idx:end_idx]
    time_hours = (time_subset - time_subset[0]) / 3600  # Convert to hours from start
    
    dynamics = {
        'time_hours': time_hours,
        'stations_km': station_locations_km
    }
    
    # Find closest grid points to requested stations
    grid_indices = []
    for station_km in station_locations_km:
        idx = np.argmin(np.abs(results['grid_km'] - station_km))
        grid_indices.append(idx)
    
    # Extract time series at stations
    dynamics['water_level'] = results['H'][start_idx:end_idx][:, grid_indices]
    dynamics['velocity'] = results['U'][start_idx:end_idx][:, grid_indices]
    
    # Extract species if available
    key_species = ['S', 'O2', 'NH4']
    for species in key_species:
        if species in results:
            dynamics[species] = results[species][start_idx:end_idx][:, grid_indices]
    
    return dynamics

def create_physics_validation_plot(profiles, theoretical_profiles, dynamics, diagnostics, 
                                  output_file="OUT/simple_physics_validation.png", show_plot=True):
    """Create comprehensive physics validation plot with theoretical comparisons."""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('JAX C-GEM Physics Validation - Simulated vs Theoretical Comparison', 
                 fontsize=18, fontweight='bold')
    
    # Color scheme
    sim_color = '#2E86AB'  # Blue for simulated
    theo_color = '#F24236' # Red for theoretical
    
    # ================================
    # TOP ROW: LONGITUDINAL PROFILES  
    # ================================
    
    # 1. Salinity Profile with Salt Intrusion Length
    ax1 = plt.subplot(3, 4, 1)
    if 'salinity' in profiles:
        plt.plot(profiles['grid_km'], profiles['salinity'], color=sim_color, linewidth=2.5, label='Simulated')
        if 'salinity' in theoretical_profiles:
            plt.plot(theoretical_profiles['grid_km'], theoretical_profiles['salinity'], 
                    color=theo_color, linewidth=2, linestyle='--', label='Theoretical')
            
            # Mark salt intrusion lengths (0.1 salinity threshold)
            sim_intrusion = np.where(profiles['salinity'] > 0.1)[0]
            if len(sim_intrusion) > 0:
                sim_intrusion_km = profiles['grid_km'][sim_intrusion[-1]]
                plt.axvline(sim_intrusion_km, color=sim_color, linestyle=':', alpha=0.7, 
                           label=f'Sim. intrusion: {sim_intrusion_km:.1f}km')
            
            if 'salt_intrusion_length' in theoretical_profiles:
                theo_intrusion_km = theoretical_profiles['salt_intrusion_length']
                plt.axvline(theo_intrusion_km, color=theo_color, linestyle=':', alpha=0.7,
                           label=f'Theo. intrusion: {theo_intrusion_km:.1f}km')
        
    plt.ylabel('Salinity (psu)', fontweight='bold')
    plt.title('Salt Intrusion Length Validation', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # 2. Tidal Amplitude
    ax2 = plt.subplot(3, 4, 2)
    if 'tidal_amplitude' in profiles:
        plt.plot(profiles['grid_km'], profiles['tidal_amplitude'], color=sim_color, linewidth=2.5, label='Simulated')
    plt.ylabel('Tidal Amplitude (m)', fontweight='bold')
    plt.title('Tidal Dynamics', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Velocity Amplitude  
    ax3 = plt.subplot(3, 4, 3)
    if 'velocity_amplitude' in profiles:
        plt.plot(profiles['grid_km'], profiles['velocity_amplitude'], color=sim_color, linewidth=2.5, label='Simulated')
    plt.ylabel('Velocity Amplitude (m/s)', fontweight='bold')
    plt.title('Flow Velocity', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Dissolved Oxygen
    ax4 = plt.subplot(3, 4, 4)
    if 'O2' in profiles:
        plt.plot(profiles['grid_km'], profiles['O2'], color=sim_color, linewidth=2.5, label='Simulated')
        if 'O2' in theoretical_profiles:
            plt.plot(theoretical_profiles['grid_km'], theoretical_profiles['O2'], 
                    color=theo_color, linewidth=2, linestyle='--', label='Dilution Theory')
    plt.ylabel('Dissolved O₂ (mg/L)', fontweight='bold')
    plt.title('Oxygen Profile', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # ================================
    # SECOND ROW: WATER QUALITY PROFILES
    # ================================
    
    # 5. Ammonium (NH4)
    ax5 = plt.subplot(3, 4, 5)
    if 'NH4' in profiles:
        plt.plot(profiles['grid_km'], profiles['NH4'], color=sim_color, linewidth=2.5, label='Simulated')
        if 'NH4' in theoretical_profiles:
            plt.plot(theoretical_profiles['grid_km'], theoretical_profiles['NH4'],
                    color=theo_color, linewidth=2, linestyle='--', label='Dilution Theory')
    plt.ylabel('NH₄-N (mg/L)', fontweight='bold')
    plt.title('Ammonium Profile', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # 6. Nitrate (NO3)
    ax6 = plt.subplot(3, 4, 6)
    if 'NO3' in profiles:
        plt.plot(profiles['grid_km'], profiles['NO3'], color=sim_color, linewidth=2.5, label='Simulated')
        if 'NO3' in theoretical_profiles:
            plt.plot(theoretical_profiles['grid_km'], theoretical_profiles['NO3'],
                    color=theo_color, linewidth=2, linestyle='--', label='Dilution Theory')
    plt.ylabel('NO₃-N (mg/L)', fontweight='bold')
    plt.title('Nitrate Profile', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # 7. Phosphate (PO4)
    ax7 = plt.subplot(3, 4, 7)
    if 'PO4' in profiles:
        plt.plot(profiles['grid_km'], profiles['PO4'], color=sim_color, linewidth=2.5, label='Simulated')
        if 'PO4' in theoretical_profiles:
            plt.plot(theoretical_profiles['grid_km'], theoretical_profiles['PO4'],
                    color=theo_color, linewidth=2, linestyle='--', label='Dilution Theory')
    plt.ylabel('PO₄-P (mg/L)', fontweight='bold')
    plt.title('Phosphate Profile', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # 8. Total Organic Carbon
    ax8 = plt.subplot(3, 4, 8)
    if 'TOC' in profiles:
        plt.plot(profiles['grid_km'], profiles['TOC'], color=sim_color, linewidth=2.5, label='Simulated')
        if 'TOC' in theoretical_profiles:
            plt.plot(theoretical_profiles['grid_km'], theoretical_profiles['TOC'],
                    color=theo_color, linewidth=2, linestyle='--', label='Dilution Theory')
    plt.ylabel('TOC (mg/L)', fontweight='bold')
    plt.title('Organic Carbon Profile', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # ================================
    # THIRD ROW: TIDAL DYNAMICS AT STATIONS
    # ================================
    
    if dynamics:
        # 9. Water Level Variations
        ax9 = plt.subplot(3, 4, 9)
        if 'water_level' in dynamics:
            for i, station_km in enumerate(dynamics['stations_km'][:3]):
                plt.plot(dynamics['time_hours'], dynamics['water_level'][:, i], 
                        linewidth=1.5, label=f'{station_km}km')
        plt.ylabel('Water Level (m)', fontweight='bold')
        plt.title('Tidal Water Levels', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # 10. Velocity Variations (Flow Reversals)
        ax10 = plt.subplot(3, 4, 10)
        if 'velocity' in dynamics:
            for i, station_km in enumerate(dynamics['stations_km'][:3]):
                plt.plot(dynamics['time_hours'], dynamics['velocity'][:, i], 
                        linewidth=1.5, label=f'{station_km}km')
            plt.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        plt.ylabel('Velocity (m/s)', fontweight='bold')
        plt.title('Tidal Flow Reversals', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # 11. Salinity Variations
        ax11 = plt.subplot(3, 4, 11)
        if 'S' in dynamics:
            for i, station_km in enumerate(dynamics['stations_km'][:3]):
                plt.plot(dynamics['time_hours'], dynamics['S'][:, i], 
                        linewidth=1.5, label=f'{station_km}km')
        plt.ylabel('Salinity (psu)', fontweight='bold')
        plt.title('Salinity Tidal Cycles', fontweight='bold')
        plt.xlabel('Time (hours)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # 12. Oxygen Variations
        ax12 = plt.subplot(3, 4, 12)
        if 'O2' in dynamics:
            for i, station_km in enumerate(dynamics['stations_km'][:3]):
                plt.plot(dynamics['time_hours'], dynamics['O2'][:, i], 
                        linewidth=1.5, label=f'{station_km}km')
        plt.ylabel('O₂ (mg/L)', fontweight='bold')
        plt.title('Oxygen Tidal Cycles', fontweight='bold')
        plt.xlabel('Time (hours)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Physics validation plot saved: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return output_file

def quick_physics_plot(results_file="OUT/complete_simulation_results.npz", show_plot=True):
    """
    Ultra-simple function to create physics validation plot with theoretical comparisons.
    
    Args:
        results_file: Path to simulation results
        show_plot: Whether to display the plot
        
    Returns:
        Path to saved plot file
    """
    return run_simple_physics_validation(results_file, show_plot=show_plot)

def run_simple_physics_validation(results_file="OUT/complete_simulation_results.npz",
                                 output_file="OUT/simple_physics_validation.png",
                                 show_plot=True):
    """
    Run complete physics validation with theoretical comparisons and diagnostics.
    
    This function provides comprehensive validation including:
    1. Salt intrusion length comparison (simulated vs theoretical)
    2. Water quality profiles vs dilution theory
    3. Diagnostic assessment of model components
    4. Debug messages for problem identification
    
    Args:
        results_file: Path to NPZ simulation results
        output_file: Path for validation plot
        show_plot: Whether to display plot
        
    Returns:
        Dict with validation results and diagnostics
    """
    
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE PHYSICS VALIDATION WITH THEORETICAL COMPARISON")
    print("="*80)
    
    try:
        # Load simulation results
        results = load_simulation_results(results_file)
        
        # Load model configuration and data loader for realistic theoretical profiles
        print("📋 Loading model configuration for theoretical comparison...")
        from config_parser import parse_model_config, parse_input_data_config
        from data_loader import DataLoader
        
        model_config = parse_model_config('config/model_config.txt')
        data_config = parse_input_data_config('config/input_data_config.txt')
        data_loader = DataLoader(data_config)
        
        # Calculate longitudinal profiles  
        profiles = calculate_longitudinal_profiles(results)
        
        # Calculate theoretical profiles for comparison using real model inputs
        theoretical_profiles = calculate_theoretical_profiles(profiles['grid_km'], model_config, data_loader)
        
        # Extract tidal dynamics
        dynamics = extract_tidal_dynamics(results)
        
        # Assess physics performance with diagnostics
        diagnostics = assess_physics_performance(profiles, theoretical_profiles, dynamics)
        
        # Print detailed debug diagnostics
        print_physics_diagnostics(diagnostics, profiles, theoretical_profiles)
        
        # Create validation plot with comparisons (no summary table)
        plot_file = create_physics_validation_plot(
            profiles, theoretical_profiles, dynamics, diagnostics, 
            output_file, show_plot
        )
        
        print("\n" + "="*80)
        print("✅ PHYSICS VALIDATION COMPLETE")
        print("="*80)
        
        return {
            'diagnostics': diagnostics,
            'profiles': profiles, 
            'theoretical_profiles': theoretical_profiles,
            'dynamics': dynamics,
            'plot_file': plot_file,
            'validation_passed': diagnostics['overall']['score'] > 0.6
        }
        
    except Exception as e:
        print(f"❌ Physics validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_physics_diagnostics(diagnostics, profiles, theoretical_profiles):
    """Print detailed diagnostic information for debugging."""
    
    print("\n🔍 DETAILED PHYSICS DIAGNOSTICS")
    print("-" * 50)
    
    # Overall assessment
    overall = diagnostics['overall']
    status_icon = {"EXCELLENT": "🟢", "GOOD": "🟡", "ISSUES": "🟠", "POOR": "🔴"}.get(overall['status'], "⚪")
    print(f"\n{status_icon} OVERALL MODEL PERFORMANCE: {overall['status']} (Score: {overall['score']:.3f})")
    
    # Hydrodynamics
    hydro = diagnostics['hydrodynamics']
    status_icon = {"GOOD": "🟢", "ISSUES": "🟠", "POOR": "🔴"}.get(hydro['status'], "⚪")
    print(f"\n{status_icon} HYDRODYNAMICS: {hydro['status']} (Score: {hydro['score']:.3f})")
    if hydro['issues']:
        for issue in hydro['issues']:
            print(f"   ⚠️  {issue}")
    else:
        print("   ✅ No hydrodynamic issues detected")
    
    # Transport/Salinity
    transport = diagnostics['transport']
    status_icon = {"GOOD": "🟢", "ISSUES": "🟠", "POOR": "🔴"}.get(transport['status'], "⚪")
    print(f"\n{status_icon} TRANSPORT/SALINITY: {transport['status']} (Score: {transport['score']:.3f})")
    
    # Add salt intrusion comparison
    if 'salinity' in profiles and 'salinity' in theoretical_profiles:
        sim_sal = profiles['salinity']
        sim_intrusion_idx = np.where(sim_sal > 0.1)[0]
        sim_intrusion_km = profiles['grid_km'][sim_intrusion_idx[-1]] if len(sim_intrusion_idx) > 0 else 0
        theo_intrusion_km = theoretical_profiles.get('salt_intrusion_length', 0)
        
        print(f"   📏 Salt Intrusion Lengths:")
        print(f"      Simulated: {sim_intrusion_km:.1f} km (0.1 salinity threshold)")
        print(f"      Theoretical: {theo_intrusion_km:.1f} km (Savenije equations)")
        print(f"      Difference: {abs(sim_intrusion_km - theo_intrusion_km):.1f} km")
        
        if abs(sim_intrusion_km - theo_intrusion_km) < 20:
            print("      ✅ Salt intrusion length within acceptable range (<20km difference)")
        else:
            print("      ⚠️  Salt intrusion length mismatch - check dispersion/discharge parameters")
    
    if transport['issues']:
        for issue in transport['issues']:
            print(f"   ⚠️  {issue}")
    else:
        print("   ✅ No transport issues detected")
    
    # Biogeochemistry
    bio = diagnostics['biogeochemistry']
    status_icon = {"GOOD": "🟢", "ISSUES": "🟠", "POOR": "🔴"}.get(bio['status'], "⚪")
    print(f"\n{status_icon} BIOGEOCHEMISTRY: {bio['status']} (Score: {bio['score']:.3f})")
    if bio['issues']:
        for issue in bio['issues']:
            print(f"   ⚠️  {issue}")
    else:
        print("   ✅ No biogeochemical issues detected")
    
    # Water quality vs theoretical comparison
    wq_species = ['NH4', 'NO3', 'PO4', 'TOC', 'O2']
    print(f"\n💧 WATER QUALITY vs DILUTION THEORY COMPARISON:")
    for species in wq_species:
        if species in profiles and species in theoretical_profiles:
            sim_profile = profiles[species]
            theo_profile = theoretical_profiles[species]
            
            # Calculate statistics
            sim_mean = np.mean(sim_profile)
            theo_mean = np.mean(theo_profile)
            rel_diff = abs(sim_mean - theo_mean) / theo_mean * 100 if theo_mean > 0 else 0
            
            status = "✅" if rel_diff < 50 else "⚠️" if rel_diff < 100 else "❌"
            print(f"   {status} {species}: Sim={sim_mean:.2f} vs Theo={theo_mean:.2f} mg/L (diff: {rel_diff:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in overall['recommendations']:
        print(f"   • {rec}")
    
    # Next steps based on performance
    if overall['score'] > 0.7:
        print(f"\n🎯 NEXT STEPS:")
        print("   ✅ Physics validation passed - model shows good theoretical agreement")
        print("   📋 Ready to proceed with field data validation (Phases 1-3)")
        print("   🚀 Consider running: 'Phase 1: Longitudinal Profiles' verification")
    else:
        print(f"\n⚠️  PHYSICS ISSUES DETECTED:")
        print("   🔧 Address identified physics problems before field validation")
        print("   📊 Review transport, dispersion, or biogeochemical parameters") 
        print("   🔄 Re-run simulation with corrected parameters")
        
    print("-" * 50)

if __name__ == "__main__":
    # Run simple physics validation when executed directly
    validation_results = run_simple_physics_validation(show_plot=True)
    
    if validation_results and validation_results['validation_passed']:
        print("✅ VALIDATION PASSED - Ready for field data comparison")
        exit(0)
    else:
        print("❌ VALIDATION ISSUES - Address physics problems before proceeding")
        exit(1)