"""
Result output module for the JAX C-GEM model.

This module handles saving simulation results in various formats including
high-performance NPZ format and traditional CSV format for compatibility.

Features:
- Fast NPZ output for large datasets
- CSV output for compatibility with legacy tools
- Compressed storage options
- Metadata and statistics generation
- Error handling and progress reporting

Author: Nguyen Truong An
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any
from pathlib import Path

# Import from core module
try:
    from ..core.model_config import SPECIES_NAMES
except ImportError:
    # Fallback if not available
    SPECIES_NAMES = ['O2', 'NO3', 'NH4', 'PO4', 'Si', 'DIC', 'ALK', 'TSS', 'Phy', 'Zoo']

# Import performance profiler if available
try:
    from ..performance.performance_profiler import get_profiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    # Fallback profiler
    def get_profiler():
        class NullProfiler:
            def start_timer(self, name): pass
            def end_timer(self, name): pass
        return NullProfiler()

def save_results_as_npz(results: Dict[str, Any], output_dir: str = "OUT"):
    """
    Save complete simulation results in high-performance NPZ format.
    
    This function is optimized for fast writing and compact storage of the complete
    simulation history. It converts lists of JAX arrays into stacked NumPy arrays
    and saves them in compressed NPZ format.
    
    Args:
        results: Dictionary containing complete simulation results
        output_dir: Directory to save the NPZ file
    """
    profiler = get_profiler()
    profiler.start_timer('npz_saving')
    
    print("💾 Saving results in high-performance NPZ format...")
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract time data from results structure
    if 'time' in results:
        time_array = np.array(results['time'])
    elif 'hydro' in results and 'time' in results['hydro']:
        time_array = np.array(results['hydro']['time'])
    else:
        print("⚠️ Warning: No time data found in results")
        # Create default time array
        if 'hydro' in results and len(results['hydro']) > 0:
            first_var = list(results['hydro'].values())[0]
            if hasattr(first_var, 'shape') and len(first_var.shape) > 0:
                n_time = first_var.shape[0] if len(first_var.shape) > 1 else len(first_var)
            else:
                n_time = len(first_var)
        else:
            n_time = 100
        time_array = np.arange(n_time) / 48.0  # Assuming 30-min intervals    # Convert hydrodynamic data
    print("  ⏩ Processing hydrodynamic data...")
    hydro_data = {}
    for var_name, var_list in results['hydro'].items():
        if len(var_list) > 0:
            print(f"    Converting {var_name}...")
            # Stack list of JAX arrays into single NumPy array
            hydro_data[var_name] = np.stack([np.array(step) for step in var_list])
    
    # Convert transport/species data
    print("  ⏩ Processing species data...")
    species_data = {}
    for species_name, species_list in results['transport'].items():
        if len(species_list) > 0:
            print(f"    Converting {species_name}...")
            # Stack list of JAX arrays into single NumPy array
            species_array = np.stack([np.array(step) for step in species_list])
            species_data[species_name] = species_array
            
            # DEBUG: Check salinity values during NPZ creation
            if species_name == 'S':
                print(f"    🧪 SALINITY DEBUG IN NPZ CREATION:")
                print(f"       Final time step shape: {species_array[-1].shape}")
                print(f"       Index 0 (should be mouth): {species_array[-1, 0]:.3f} PSU")
                print(f"       Index -1 (should be head): {species_array[-1, -1]:.3f} PSU")
                if species_array[-1, 0] > species_array[-1, -1]:
                    print(f"       ✅ Correct gradient in NPZ writer")
                else:
                    print(f"       ❌ INVERTED gradient in NPZ writer!")
    
    # Save main results file
    main_file = os.path.join(output_dir, "complete_simulation_results.npz")
    print(f"  💾 Writing main results to {main_file}...")
    
    # Combine all data into single NPZ file
    all_data = {
        'time': time_array,
        **hydro_data,
        **species_data
    }
    
    # Add metadata
    metadata = {
        'simulation_days': float(time_array[-1]) if len(time_array) > 0 else 0.0,
        'time_steps': len(time_array),
        'grid_points': hydro_data[list(hydro_data.keys())[0]].shape[1] if hydro_data else 0,
        'format_version': '1.0',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hydro_variables': list(hydro_data.keys()),
        'species_variables': list(species_data.keys())
    }
    
    # Save metadata separately for easy access
    metadata_file = os.path.join(output_dir, "simulation_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save compressed NPZ file
    np.savez_compressed(main_file, **all_data)
    
    # Also save individual variable files for selective loading
    hydro_dir = os.path.join(output_dir, "hydro_vars")
    species_dir = os.path.join(output_dir, "species_vars")
    os.makedirs(hydro_dir, exist_ok=True)
    os.makedirs(species_dir, exist_ok=True)
    
    # Save individual hydrodynamic variables
    for var_name, var_data in hydro_data.items():
        var_file = os.path.join(hydro_dir, f"{var_name}.npz")
        np.savez_compressed(var_file, data=var_data, time=time_array)
    
    # Save individual species variables
    for species_name, species_array in species_data.items():
        species_file = os.path.join(species_dir, f"{species_name}.npz")
        np.savez_compressed(species_file, data=species_array, time=time_array)
    
    # Save summary statistics
    stats_file = os.path.join(output_dir, "simulation_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("JAX C-GEM Simulation Statistics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Simulation duration: {metadata['simulation_days']:.2f} days\n")
        f.write(f"Time steps: {metadata['time_steps']}\n")
        f.write(f"Grid points: {metadata['grid_points']}\n")
        f.write(f"Created: {metadata['created']}\n\n")
        
        # Variable ranges
        f.write("Variable Ranges:\n")
        f.write("-" * 20 + "\n")
        for var_name, var_data in hydro_data.items():
            f.write(f"{var_name}: [{np.min(var_data):.6e}, {np.max(var_data):.6e}]\n")
        
        for species_name, species_array in species_data.items():
            f.write(f"{species_name}: [{np.min(species_array):.6e}, {np.max(species_array):.6e}]\n")
    
    end_time = time.time()
    profiler.end_timer('npz_saving')
    
    # File size information
    main_file_size = os.path.getsize(main_file) / (1024 * 1024)  # MB
    
    print(f"✅ NPZ output completed in {end_time-start_time:.2f}s")
    print(f"📂 Main file: {main_file} ({main_file_size:.1f} MB)")
    print(f"📊 Metadata: {metadata_file}")
    print(f"📈 Statistics: {stats_file}")
    print(f"🔍 Individual variables: {hydro_dir}/, {species_dir}/")

def save_results_as_csv(results: Dict[str, Any], output_dir: str = "OUT"):
    """
    Save simulation results in CSV format for compatibility.
    
    This function maintains backward compatibility by saving results in the
    traditional CSV format used by the original C-GEM model.
    
    Args:
        results: Dictionary containing simulation results
        output_dir: Directory to save CSV files
    """
    profiler = get_profiler()
    profiler.start_timer('csv_saving')
    
    print("📝 Saving results in CSV format for compatibility...")
    start_time = time.time()
    
    # Create output directories
    os.makedirs(f"{output_dir}/Hydrodynamics", exist_ok=True)
    os.makedirs(f"{output_dir}/Reaction", exist_ok=True)
    
    # Extract time data from results structure
    if 'time' in results:
        time_days = np.array(results['time'])
    elif 'hydro' in results and 'time' in results['hydro']:
        time_days = np.array(results['hydro']['time'])
    else:
        print("⚠️ Warning: No time data found in results")
        # Create default time array
        if 'hydro' in results and len(results['hydro']) > 0:
            first_var = list(results['hydro'].values())[0]
            if hasattr(first_var, 'shape') and len(first_var.shape) > 0:
                n_time = first_var.shape[0] if len(first_var.shape) > 1 else len(first_var)
            else:
                n_time = len(first_var)
        else:
            n_time = 100
        time_days = np.arange(n_time) / 48.0  # Assuming 30-min intervals
    
    # Save hydrodynamic variables
    hydro_vars = ['H', 'U', 'D', 'PROF']
    available_hydro = [v for v in hydro_vars if v in results['hydro'] and len(results['hydro'][v]) > 0]
    
    print(f"📊 Writing {len(available_hydro)} hydrodynamic CSV files...")
    
    for var_name in available_hydro:
        print(f"  Writing {var_name}...")
        file_path = f"{output_dir}/Hydrodynamics/{var_name}.csv"
        
        try:
            # Get dimensions
            total_time_steps = len(time_days)
            space_dim = len(results['hydro'][var_name][0])
            
            # Create and fill data matrix
            data_matrix = np.zeros((total_time_steps, space_dim + 1))
            data_matrix[:, 0] = time_days
            
            for t_idx in range(total_time_steps):
                data_matrix[t_idx, 1:] = np.array(results['hydro'][var_name][t_idx])
            
            # Save efficiently
            np.savetxt(file_path, data_matrix, delimiter=',', 
                      fmt=['%.6f'] + ['%.6e'] * space_dim)
            
        except Exception as e:
            print(f"  ❌ Error saving {var_name}: {e}")
    
    # Save species concentration files
    print(f"📊 Writing {len(SPECIES_NAMES)} species concentration files...")
    
    for species_name in SPECIES_NAMES:
        if species_name in results['transport'] and len(results['transport'][species_name]) > 0:
            print(f"  Writing {species_name}...")
            file_path = f"{output_dir}/Reaction/{species_name}.csv"
            
            try:
                # Get dimensions
                total_time_steps = len(time_days)
                space_dim = len(results['transport'][species_name][0])
                
                # Create and fill data matrix
                data_matrix = np.zeros((total_time_steps, space_dim + 1))
                data_matrix[:, 0] = time_days
                
                for t_idx in range(total_time_steps):
                    data_matrix[t_idx, 1:] = np.array(results['transport'][species_name][t_idx])
                
                # Save efficiently
                np.savetxt(file_path, data_matrix, delimiter=',', 
                          fmt=['%.6f'] + ['%.6e'] * space_dim)
            
            except Exception as e:
                print(f"  ❌ Error saving {species_name}: {e}")
    
    end_time = time.time()
    profiler.end_timer('csv_saving')
    
    print(f"✅ CSV output completed in {end_time-start_time:.2f}s")
    print(f"📂 Hydrodynamics: {output_dir}/Hydrodynamics/")
    print(f"📂 Species: {output_dir}/Reaction/")
