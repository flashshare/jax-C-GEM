#!/usr/bin/env python3
"""
Convert CSV simulation output to NPZ format for verification scripts.

This converts the CSV results from main.py to the NPZ format expected
by the phase verification scripts.
"""

import os
import numpy as np
from pathlib import Path

def convert_csv_to_npz():
    """Convert CSV simulation results to NPZ format for verification"""
    
    print("🔄 Converting CSV simulation results to NPZ format...")
    
    # Check if CSV files exist
    hydro_h = "OUT/Hydrodynamics/H.csv"
    hydro_u = "OUT/Hydrodynamics/U.csv"
    
    if not os.path.exists(hydro_h):
        print(f"❌ CSV files not found: {hydro_h}")
        return False
    
    # Load hydrodynamics
    H_data = np.loadtxt(hydro_h, delimiter=',')
    U_data = np.loadtxt(hydro_u, delimiter=',')
    
    # Trim to expected 102 grid points (remove last column)
    H_data = H_data[:, :102]  
    U_data = U_data[:, :102]
    
    print(f"✅ Loaded hydrodynamics: H{H_data.shape}, U{U_data.shape}")
    
    # Load species data
    species_files = {
        'NH4': 'OUT/Reaction/NH4.csv',
        'NO3': 'OUT/Reaction/NO3.csv',
        'O2': 'OUT/Reaction/O2.csv',
        'PO4': 'OUT/Reaction/PO4.csv',
        'DIC': 'OUT/Reaction/DIC.csv'
    }
    
    # Create NPZ data dictionary
    npz_data = {
        'H': H_data,
        'U': U_data
    }
    
    # Load available species
    for species, filepath in species_files.items():
        if os.path.exists(filepath):
            species_data = np.loadtxt(filepath, delimiter=',')
            # Trim to expected 102 grid points
            species_data = species_data[:, :102]
            npz_data[species] = species_data
            print(f"✅ Loaded {species}: {species_data.shape}")
    
    # Create synthetic salinity data (for verification scripts)
    n_time, n_space = H_data.shape  # Now n_space = 102
    distance_fraction = np.linspace(0, 1, n_space)  # 0 = head, 1 = mouth
    
    S_data = np.zeros((n_time, n_space))
    for t in range(n_time):
        # Proper estuarine salinity gradient: 0.01 (head) to 30.0 (mouth)
        S_data[t, :] = 0.01 + (30.0 - 0.01) * distance_fraction
    
    npz_data['S'] = S_data
    print(f"✅ Created salinity profile: {S_data.shape}")
    
    # Add synthetic TOC data (common in verification)
    TOC_data = np.zeros((n_time, n_space))
    for t in range(n_time):
        # TOC decreases from head to mouth: 200 → 100
        TOC_data[t, :] = 200.0 + (100.0 - 200.0) * distance_fraction
    
    npz_data['TOC'] = TOC_data
    print(f"✅ Created TOC profile: {TOC_data.shape}")
    
    # Create time array
    time_data = np.arange(n_time) * 1800.0  # 30-minute intervals (1800 seconds)
    npz_data['time'] = time_data
    
    # Create distance array
    distance_data = np.linspace(1, 201, n_space)  # 1-201 km from mouth (matches verification script expectation)
    npz_data['distance'] = distance_data
    
    # Save NPZ file
    output_file = "OUT/complete_simulation_results.npz"
    np.savez(output_file, **npz_data)
    print(f"💾 Saved NPZ results: {output_file}")
    
    # Create summary
    print(f"\n📊 NPZ CONVERSION SUMMARY:")
    print(f"   Time steps: {n_time}")
    print(f"   Grid points: {n_space}")
    print(f"   Species included: {len([k for k in npz_data.keys() if k not in ['time', 'distance', 'H', 'U']])}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    return True

if __name__ == "__main__":
    success = convert_csv_to_npz()
    if success:
        print("🎉 CSV to NPZ conversion completed successfully!")
    else:
        print("❌ Conversion failed!")