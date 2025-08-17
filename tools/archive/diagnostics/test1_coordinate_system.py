#!/usr/bin/env python3
"""
Test 1: Coordinate System Verification

This test verifies how grid indices map to physical locations in both JAX C-GEM
and original C-GEM, checking if there's a mismatch in coordinate systems.

Potential Issue: JAX implementation may have the river at index 0 while original
C-GEM has ocean at index 0, or vice versa.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_parser import parse_model_config
from core.data_loader import DataLoader

def analyze_coordinates():
    """Analyze coordinate system in both implementations."""
    
    print("=== TEST 1: COORDINATE SYSTEM VERIFICATION ===\n")
    
    # 1. Analyze JAX C-GEM coordinate system
    print("1. Analyzing JAX C-GEM coordinate system...")
    model_config = parse_model_config('config/model_config.txt')
    M = model_config['M']
    DELXI = model_config['DELXI']
    
    # Load geometry
    print("2. Loading geometry data...")
    try:
        # Try to load geometry data manually
        geometry_csv = 'INPUT/Geometry/Geometry.csv'
        if os.path.exists(geometry_csv):
            geometry_data = np.loadtxt(geometry_csv, delimiter=',', skiprows=1)
            if geometry_data.shape[1] >= 2:  # At least 2 columns (km, width)
                distances_km = geometry_data[:, 0]
                width_array = geometry_data[:, 1]
                if geometry_data.shape[1] >= 3:
                    depth_array = geometry_data[:, 2]
                else:
                    depth_array = np.ones_like(width_array) * 5.0  # Default depth
                
                print(f"Geometry data loaded: {len(width_array)} points")
                print(f"First 3 width values: {width_array[:3]}")
                print(f"Last 3 width values: {width_array[-3:]}")
                
                geometry_data = {'B': width_array, 'H': depth_array}
            else:
                print(f"Invalid geometry data format")
                geometry_data = None
        else:
            print(f"Geometry file not found: {geometry_csv}")
            geometry_data = None
    except Exception as e:
        print(f"Error loading geometry: {e}")
        geometry_data = None
    
    # Calculate distance array
    x_vals = np.arange(M) * DELXI  # Distance from mouth (m)
    distance_km = x_vals / 1000.0
    
    print(f"\nGrid information:")
    print(f"  M (grid cells): {M}")
    print(f"  DELXI (cell size): {DELXI}m")
    print(f"  Total length: {distance_km[-1]:.1f}km")
    
    print(f"\nCoordinate mapping:")
    print(f"  Index 0: {distance_km[0]:.1f} km (should be mouth/ocean)")
    print(f"  Index M-1 ({M-1}): {distance_km[-1]:.1f} km (should be head/river)")
    
    # 2. Compare with C code define.h
    print("\n3. Analyzing original C-GEM coordinate system...")
    try:
        with open('deprecated/original-C-GEM/define.h', 'r') as f:
            define_h = f.read()
            
        # Extract key constants
        import re
        el_match = re.search(r'#define\s+EL\s+([0-9]+)', define_h)
        delxi_match = re.search(r'#define\s+DELXI\s+([0-9]+)', define_h)
        
        if el_match and delxi_match:
            c_el = int(el_match.group(1))
            c_delxi = int(delxi_match.group(1))
            c_length_km = c_el / 1000.0
            
            print(f"Original C-GEM constants:")
            print(f"  EL (total length): {c_el}m = {c_length_km:.1f}km")
            print(f"  DELXI (cell size): {c_delxi}m")
            
            # Calculate implied grid size
            implied_M = c_el // c_delxi + 1
            print(f"  Implied grid cells: {implied_M}")
            
            # Compare
            print(f"\nCoordinate system comparison:")
            print(f"  JAX C-GEM: {M} cells, {distance_km[-1]:.1f}km total")
            print(f"  Original C-GEM: ~{implied_M} cells, {c_length_km:.1f}km total")
            
            if abs(distance_km[-1] - c_length_km) < 0.1:
                print(f"  ✅ Total lengths match")
            else:
                print(f"  ❌ Total lengths differ")
                
            if M == implied_M:
                print(f"  ✅ Grid cell counts match")
            else:
                print(f"  ❌ Grid cell counts differ")
        else:
            print("Could not extract EL and DELXI from define.h")
    except Exception as e:
        print(f"Error analyzing C code: {e}")
    
    # 3. Try to find evidence of coordinate system differences
    print("\n4. Searching for coordinate system indicators...")
    
    # Look at geometry data again for clues
    if geometry_data is not None:
        width_ratio = width_array[0] / width_array[-1]
        print(f"Width ratio (first/last): {width_ratio:.2f}")
        print(f"  Expectation: Width should be LARGER at mouth than at river head")
        if width_ratio > 1:
            print(f"  ✅ Width correctly decreases from mouth to river (index 0 = mouth)")
        else:
            print(f"  ❌ Width increases from index 0 to index M-1 (suggests index 0 = river)")
    
    # 4. Create a visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Width profile
    if geometry_data is not None:
        plt.subplot(2, 1, 1)
        plt.plot(distance_km, width_array, 'b-', linewidth=2)
        plt.title('Width Profile (Expectation: Wide at ocean/mouth, narrow at river/head)')
        plt.xlabel('Distance from Index 0 (km)')
        plt.ylabel('Width (m)')
        plt.grid(True)
        
        # Annotate
        plt.annotate('Index 0', xy=(distance_km[0], width_array[0]), 
                     xytext=(distance_km[0]+5, width_array[0]), 
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.annotate(f'Index {M-1}', xy=(distance_km[-1], width_array[-1]), 
                     xytext=(distance_km[-1]-15, width_array[-1]), 
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Plot 2: Conceptual coordinate system
    plt.subplot(2, 1, 2)
    plt.plot([0, distance_km[-1]], [0, 0], 'k-', linewidth=3)
    plt.scatter([0, distance_km[-1]], [0, 0], s=100, c=['blue', 'green'])
    
    plt.annotate('Index 0 (mouth/ocean?)', xy=(0, 0), xytext=(0, 0.1), 
                 ha='center', va='bottom', fontsize=12)
    
    plt.annotate(f'Index {M-1} (head/river?)', xy=(distance_km[-1], 0), 
                 xytext=(distance_km[-1], 0.1), ha='center', va='bottom', fontsize=12)
    
    plt.title('Conceptual Coordinate System')
    plt.xlim(-5, distance_km[-1]+5)
    plt.ylim(-0.5, 0.5)
    plt.yticks([])
    plt.xlabel('Distance (km)')
    
    # Save figure
    os.makedirs('OUT/diagnostics', exist_ok=True)
    plt.tight_layout()
    plt.savefig('OUT/diagnostics/test1_coordinate_system.png', dpi=300)
    print("\nSaved visualization to OUT/diagnostics/test1_coordinate_system.png")
    
    return {
        'jax_grid_size': M,
        'jax_delxi': DELXI,
        'jax_total_length': distance_km[-1],
        'c_gem_total_length': c_length_km if 'c_length_km' in locals() else None
    }

if __name__ == "__main__":
    analyze_coordinates()
    print("\nTest 1 complete.")