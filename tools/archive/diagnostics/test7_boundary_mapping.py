#!/usr/bin/env python3
"""
Test 7: Forcing Data and Boundary Mapping Test

This test examines how boundary condition data is loaded, mapped, and applied
in both implementations to identify potential inversions or mapping issues.

Potential Issue: Boundary data might be mapped to the wrong locations,
or lower and upper boundary might be swapped.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_parser import parse_model_config, parse_input_data_config
from core.data_loader import DataLoader

def test_boundary_mapping():
    """Test how boundary data is mapped to the grid."""
    
    print("=== TEST 7: FORCING DATA AND BOUNDARY MAPPING ===\n")
    
    # 1. Load configurations
    print("1. Loading configurations...")
    model_config = parse_model_config('config/model_config.txt')
    data_config = parse_input_data_config('config/input_data_config.txt')
    
    # Extract key parameters
    M = model_config['M']
    DELXI = model_config['DELXI']
    
    # Calculate distance array
    x_vals = np.arange(M) * DELXI  # Distance from mouth (m)
    distance_km = x_vals / 1000.0
    
    # 2. Examine boundary data structure
    print("\n2. Examining boundary data structure...")
    
    # Load data for a specific time
    data_loader = DataLoader(data_config)
    boundary_data = data_loader.get_boundary_conditions(0.0)  # At time=0
    
    print("Boundary data structure:")
    for boundary_name in boundary_data:
        print(f"  {boundary_name}:")
        for var_name in boundary_data[boundary_name]:
            value = boundary_data[boundary_name][var_name]
            print(f"    {var_name}: {value}")
    
    # Extract salinity values specifically
    lb_sal = boundary_data.get('Downstream', {}).get('Sal', None)
    ub_sal = boundary_data.get('Upstream', {}).get('Sal', None)
    
    print("\nSalinity boundary values:")
    print(f"  Downstream (mouth/ocean): {lb_sal} PSU")
    print(f"  Upstream (head/river): {ub_sal} PSU")
    
    # Check if values make physical sense
    if lb_sal is not None and ub_sal is not None:
        if lb_sal > ub_sal:
            print("  ✅ Salinity boundary values are physically correct (higher at mouth)")
        else:
            print("  ❌ Salinity boundary values are inverted (higher at river)")
    
    # 3. Check input files directly
    print("\n3. Checking boundary input files directly...")
    
    # Find the downstream salinity file
    if 'boundaries' in data_config:
        for boundary in data_config['boundaries']:
            if boundary.get('name') == 'Downstream' or boundary.get('type') == 'LowerBoundary':
                sal_file = boundary.get('Sal')
                if sal_file:
                    print(f"Downstream salinity file: {sal_file}")
                    try:
                        with open(sal_file, 'r') as f:
                            lines = f.readlines()
                            # Read first few lines to check format
                            print(f"  First few lines:")
                            for i, line in enumerate(lines[:5]):
                                print(f"    {i+1}: {line.strip()}")
                    except Exception as e:
                        print(f"  Error reading file: {e}")
            
            if boundary.get('name') == 'Upstream' or boundary.get('type') == 'UpperBoundary':
                sal_file = boundary.get('Sal')
                if sal_file:
                    print(f"Upstream salinity file: {sal_file}")
                    try:
                        with open(sal_file, 'r') as f:
                            lines = f.readlines()
                            # Read first few lines to check format
                            print(f"  First few lines:")
                            for i, line in enumerate(lines[:5]):
                                print(f"    {i+1}: {line.strip()}")
                    except Exception as e:
                        print(f"  Error reading file: {e}")
    
    # 4. Examine C code boundary mapping
    print("\n4. Examining C code boundary mapping...")
    
    try:
        with open('deprecated/original-C-GEM/bcforcing.c', 'r') as f:
            bcforcing_c = f.read()
        
        # Look for boundary condition application
        bg_boundary_match = re.search(r'void\s+bgboundary\s*\(.*?\)\s*{(.*?)}', bcforcing_c, re.DOTALL)
        if bg_boundary_match:
            bg_boundary = bg_boundary_match.group(1)
            
            # Look for salinity assignments
            sal_assignments = re.findall(r'BC\[Sal\]\s*\[\s*(\w+)\s*\]\s*=\s*(.+?);', bg_boundary)
            
            print("C code salinity boundary assignments:")
            for location, value in sal_assignments:
                print(f"  BC[Sal][{location}] = {value}")
            
            # Check if UB and LB are mapped correctly
            if any('UB' in loc for loc, _ in sal_assignments) and any('LB' in loc for loc, _ in sal_assignments):
                print("  ✅ C code has distinct UB and LB assignments")
            else:
                print("  ⚠️ Could not identify UB/LB assignments clearly")
        else:
            print("Could not find bgboundary function")
    except Exception as e:
        print(f"Error examining bcforcing.c: {e}")
    
    # 5. Examine JAX boundary mapping
    print("\n5. Examining JAX boundary mapping...")
    
    try:
        with open('src/core/transport.py', 'r') as f:
            transport_py = f.read()
        
        # Look for boundary condition application
        bc_match = re.search(r'def\s+apply_boundary_conditions_transport\s*\(.*?\).*?:(.*?)return', transport_py, re.DOTALL)
        if bc_match:
            bc_func = bc_match.group(1)
            
            # Look for key lines with boundary application
            bc_lines = bc_func.strip().split('\n')
            
            print("JAX boundary condition application lines:")
            for i, line in enumerate(bc_lines):
                if 'boundary_conditions' in line and ('UB_' in line or 'LB_' in line):
                    print(f"  {i+1}: {line.strip()}")
                elif 'concentrations' in line and 'where' in line:
                    print(f"  {i+1}: {line.strip()}")
            
            # Check indices where boundaries are applied
            lb_indices = re.findall(r'concentrations\[.*?\]\[(\d+)\]\s*=.*?LB_', bc_func)
            ub_indices = re.findall(r'concentrations\[.*?\]\[(\d+|\-\d+)\]\s*=.*?UB_', bc_func)
            
            if lb_indices:
                print(f"  LB (mouth) applied at indices: {', '.join(lb_indices)}")
                if '0' in lb_indices:
                    print("  ✅ LB applied at index 0 (expected for mouth)")
                else:
                    print("  ⚠️ LB not applied at index 0")
            
            if ub_indices:
                print(f"  UB (river) applied at indices: {', '.join(ub_indices)}")
                if '-1' in ub_indices or str(M-1) in ub_indices:
                    print(f"  ✅ UB applied at end of grid (index {M-1})")
                else:
                    print(f"  ⚠️ UB not applied at end of grid (index {M-1})")
        else:
            print("Could not find apply_boundary_conditions_transport function")
    except Exception as e:
        print(f"Error examining transport.py: {e}")
    
    # 6. Test boundary application specifically
    print("\n6. Testing boundary application with controlled values...")
    
    try:
        from core.transport import apply_boundary_conditions_transport
        import jax.numpy as jnp
        
        # Create test data
        MAXV = 17  # From model_config.py
        test_concentrations = jnp.ones((MAXV, M)) * 5.0  # All species at 5.0
        test_velocity = jnp.ones(M) * 0.1  # Positive flow from ocean to river
        
        # Create test boundary conditions with very distinct values
        test_bc = {
            'LB_Sal': 35.0,  # Distinctive ocean value
            'UB_Sal': 0.5    # Distinctive river value
        }
        
        # Apply boundary conditions
        new_concentrations = apply_boundary_conditions_transport(
            test_concentrations, test_velocity, test_bc, 
            DELTI=180.0, DELXI=2000.0  # Standard values from model_config
        )
        
        # Check results at key indices
        sal_index = 9  # Salinity index
        
        print("Boundary application test results:")
        print(f"  Original salinity - uniform {test_concentrations[sal_index, 0]:.1f} PSU")
        print(f"  After BC application:")
        print(f"    Index 0 (mouth): {new_concentrations[sal_index, 0]:.1f} PSU")
        print(f"    Index {M-1} (head): {new_concentrations[sal_index, M-1]:.1f} PSU")
        
        # Check if boundaries are applied to correct locations
        if abs(new_concentrations[sal_index, 0] - test_bc['LB_Sal']) < 0.1:
            print("  ✅ LB correctly applied to index 0 (mouth)")
        else:
            print("  ❌ LB not applied to index 0")
            
        if abs(new_concentrations[sal_index, M-1] - test_concentrations[sal_index, M-1]) < 0.1:
            print("  ✅ UB correctly NOT applied to index M-1 (due to positive velocity)")
        elif abs(new_concentrations[sal_index, M-1] - test_bc['UB_Sal']) < 0.1:
            print("  ❌ UB incorrectly applied to index M-1 despite positive velocity")
        else:
            print("  ⚠️ Unexpected value at index M-1")
        
        # Test with reversed velocity
        test_velocity_reverse = jnp.ones(M) * -0.1  # Negative flow from river to ocean
        
        # Apply boundary conditions
        new_concentrations_reverse = apply_boundary_conditions_transport(
            test_concentrations, test_velocity_reverse, test_bc,
            DELTI=180.0, DELXI=2000.0  # Standard values from model_config
        )
        
        print("\n  With reversed velocity (river to ocean):")
        print(f"    Index 0 (mouth): {new_concentrations_reverse[sal_index, 0]:.1f} PSU")
        print(f"    Index {M-1} (head): {new_concentrations_reverse[sal_index, M-1]:.1f} PSU")
        
        # Check if boundaries are applied to correct locations
        if abs(new_concentrations_reverse[sal_index, 0] - test_concentrations[sal_index, 0]) < 0.1:
            print("  ✅ LB correctly NOT applied to index 0 (due to negative velocity)")
        elif abs(new_concentrations_reverse[sal_index, 0] - test_bc['LB_Sal']) < 0.1:
            print("  ❌ LB incorrectly applied to index 0 despite negative velocity")
        else:
            print("  ⚠️ Unexpected value at index 0")
            
        if abs(new_concentrations_reverse[sal_index, M-1] - test_bc['UB_Sal']) < 0.1:
            print("  ✅ UB correctly applied to index M-1 (head)")
        else:
            print("  ❌ UB not applied to index M-1")
        
    except Exception as e:
        print(f"Error testing boundary application: {e}")
    
    # 7. Create visualization
    print("\n7. Creating visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Physical layout with boundary mapping
    plt.subplot(2, 1, 1)
    
    # Draw the estuary schematic
    width = np.linspace(3, 1, M)  # Decreasing width from mouth to head
    
    # Left bank
    left_bank = np.zeros(M) - width/2
    plt.plot(distance_km, left_bank, 'k-', linewidth=2)
    
    # Right bank
    right_bank = np.zeros(M) + width/2
    plt.plot(distance_km, right_bank, 'k-', linewidth=2)
    
    # Mark mouth and head
    plt.scatter([distance_km[0]], [0], s=100, c='blue', zorder=5, label='Ocean/Mouth (Index 0)')
    plt.scatter([distance_km[-1]], [0], s=100, c='green', zorder=5, label='River/Head (Index M-1)')
    
    # Add labels
    plt.annotate('LB: Ocean/Salt Water', xy=(distance_km[0], 0), xytext=(distance_km[0], -1),
                 arrowprops=dict(facecolor='blue', shrink=0.05), ha='center')
    
    plt.annotate('UB: River/Fresh Water', xy=(distance_km[-1], 0), xytext=(distance_km[-1], -1),
                 arrowprops=dict(facecolor='green', shrink=0.05), ha='center')
    
    # Show salinity values
    if lb_sal is not None and ub_sal is not None:
        plt.text(distance_km[0], 1, f"LB_Sal = {lb_sal:.2f} PSU", 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        plt.text(distance_km[-1], 1, f"UB_Sal = {ub_sal:.2f} PSU", 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Estuary Schematic with Boundary Mapping')
    plt.xlabel('Distance from Mouth (km)')
    plt.ylabel('Width (normalized)')
    plt.grid(False)
    plt.legend()
    plt.ylim(-2, 2)
    
    # Plot 2: Boundary application logic
    plt.subplot(2, 1, 2)
    
    # Create flowchart-like diagram
    plt.axis('off')
    plt.title('Boundary Application Logic Flow')
    
    # Create basic flowchart
    flowchart = [
        "Load BC Data from Files",
        "↓",
        "Map to Boundary Conditions Dict",
        "↓",
        "Apply velocity-dependent boundary conditions",
        "↓",
        "Apply advection & dispersion"
    ]
    
    for i, step in enumerate(flowchart):
        y_pos = 0.8 - i*0.1
        if "↓" not in step:
            plt.text(0.5, y_pos, step, ha='center', va='center',
                     bbox=dict(facecolor='lightblue', edgecolor='blue', boxstyle='round,pad=0.5'))
    
    # Add key checks
    plt.text(0.1, 0.4, "if velocity[0] > 0:\n    Apply LB at index 0", ha='left', va='center',
             bbox=dict(facecolor='lightyellow', edgecolor='orange', boxstyle='round,pad=0.5'))
    
    plt.text(0.7, 0.4, "if velocity[M-1] < 0:\n    Apply UB at index M-1", ha='left', va='center',
             bbox=dict(facecolor='lightyellow', edgecolor='orange', boxstyle='round,pad=0.5'))
    
    # Add warning for potential issue
    plt.text(0.5, 0.15, "⚠️ Check for Coordinate System Mismatch\nOcean at index 0 vs. River at index 0", 
             ha='center', va='center', color='red', fontweight='bold',
             bbox=dict(facecolor='mistyrose', edgecolor='red', boxstyle='round,pad=0.5'))
    
    # Save figure
    os.makedirs('OUT/diagnostics', exist_ok=True)
    plt.tight_layout()
    plt.savefig('OUT/diagnostics/test7_boundary_mapping.png', dpi=300)
    print("Saved visualization to OUT/diagnostics/test7_boundary_mapping.png")
    
    # 8. Summarize findings
    print("\n8. Summarizing findings...")
    
    findings = {
        "boundary_values": {
            "lb_sal": float(lb_sal) if lb_sal is not None else None,
            "ub_sal": float(ub_sal) if ub_sal is not None else None,
            "physically_correct": lb_sal > ub_sal if lb_sal is not None and ub_sal is not None else None
        },
        "boundary_indices": {
            "lb_indices": lb_indices if 'lb_indices' in locals() else [],
            "ub_indices": ub_indices if 'ub_indices' in locals() else []
        },
        "test_results": {
            "lb_applied_correctly": True if 'new_concentrations' in locals() and 
                                           abs(new_concentrations[sal_index, 0] - test_bc['LB_Sal']) < 0.1 else False,
            "ub_not_applied_with_positive_velocity": True if 'new_concentrations' in locals() and 
                                                          abs(new_concentrations[sal_index, M-1] - test_concentrations[sal_index, M-1]) < 0.1 else False,
            "ub_applied_with_negative_velocity": True if 'new_concentrations_reverse' in locals() and 
                                                       abs(new_concentrations_reverse[sal_index, M-1] - test_bc['UB_Sal']) < 0.1 else False
        }
    }
    
    # Save findings to JSON
    with open('OUT/diagnostics/test7_findings.json', 'w') as f:
        json.dump(findings, f, indent=2)
    print("Saved findings to OUT/diagnostics/test7_findings.json")
    
    return findings

if __name__ == "__main__":
    test_boundary_mapping()
    print("\nTest 7 complete.")