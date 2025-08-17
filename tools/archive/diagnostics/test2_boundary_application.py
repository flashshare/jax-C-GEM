#!/usr/bin/env python3
"""
Test 2: Boundary Condition Application Test

This test verifies if boundary conditions are correctly applied in the JAX C-GEM
transport module, checking exact values and locations.

Potential Issue: Boundary conditions might be applied to wrong indices or 
with incorrect values or logic.
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_parser import parse_model_config, parse_input_data_config
from core.data_loader import DataLoader
from core.transport import apply_boundary_conditions_transport

def test_boundary_application():
    """Test boundary condition application logic."""
    
    print("=== TEST 2: BOUNDARY CONDITION APPLICATION ===\n")
    
    # 1. Load configurations
    print("1. Loading configurations...")
    model_config = parse_model_config('config/model_config.txt')
    data_config = parse_input_data_config('config/input_data_config.txt')
    
    M = model_config['M']
    DELXI = model_config['DELXI']
    MAXV = 17  # From model_config.py
    
    # 2. Setup test data
    print("2. Setting up test data...")
    # Create a simple velocity field (positive = flow from ocean to river)
    # Test both positive and negative velocity scenarios
    
    # Case 1: River to ocean flow (negative velocity at boundaries)
    u_river_to_ocean = jnp.ones(M) * -0.5  # -0.5 m/s
    
    # Case 2: Ocean to river flow (positive velocity at boundaries)
    u_ocean_to_river = jnp.ones(M) * 0.5   # 0.5 m/s
    
    # Case 3: Mixed flow (typical estuary)
    u_mixed = jnp.linspace(0.5, -0.5, M)   # Ocean to river transition
    
    # Create test concentration field (same for all cases)
    # Setup with proper shape but unrealistic values for clear testing
    concentrations = jnp.ones((MAXV, M)) * 10.0  # All species at 10.0
    
    # Get boundary values from input data
    data_loader = DataLoader(data_config)
    boundary_data = data_loader.get_boundary_conditions(0.0)  # Time=0
    
    # Extract salinity boundary values
    if 'Downstream' in boundary_data and 'Sal' in boundary_data['Downstream']:
        lb_sal = boundary_data['Downstream']['Sal']
        print(f"LB Salinity (mouth/ocean): {lb_sal} PSU")
    else:
        lb_sal = 30.0  # Default seawater
        print(f"Using default LB Salinity: {lb_sal} PSU")
        
    if 'Upstream' in boundary_data and 'Sal' in boundary_data['Upstream']:
        ub_sal = boundary_data['Upstream']['Sal'] 
        print(f"UB Salinity (river/head): {ub_sal} PSU")
    else:
        ub_sal = 0.1  # Default freshwater
        print(f"Using default UB Salinity: {ub_sal} PSU")
    
    # Create a simple boundary conditions dictionary for testing
    boundary_conditions = {
        'LB_Sal': lb_sal,  # Ocean/mouth
        'UB_Sal': ub_sal   # River/head
    }
    
    print("\n3. Testing boundary condition application...")
    
    # Test Case 1: River to ocean flow
    print("\nTest Case 1: River to Ocean Flow (u < 0)")
    sal_index = 9  # Index for salinity in the model
    
    # Before applying boundary conditions
    print(f"Before BC application:")
    print(f"  Salinity at index 0: {concentrations[sal_index, 0]:.3f}")
    print(f"  Salinity at index {M-1}: {concentrations[sal_index, M-1]:.3f}")
    
        # Apply boundary conditions for the river to ocean case
    try:
        new_concentrations = apply_boundary_conditions_transport(
            concentrations, 
            u_river_to_ocean, 
            boundary_conditions,
            DELTI=180.0,  # From model_config
            DELXI=2000.0  # From model_config
        )        # Check results
        print(f"After BC application (river to ocean):")
        print(f"  Salinity at index 0: {new_concentrations[sal_index, 0]:.3f}")
        print(f"  Salinity at index {M-1}: {new_concentrations[sal_index, M-1]:.3f}")
        
        # Check if boundary conditions were correctly applied
        if abs(new_concentrations[sal_index, 0] - 10.0) < 0.01:
            print("  ✅ BC correctly NOT applied at index 0 (expected for outflow)")
        else:
            print("  ❌ BC incorrectly applied at index 0 (changed despite outflow)")
            
        if abs(new_concentrations[sal_index, M-1] - ub_sal) < 0.01:
            print("  ✅ BC correctly applied at index M-1")
        else:
            print("  ❌ BC incorrectly applied at index M-1")
    except Exception as e:
        print(f"Error in Case 1: {e}")
    
    # Test Case 2: Ocean to river flow
    print("\nTest Case 2: Ocean to River Flow (u > 0)")
    
    # Reset concentrations
    concentrations = jnp.ones((MAXV, M)) * 10.0
    
    # Apply boundary conditions for the ocean to river case
    try:
        new_concentrations = apply_boundary_conditions_transport(
            concentrations, 
            u_ocean_to_river, 
            boundary_conditions,
            DELTI=180.0,  # From model_config
            DELXI=2000.0  # From model_config
        )
        
        # Check results
        print(f"After BC application (ocean to river):")
        print(f"  Salinity at index 0: {new_concentrations[sal_index, 0]:.3f}")
        print(f"  Salinity at index {M-1}: {new_concentrations[sal_index, M-1]:.3f}")
        
        # Check if boundary conditions were correctly applied
        if abs(new_concentrations[sal_index, 0] - lb_sal) < 0.01:
            print("  ✅ BC correctly applied at index 0")
        else:
            print("  ❌ BC incorrectly applied at index 0")
            
        if abs(new_concentrations[sal_index, M-1] - 10.0) < 0.01:
            print("  ✅ BC correctly NOT applied at index M-1 (expected for inflow)")
        else:
            print("  ❌ BC incorrectly applied at index M-1 (changed despite inflow)")
    except Exception as e:
        print(f"Error in Case 2: {e}")
    
    # Test Case 3: Mixed flow
    print("\nTest Case 3: Mixed Flow (u varies)")
    
    # Reset concentrations
    concentrations = jnp.ones((MAXV, M)) * 10.0
    
    # Apply boundary conditions for the mixed case
    try:
        new_concentrations = apply_boundary_conditions_transport(
            concentrations, 
            u_mixed, 
            boundary_conditions,
            DELTI=180.0,  # From model_config
            DELXI=2000.0  # From model_config
        )
        
        # Check results
        print(f"After BC application (mixed flow):")
        print(f"  Salinity at index 0: {new_concentrations[sal_index, 0]:.3f}")
        print(f"  Salinity at index {M-1}: {new_concentrations[sal_index, M-1]:.3f}")
        print(f"  u at index 0: {u_mixed[0]:.3f}")
        print(f"  u at index {M-1}: {u_mixed[M-1]:.3f}")
        
        # Check if boundary conditions were correctly applied
        if u_mixed[0] > 0 and abs(new_concentrations[sal_index, 0] - lb_sal) < 0.01:
            print("  ✅ BC correctly applied at index 0 (inflow)")
        elif u_mixed[0] <= 0 and abs(new_concentrations[sal_index, 0] - 10.0) < 0.01:
            print("  ✅ BC correctly NOT applied at index 0 (outflow)")
        else:
            print("  ❌ BC incorrectly applied at index 0")
            
        if u_mixed[M-1] < 0 and abs(new_concentrations[sal_index, M-1] - ub_sal) < 0.01:
            print("  ✅ BC correctly applied at index M-1 (inflow)")
        elif u_mixed[M-1] >= 0 and abs(new_concentrations[sal_index, M-1] - 10.0) < 0.01:
            print("  ✅ BC correctly NOT applied at index M-1 (outflow)")
        else:
            print("  ❌ BC incorrectly applied at index M-1")
    except Exception as e:
        print(f"Error in Case 3: {e}")
    
    # 4. Create visualization
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Case 1: River to ocean
    ax1 = axes[0]
    # Reset and apply for visualization
    concentrations = jnp.ones((MAXV, M)) * 10.0
    new_conc = apply_boundary_conditions_transport(concentrations, u_river_to_ocean, boundary_conditions, 
                                                   DELTI=180.0, DELXI=2000.0)
    
    x = np.arange(M)
    ax1.plot(x, u_river_to_ocean, 'b-', label='Velocity (River→Ocean)')
    ax1.plot(x, new_conc[sal_index], 'g-', label='Salinity After BC')
    ax1.plot(x, concentrations[sal_index], 'g--', label='Original Salinity', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('Case 1: River to Ocean Flow (u < 0)')
    ax1.legend()
    ax1.set_xlabel('Grid Index')
    ax1.grid(True)
    
    # Case 2: Ocean to river
    ax2 = axes[1]
    # Reset and apply for visualization
    concentrations = jnp.ones((MAXV, M)) * 10.0
    new_conc = apply_boundary_conditions_transport(concentrations, u_ocean_to_river, boundary_conditions, 
                                                   DELTI=180.0, DELXI=2000.0)
    
    ax2.plot(x, u_ocean_to_river, 'b-', label='Velocity (Ocean→River)')
    ax2.plot(x, new_conc[sal_index], 'g-', label='Salinity After BC')
    ax2.plot(x, concentrations[sal_index], 'g--', label='Original Salinity', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Case 2: Ocean to River Flow (u > 0)')
    ax2.legend()
    ax2.set_xlabel('Grid Index')
    ax2.grid(True)
    
    # Case 3: Mixed flow
    ax3 = axes[2]
    # Reset and apply for visualization
    concentrations = jnp.ones((MAXV, M)) * 10.0
    new_conc = apply_boundary_conditions_transport(concentrations, u_mixed, boundary_conditions, 
                                                   DELTI=180.0, DELXI=2000.0)
    
    ax3.plot(x, u_mixed, 'b-', label='Velocity (Mixed Flow)')
    ax3.plot(x, new_conc[sal_index], 'g-', label='Salinity After BC')
    ax3.plot(x, concentrations[sal_index], 'g--', label='Original Salinity', alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_title('Case 3: Mixed Flow')
    ax3.legend()
    ax3.set_xlabel('Grid Index')
    ax3.grid(True)
    
    # Save figure
    os.makedirs('OUT/diagnostics', exist_ok=True)
    plt.tight_layout()
    plt.savefig('OUT/diagnostics/test2_boundary_application.png', dpi=300)
    print("Saved visualization to OUT/diagnostics/test2_boundary_application.png")
    
    return {
        'boundary_conditions': boundary_conditions,
        'results': {
            'river_to_ocean': new_conc[sal_index, :].tolist() if 'new_conc' in locals() else None,
            'ocean_to_river': new_conc[sal_index, :].tolist() if 'new_conc' in locals() else None,
            'mixed_flow': new_conc[sal_index, :].tolist() if 'new_conc' in locals() else None
        }
    }

if __name__ == "__main__":
    test_boundary_application()
    print("\nTest 2 complete.")