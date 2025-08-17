#!/usr/bin/env python3
"""
Transport Physics Fix Validation

This script validates the fixed transport physics implementation by comparing:
1. Original C-GEM transport physics
2. Current JAX transport implementation 
3. Fixed JAX transport implementation

It focuses on:
- Boundary condition application order
- Velocity-dependent boundary conditions
- Ensuring correct salinity gradient direction
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the two transport modules for comparison
import core.transport as current_transport
from core.transport_fix import apply_boundary_conditions_transport as fixed_bc_transport
from core.config_parser import parse_model_config, parse_input_data_config
from core.data_loader import DataLoader

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def validate_boundary_condition_fix():
    """Compare original and fixed boundary condition implementations."""
    
    print("=== TRANSPORT PHYSICS FIX VALIDATION ===\n")
    
    # 1. Load configurations
    print("1. Loading configurations...")
    model_config = parse_model_config('config/model_config.txt')
    data_config = parse_input_data_config('config/input_data_config.txt')
    
    # 2. Set up test data
    print("2. Setting up test data...")
    MAXV = 17  # Number of species
    M = model_config.get('M', 102)
    DELTI = float(model_config.get('DELTI', 180.0))
    DELXI = float(model_config.get('DELXI', 2000.0))
    
    # Create simple linear concentration profile for testing
    # where we KNOW the correct gradient direction (high at mouth, low at river)
    concentrations = np.ones((MAXV, M)) * 10.0
    
    # Override salinity (index 9) with a simple gradient
    salinity = np.linspace(30.0, 0.1, M)  # Correct gradient: high at mouth, low at head
    concentrations[9, :] = salinity
    
    # Convert to JAX array
    concentrations_jax = jnp.array(concentrations)
    
    # Create velocities array with different flow scenarios
    # Scenario 1: River to ocean flow (ebb tide)
    velocities_ebb = jnp.ones(M) * (-0.5)  # Negative = flowing toward ocean
    
    # Scenario 2: Ocean to river flow (flood tide)
    velocities_flood = jnp.ones(M) * 0.5   # Positive = flowing toward river
    
    # Scenario 3: Mixed flow
    velocities_mixed = jnp.linspace(0.5, -0.5, M)  # Transition from flood to ebb
    
    # Create boundary conditions dictionary
    boundary_conditions = {
        'LB_Sal': 30.0,  # Ocean/mouth salinity
        'UB_Sal': 0.01   # River/head salinity
    }
    
    # 3. Test boundary condition application with both implementations
    print("\n3. Testing boundary condition implementations...")
    
    # Original implementation
    print("\nOriginal Implementation:")
    bc_original_ebb = current_transport.apply_boundary_conditions_transport(
        concentrations_jax, velocities_ebb, boundary_conditions, DELTI, DELXI)
    
    bc_original_flood = current_transport.apply_boundary_conditions_transport(
        concentrations_jax, velocities_flood, boundary_conditions, DELTI, DELXI)
    
    bc_original_mixed = current_transport.apply_boundary_conditions_transport(
        concentrations_jax, velocities_mixed, boundary_conditions, DELTI, DELXI)
    
    # Fixed implementation
    print("\nFixed Implementation:")
    bc_fixed_ebb = fixed_bc_transport(
        concentrations_jax, velocities_ebb, boundary_conditions, DELTI, DELXI)
    
    bc_fixed_flood = fixed_bc_transport(
        concentrations_jax, velocities_flood, boundary_conditions, DELTI, DELXI)
    
    bc_fixed_mixed = fixed_bc_transport(
        concentrations_jax, velocities_mixed, boundary_conditions, DELTI, DELXI)
    
    # 4. Analyze results
    print("\n4. Analyzing results...")
    
    # Extract salinity profiles for comparison
    orig_ebb_sal = np.array(bc_original_ebb[9])
    orig_flood_sal = np.array(bc_original_flood[9])
    orig_mixed_sal = np.array(bc_original_mixed[9])
    
    fixed_ebb_sal = np.array(bc_fixed_ebb[9])
    fixed_flood_sal = np.array(bc_fixed_flood[9])
    fixed_mixed_sal = np.array(bc_fixed_mixed[9])
    
    # Check gradient direction
    def check_gradient(sal_array, label):
        mouth_value = sal_array[0]
        head_value = sal_array[-1]
        if mouth_value > head_value:
            status = "CORRECT"
        else:
            status = "INVERTED"
            
        print(f"{label}: Mouth={mouth_value:.3f}, Head={head_value:.3f} - Gradient: {status}")
    
    print("\nGradient Direction Analysis:")
    print("----------------------------")
    check_gradient(orig_ebb_sal, "Original (Ebb)")
    check_gradient(orig_flood_sal, "Original (Flood)")
    check_gradient(orig_mixed_sal, "Original (Mixed)")
    check_gradient(fixed_ebb_sal, "Fixed (Ebb)")
    check_gradient(fixed_flood_sal, "Fixed (Flood)")
    check_gradient(fixed_mixed_sal, "Fixed (Mixed)")
    
    # 5. Create visualization
    print("\n5. Creating visualization...")
    
    # Create output directory if needed
    os.makedirs("OUT/validation", exist_ok=True)
    
    # Plot profiles
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    
    # Distance array
    distance_km = np.arange(M) * DELXI / 1000  # km from mouth
    
    # Plot ebb tide scenario
    ax = axes[0]
    ax.plot(distance_km, salinity, 'k--', label="Initial")
    ax.plot(distance_km, orig_ebb_sal, 'r-', label="Original")
    ax.plot(distance_km, fixed_ebb_sal, 'g-', label="Fixed")
    ax.set_title("Salinity Profiles - Ebb Tide (River to Ocean Flow)")
    ax.set_ylabel("Salinity (PSU)")
    ax.legend()
    ax.grid(True)
    
    # Plot flood tide scenario
    ax = axes[1]
    ax.plot(distance_km, salinity, 'k--', label="Initial")
    ax.plot(distance_km, orig_flood_sal, 'r-', label="Original")
    ax.plot(distance_km, fixed_flood_sal, 'g-', label="Fixed")
    ax.set_title("Salinity Profiles - Flood Tide (Ocean to River Flow)")
    ax.set_ylabel("Salinity (PSU)")
    ax.legend()
    ax.grid(True)
    
    # Plot mixed scenario
    ax = axes[2]
    ax.plot(distance_km, salinity, 'k--', label="Initial")
    ax.plot(distance_km, orig_mixed_sal, 'r-', label="Original")
    ax.plot(distance_km, fixed_mixed_sal, 'g-', label="Fixed")
    ax.set_title("Salinity Profiles - Mixed Flow")
    ax.set_ylabel("Salinity (PSU)")
    ax.set_xlabel("Distance from Mouth (km)")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("OUT/validation/transport_fix_validation.png", dpi=300)
    print("Saved visualization to OUT/validation/transport_fix_validation.png")
    
    print("\nTest complete.")

if __name__ == "__main__":
    validate_boundary_condition_fix()