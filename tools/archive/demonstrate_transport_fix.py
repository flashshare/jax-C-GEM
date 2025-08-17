#!/usr/bin/env python3
"""
Transport Fix Integration - Simplified

This script demonstrates the transport fix by directly replacing the
boundary condition application function and validating its effect
on a simple test case.
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

# Import needed modules
from core.config_parser import parse_model_config

def fix_demonstration():
    """Demonstrate the fix for the transport boundary conditions."""
    print("=== TRANSPORT FIX DEMONSTRATION ===\n")
    
    # Load configuration
    model_config = parse_model_config('config/model_config.txt')
    DELTI = float(model_config.get('DELTI', 180.0))
    DELXI = float(model_config.get('DELXI', 2000.0))
    M = model_config.get('M', 102)
    
    print(f"Grid: M={M}, DELXI={DELXI}m, DELTI={DELTI}s\n")
    
    # Create a salinity field with the CORRECT gradient (high at mouth, low at head)
    correct_salinity = jnp.linspace(30.0, 0.1, M)
    print("Initial correct salinity gradient:")
    print(f"  Mouth (index 0): {correct_salinity[0]:.3f} PSU")
    print(f"  Head (index {M-1}): {correct_salinity[-1]:.3f} PSU\n")
    
    # Create a salinity field with the INVERTED gradient (problem case)
    inverted_salinity = jnp.linspace(0.1, 30.0, M)
    print("Initial inverted salinity gradient (problem case):")
    print(f"  Mouth (index 0): {inverted_salinity[0]:.3f} PSU")
    print(f"  Head (index {M-1}): {inverted_salinity[-1]:.3f} PSU\n")
    
    # Create velocities for flood tide (incoming tide)
    velocities_flood = jnp.ones(M) * 0.5  # Positive = from ocean to river
    
    # Create fake boundary condition dictionary
    boundary_conditions = {
        'LB_Sal': 30.0,  # Ocean/mouth salinity
        'UB_Sal': 0.1    # River/head salinity
    }
    
    # Import both boundary condition implementations
    from core.transport import apply_boundary_conditions_transport as current_bc_fn
    
    # Create the fixed implementation right here for demonstration
    @jax.jit  
    def fixed_bc_transport(concentrations, velocities, boundary_conditions, DELTI, DELXI):
        """Fixed boundary condition implementation - EXACT C-GEM LOGIC."""
        MAXV, M = concentrations.shape
        c_new = concentrations.copy()
        
        # Get velocity at index 2 and M-2 (like C-GEM)
        u_lower = velocities[1] if M > 2 else velocities[0]  # U[2] in C-GEM
        u_upper = velocities[-2] if M > 2 else velocities[-1]  # U[M1] in C-GEM
        
        # Get boundary target values for salinity
        clb_target = boundary_conditions['LB_Sal']  # Ocean boundary (mouth)
        cub_target = boundary_conditions['UB_Sal']  # River boundary (head)
        
        # Process only salinity for this demo
        species_idx = 9  # Salinity index
        
        # Get current concentrations - EXACT C-GEM indexing equivalents
        co_1 = concentrations[species_idx, 0]  # co[1] in C-GEM (mouth)
        co_3 = concentrations[species_idx, 2] if M > 2 else concentrations[species_idx, 0]  # co[3] in C-GEM
        co_M = concentrations[species_idx, -1]  # co[M] in C-GEM (head) 
        co_M2 = concentrations[species_idx, -3] if M > 2 else concentrations[species_idx, -1]  # co[M2] in C-GEM
        
        # EXACT C-GEM BOUNDARY CONDITION APPLICATION
        
        # LOWER BOUNDARY (MOUTH) - co[1] in C-GEM = index 0 in JAX
        mouth_bc_inflow = co_1 - (co_1 - clb_target) * u_lower * DELTI / DELXI  # U[2]>=0.0 case
        mouth_bc_outflow = co_1 - (co_3 - co_1) * u_lower * DELTI / DELXI  # U[2]<0.0 case
        new_mouth = jnp.where(u_lower >= 0.0, mouth_bc_inflow, mouth_bc_outflow)
        c_new = c_new.at[species_idx, 0].set(new_mouth)
        
        # UPPER BOUNDARY (HEAD) - co[M] in C-GEM = index -1 in JAX
        head_bc_inflow = co_M - (co_M - co_M2) * u_upper * DELTI / DELXI  # U[M1]>=0.0 case
        head_bc_outflow = co_M - (cub_target - co_M) * u_upper * DELTI / DELXI  # U[M1]<0.0 case
        new_head = jnp.where(u_upper >= 0.0, head_bc_inflow, head_bc_outflow)
        c_new = c_new.at[species_idx, -1].set(new_head)
        
        return c_new
    
    # Create concentration arrays with just salinity
    MAXV = 17
    conc_correct = jnp.ones((MAXV, M)) * 1.0
    conc_correct = conc_correct.at[9].set(correct_salinity)
    
    conc_inverted = jnp.ones((MAXV, M)) * 1.0
    conc_inverted = conc_inverted.at[9].set(inverted_salinity)
    
    # Apply boundary conditions with current implementation
    print("Applying boundary conditions with current implementation:")
    current_result_correct = current_bc_fn(conc_correct, velocities_flood, boundary_conditions, DELTI, DELXI)
    current_result_inverted = current_bc_fn(conc_inverted, velocities_flood, boundary_conditions, DELTI, DELXI)
    
    current_sal_correct = np.array(current_result_correct[9])
    current_sal_inverted = np.array(current_result_inverted[9])
    
    print(f"  Correct gradient case:")
    print(f"    Mouth: {current_sal_correct[0]:.3f} PSU, Head: {current_sal_correct[-1]:.3f} PSU")
    print(f"    Gradient: {'CORRECT' if current_sal_correct[0] > current_sal_correct[-1] else 'INVERTED'}")
    
    print(f"  Inverted gradient case:")
    print(f"    Mouth: {current_sal_inverted[0]:.3f} PSU, Head: {current_sal_inverted[-1]:.3f} PSU")
    print(f"    Gradient: {'CORRECT' if current_sal_inverted[0] > current_sal_inverted[-1] else 'INVERTED'}\n")
    
    # Apply boundary conditions with fixed implementation
    print("Applying boundary conditions with fixed implementation:")
    fixed_result_correct = fixed_bc_transport(conc_correct, velocities_flood, boundary_conditions, DELTI, DELXI)
    fixed_result_inverted = fixed_bc_transport(conc_inverted, velocities_flood, boundary_conditions, DELTI, DELXI)
    
    fixed_sal_correct = np.array(fixed_result_correct[9])
    fixed_sal_inverted = np.array(fixed_result_inverted[9])
    
    print(f"  Correct gradient case:")
    print(f"    Mouth: {fixed_sal_correct[0]:.3f} PSU, Head: {fixed_sal_correct[-1]:.3f} PSU")
    print(f"    Gradient: {'CORRECT' if fixed_sal_correct[0] > fixed_sal_correct[-1] else 'INVERTED'}")
    
    print(f"  Inverted gradient case:")
    print(f"    Mouth: {fixed_sal_inverted[0]:.3f} PSU, Head: {fixed_sal_inverted[-1]:.3f} PSU")
    print(f"    Gradient: {'CORRECT' if fixed_sal_inverted[0] > fixed_sal_inverted[-1] else 'INVERTED'}\n")
    
    # Create visualization
    distance_km = np.arange(M) * DELXI / 1000  # Distance in km
    
    # Create output directory if needed
    os.makedirs("OUT/validation", exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Initially correct gradient case
    plt.subplot(2, 1, 1)
    plt.plot(distance_km, correct_salinity, 'k--', label="Initial")
    plt.plot(distance_km, current_sal_correct, 'r-', label="Current Implementation")
    plt.plot(distance_km, fixed_sal_correct, 'g-', label="Fixed Implementation")
    plt.title("Initially Correct Gradient Case")
    plt.ylabel("Salinity (PSU)")
    plt.legend()
    plt.grid(True)
    
    # Initially inverted gradient case
    plt.subplot(2, 1, 2)
    plt.plot(distance_km, inverted_salinity, 'k--', label="Initial")
    plt.plot(distance_km, current_sal_inverted, 'r-', label="Current Implementation")
    plt.plot(distance_km, fixed_sal_inverted, 'g-', label="Fixed Implementation")
    plt.title("Initially Inverted Gradient Case")
    plt.xlabel("Distance from Mouth (km)")
    plt.ylabel("Salinity (PSU)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("OUT/validation/transport_fix_demo.png", dpi=300)
    print("Saved visualization to OUT/validation/transport_fix_demo.png")
    
    # Key takeaways
    print("\nFINDINGS AND RECOMMENDATIONS:")
    print("============================")
    print("1. Original C-GEM applies boundary conditions BEFORE advection")
    print("2. JAX C-GEM applies boundary conditions AFTER advection")
    print("3. C-GEM checks velocity at indices 2 and M-2 for boundary conditions")
    print("4. JAX C-GEM checks velocity at indices 0 and M-1")
    print("\nTo fix the salinity gradient inversion:")
    print("1. Replace the boundary condition function with the fixed implementation")
    print("2. Apply boundary conditions BEFORE advection in transport_step")
    print("3. Ensure index mapping is correct between C-GEM and JAX-C-GEM")
    print("4. Validate with test cases and a full simulation")
    
    print("\nTest complete.")
    
if __name__ == "__main__":
    fix_demonstration()