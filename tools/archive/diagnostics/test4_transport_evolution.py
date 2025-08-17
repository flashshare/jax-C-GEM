#!/usr/bin/env python3
"""
Test 4: Transport Evolution Test

This test runs a very short simulation and tracks how salinity evolves over
just a few time steps to see where the inversion might be occurring.

Potential Issue: Some step in the transport process might be introducing
the inversion during the simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_parser import parse_model_config, parse_input_data_config
from core.data_loader import DataLoader
import jax.numpy as jnp

def run_short_transport_simulation():
    """Run a very short transport simulation and track salinity evolution."""
    
    print("=== TEST 4: TRANSPORT EVOLUTION TEST ===\n")
    
    # 1. Load configurations
    print("1. Loading configurations...")
    model_config = parse_model_config('config/model_config.txt')
    data_config = parse_input_data_config('config/input_data_config.txt')
    
    M = model_config['M']
    DELXI = model_config['DELXI']
    DELTI = model_config['DELTI']
    
    # 2. Calculate spatial grid
    x_vals = np.arange(M) * DELXI  # Distance from mouth (m)
    distance_km = x_vals / 1000.0
    
    # 3. Import necessary simulation components
    print("\n2. Importing simulation components...")
    try:
        from core.transport import apply_boundary_conditions_transport, transport_step, TransportState
        from core.hydrodynamics import hydrodynamic_step, HydroState
        
        print("Successfully imported simulation components")
    except ImportError as e:
        print(f"Error importing simulation components: {e}")
        return None
    
    # 4. Setup initial conditions
    print("\n3. Setting up initial conditions...")
    
    # Create simple hydrodynamic state with constant velocity and water level
    try:
        # Basic hydrodynamic state
        # Simplified flow from ocean to river (positive velocity)
        velocity = jnp.ones(M) * 0.1  # 0.1 m/s flow from ocean to river
        water_level = jnp.zeros(M)     # Zero water level for simplicity
        depth = jnp.ones(M) * 5.0      # 5m depth everywhere
        
        hydro_state = HydroState(
            u=velocity,
            h=water_level,
            d=depth
        )
        
        print("Created simplified hydrodynamic state")
        print(f"  Velocity: constant {velocity[0]:.2f} m/s")
        print(f"  Depth: constant {depth[0]:.2f} m")
        
        # Load data loader for boundary conditions
        data_loader = DataLoader(data_config)
        boundary_conditions = data_loader.get_boundary_conditions(0.0)  # At time=0
        
        # Extract salinity boundary conditions
        lb_sal = boundary_conditions.get('Downstream', {}).get('Sal', 30.0)
        ub_sal = boundary_conditions.get('Upstream', {}).get('Sal', 0.1)
        
        # Create simple boundary conditions dict
        simple_bc = {
            'LB_Sal': lb_sal,
            'UB_Sal': ub_sal
        }
        
        print(f"Boundary conditions:")
        print(f"  LB (mouth/ocean) salinity: {lb_sal} PSU")
        print(f"  UB (head/river) salinity: {ub_sal} PSU")
        
        # Initialize salinity with exponential gradient (realistic)
        # S = S0 * exp(-x/b) where b is a decay constant
        b = 50000  # Decay length scale in meters
        x_array = np.arange(M) * DELXI
        initial_salinity = lb_sal * np.exp(-x_array / b)
        
        # Initialize all other species with constant values
        MAXV = 17  # From model_config.py
        concentrations = jnp.ones((MAXV, M)) * 0.001
        
        # Set the salinity (species index 9)
        concentrations = concentrations.at[9, :].set(initial_salinity)
        
        # Create transport state
        transport_state = TransportState(
            concentrations=concentrations,
            boundary_conditions=simple_bc
        )
        
        print("Created transport state with exponential salinity gradient")
        print(f"  Initial salinity at mouth: {concentrations[9, 0]:.3f} PSU")
        print(f"  Initial salinity at head: {concentrations[9, -1]:.3f} PSU")
        
        if concentrations[9, 0] > concentrations[9, -1]:
            print("  ✅ Initial gradient is correct (high at mouth, low at head)")
        else:
            print("  ❌ Initial gradient is inverted (low at mouth, high at head)")
            
    except Exception as e:
        print(f"Error setting up initial conditions: {e}")
        return None
    
    # 5. Run short simulation
    print("\n4. Running short transport simulation...")
    
    # Store salinity profiles at each step
    num_steps = 10
    salinity_profiles = np.zeros((num_steps + 1, M))
    salinity_profiles[0, :] = np.array(concentrations[9, :])
    
    try:
        for step in range(1, num_steps + 1):
            # Apply boundary conditions
            concentrations_with_bc = apply_boundary_conditions_transport(
                concentrations, velocity, simple_bc, DELTI=DELTI, DELXI=DELXI
            )
            
            # Update transport state
            transport_state = TransportState(
                concentrations=concentrations_with_bc,
                boundary_conditions=simple_bc
            )
            
            # Take a transport step
            new_transport_state = transport_step(
                transport_state, hydro_state, DELTI, DELXI
            )
            
            # Extract updated concentrations
            concentrations = new_transport_state.concentrations
            
            # Store salinity profile
            salinity_profiles[step, :] = np.array(concentrations[9, :])
            
            # Report progress
            print(f"  Step {step}: Mouth={concentrations[9, 0]:.3f}, Head={concentrations[9, -1]:.3f}")
            
            # Check gradient direction
            if concentrations[9, 0] > concentrations[9, -1]:
                gradient_status = "✅ CORRECT"
            else:
                gradient_status = "❌ INVERTED"
                
            print(f"  Gradient status: {gradient_status}")
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        if 'concentrations' in locals():
            print(f"Last salinity values - Mouth: {concentrations[9, 0]:.3f}, Head: {concentrations[9, -1]:.3f}")
        return None
    
    # 6. Create visualization
    print("\n5. Creating visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Salinity profiles over time
    plt.subplot(2, 1, 1)
    
    # Plot initial profile
    plt.plot(distance_km, salinity_profiles[0, :], 'k-', linewidth=2, label='Initial')
    
    # Plot intermediate steps with increasing intensity
    cmap = plt.cm.viridis
    for i in range(1, num_steps):
        color = cmap(i / num_steps)
        plt.plot(distance_km, salinity_profiles[i, :], color=color, 
                 alpha=0.7, linewidth=1, label=f'Step {i}')
    
    # Plot final profile
    plt.plot(distance_km, salinity_profiles[-1, :], 'r-', linewidth=2, label='Final')
    
    plt.title('Salinity Profile Evolution')
    plt.xlabel('Distance from Mouth (km)')
    plt.ylabel('Salinity (PSU)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Mouth and head salinity over time
    plt.subplot(2, 1, 2)
    
    time_steps = np.arange(num_steps + 1)
    
    # Mouth salinity (index 0)
    plt.plot(time_steps, salinity_profiles[:, 0], 'b-o', linewidth=2, 
             label=f'Mouth (0 km)')
    
    # Mid-estuary salinity (middle index)
    mid_index = M // 2
    mid_distance = distance_km[mid_index]
    plt.plot(time_steps, salinity_profiles[:, mid_index], 'g-o', linewidth=2, 
             label=f'Mid ({mid_distance:.1f} km)')
    
    # Head salinity (last index)
    plt.plot(time_steps, salinity_profiles[:, -1], 'r-o', linewidth=2, 
             label=f'Head ({distance_km[-1]:.1f} km)')
    
    plt.title('Salinity at Key Locations Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Salinity (PSU)')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    os.makedirs('OUT/diagnostics', exist_ok=True)
    plt.tight_layout()
    plt.savefig('OUT/diagnostics/test4_transport_evolution.png', dpi=300)
    print("Saved visualization to OUT/diagnostics/test4_transport_evolution.png")
    
    # 7. Analyze results
    print("\n6. Analyzing results...")
    
    # Check if the gradient inverted at any point
    initial_is_correct = salinity_profiles[0, 0] > salinity_profiles[0, -1]
    final_is_correct = salinity_profiles[-1, 0] > salinity_profiles[-1, -1]
    
    if initial_is_correct and final_is_correct:
        print("✅ Gradient remained correct throughout simulation")
    elif not initial_is_correct and final_is_correct:
        print("🔄 Gradient was initially inverted but corrected during simulation")
    elif initial_is_correct and not final_is_correct:
        print("❌ Gradient was initially correct but became inverted during simulation")
        
        # Find when the inversion happened
        for step in range(1, num_steps + 1):
            if salinity_profiles[step, 0] <= salinity_profiles[step, -1]:
                print(f"  Inversion occurred at step {step}")
                break
    else:
        print("❌ Gradient was inverted throughout simulation")
    
    return {
        'distance_km': distance_km.tolist(),
        'salinity_profiles': salinity_profiles.tolist()
    }

if __name__ == "__main__":
    run_short_transport_simulation()
    print("\nTest 4 complete.")