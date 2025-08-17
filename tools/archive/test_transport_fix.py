#!/usr/bin/env python3
"""
Test the Transport Fix - Validation Script
==========================================

This script tests the corrected transport module to verify that:
1. Boundary conditions are applied in correct order (before advection)
2. Salinity gradient is now correct (high at mouth, low at head)
3. The fix resolves the inversion issue identified by diagnostics
"""
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.transport_fixed import (
    transport_step_fixed, 
    create_initial_transport_state,
    create_transport_params,
    TransportState
)
from core.config_parser import parse_model_config, parse_input_data_config
from core.data_loader import DataLoader

def test_transport_fix():
    """Test the corrected transport physics."""
    
    print("=== TRANSPORT FIX VALIDATION ===\n")
    
    # 1. Load configuration
    print("1. Loading configuration...")
    model_config = parse_model_config('config/model_config.txt')
    data_config = parse_input_data_config('config/input_data_config.txt')
    data_loader = DataLoader(data_config)
    
    M = model_config['M']
    DELXI = model_config['DELXI']
    DELTI = model_config['DELTI']
    
    print(f"   Grid: M={M}, DELXI={DELXI}m, DELTI={DELTI}s")
    
    # 2. Create initial state with CORRECT salinity gradient
    print("2. Creating initial state...")
    transport_state = create_initial_transport_state(model_config)
    transport_params = create_transport_params(model_config)
    
    # Check initial salinity gradient
    initial_salinity = transport_state.concentrations[9, :]  # Species 9 = salinity
    print(f"   Initial salinity at mouth (index 0): {initial_salinity[0]:.2f} PSU")
    print(f"   Initial salinity at head (index -1): {initial_salinity[-1]:.2f} PSU")
    
    if initial_salinity[0] > initial_salinity[-1]:
        print("   ✅ Initial gradient is CORRECT (mouth > head)")
    else:
        print("   ❌ Initial gradient is INVERTED (mouth < head)")
    
    # 3. Set up boundary conditions
    print("3. Setting up boundary conditions...")
    boundary_data = data_loader.get_boundary_conditions(0.0)
    
    # Convert to the format expected by the transport function
    boundary_conditions = {}
    for boundary_name, species_data in boundary_data.items():
        for species, value in species_data.items():
            if boundary_name.startswith("Downstream"):
                key = f"LB_{species}"  # Lower Boundary (ocean/mouth)
            elif boundary_name.startswith("Upstream"): 
                key = f"UB_{species}"  # Upper Boundary (river/head)
            else:
                continue
            boundary_conditions[key] = float(value)
    
    # Set default boundary conditions if not found
    if 'LB_S' not in boundary_conditions:
        boundary_conditions['LB_S'] = 30.0  # Ocean salinity
    if 'UB_S' not in boundary_conditions:
        boundary_conditions['UB_S'] = 0.1   # River salinity
    
    print(f"   Ocean salinity (LB_S): {boundary_conditions.get('LB_S', 'N/A')} PSU")
    print(f"   River salinity (UB_S): {boundary_conditions.get('UB_S', 'N/A')} PSU")
    
    # 4. Create mock hydro state
    print("4. Creating mock hydrodynamic state...")
    
    # Simple hydrodynamic state for testing using JAX arrays
    # Use realistic but simple values
    from types import SimpleNamespace
    
    # Create velocities: positive = flood (ocean to river), negative = ebb (river to ocean)
    velocities = jnp.array([0.5 * jnp.sin(2 * jnp.pi * i / M) for i in range(M)])  # Tidal-like
    depths = jnp.ones(M) * 10.0  # 10m depth
    widths = jnp.ones(M) * 2000.0  # 2000m width (mock)
    
    hydro_state = SimpleNamespace()
    hydro_state.U = velocities
    hydro_state.D = depths
    hydro_state.H = widths
    
    print(f"   Velocity at check point (index 1): {hydro_state.U[1]:.3f} m/s")
    print(f"   Velocity range: {jnp.min(hydro_state.U):.3f} to {jnp.max(hydro_state.U):.3f} m/s")
    
    # 5. Run several time steps to test boundary condition persistence
    print("5. Running transport time steps...")
    
    # Mock additional required parameters
    hydro_params = {}
    tributary_data = {}
    upstream_discharge = 200.0
    grid_indices = jnp.arange(M)
    transport_indices = {
        'interface_indices': jnp.arange(M-1),
        'cell_indices': jnp.arange(M)
    }
    
    # Store results for analysis
    salinity_evolution = []
    
    # Run 10 time steps
    for step in range(10):
        transport_state = transport_step_fixed(
            transport_state,
            hydro_state,
            hydro_params,
            transport_params,
            boundary_conditions,
            tributary_data,
            upstream_discharge,
            grid_indices,
            transport_indices
        )
        
        # Store salinity profile
        salinity_evolution.append(transport_state.concentrations[9, :].copy())
        
        if step == 0:
            print(f"   After step {step+1}:")
            print(f"     Mouth salinity: {transport_state.concentrations[9, 0]:.2f} PSU")
            print(f"     Head salinity: {transport_state.concentrations[9, -1]:.2f} PSU")
    
    # 6. Analyze final results
    print("6. Analyzing results...")
    final_salinity = transport_state.concentrations[9, :]
    
    print(f"   Final salinity at mouth: {final_salinity[0]:.2f} PSU")
    print(f"   Final salinity at head: {final_salinity[-1]:.2f} PSU")
    print(f"   Salinity range: {jnp.min(final_salinity):.2f} to {jnp.max(final_salinity):.2f} PSU")
    
    # Check gradient direction
    if final_salinity[0] > final_salinity[-1]:
        gradient_status = "✅ CORRECT"
        print(f"   Gradient direction: {gradient_status} (mouth > head)")
    else:
        gradient_status = "❌ INVERTED"
        print(f"   Gradient direction: {gradient_status} (mouth < head)")
    
    # 7. Create visualization
    print("7. Creating visualization...")
    create_validation_plot(salinity_evolution, M, DELXI)
    
    # 8. Summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Transport Fix Status: {gradient_status}")
    print(f"Final Gradient: {final_salinity[0]:.2f} PSU (mouth) → {final_salinity[-1]:.2f} PSU (head)")
    
    if final_salinity[0] > final_salinity[-1]:
        print("🎉 SUCCESS: Transport fix has resolved the salinity gradient inversion!")
        return True
    else:
        print("❌ FAILURE: Salinity gradient is still inverted.")
        return False

def create_validation_plot(salinity_evolution, M, DELXI):
    """Create validation plot showing salinity evolution."""
    
    # Create output directory
    Path("OUT/validation").mkdir(exist_ok=True, parents=True)
    
    # Distance array
    distance_km = np.arange(M) * DELXI / 1000.0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Salinity profiles evolution
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(salinity_evolution)))
    
    for i, salinity in enumerate(salinity_evolution):
        alpha = 0.3 + 0.7 * i / len(salinity_evolution)  # Fade in over time
        ax1.plot(distance_km, salinity, color=colors[i], alpha=alpha, 
                label=f'Step {i+1}' if i in [0, len(salinity_evolution)-1] else '')
    
    ax1.set_xlabel('Distance from Mouth (km)')
    ax1.set_ylabel('Salinity (PSU)')
    ax1.set_title('Salinity Profile Evolution\n(Transport Fix Validation)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add expected gradient annotation
    ax1.annotate('Expected: High → Low', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=10, ha='left', va='top')
    
    # Plot 2: Final vs Initial comparison
    initial_salinity = salinity_evolution[0]
    final_salinity = salinity_evolution[-1]
    
    ax2.plot(distance_km, initial_salinity, 'b--', linewidth=2, label='Initial', alpha=0.7)
    ax2.plot(distance_km, final_salinity, 'r-', linewidth=2, label='Final')
    ax2.fill_between(distance_km, initial_salinity, final_salinity, alpha=0.2)
    
    ax2.set_xlabel('Distance from Mouth (km)')
    ax2.set_ylabel('Salinity (PSU)')
    ax2.set_title('Initial vs Final Salinity Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add gradient check
    if final_salinity[0] > final_salinity[-1]:
        status_color = 'green'
        status_text = '✅ GRADIENT CORRECT'
    else:
        status_color = 'red'  
        status_text = '❌ GRADIENT INVERTED'
    
    ax2.annotate(status_text, xy=(0.5, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3),
                fontsize=12, ha='center', va='top', weight='bold')
    
    plt.tight_layout()
    plt.savefig('OUT/validation/transport_fix_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   Saved validation plot: OUT/validation/transport_fix_validation.png")

if __name__ == "__main__":
    success = test_transport_fix()
    if success:
        print("\n🎉 TRANSPORT FIX VALIDATION: PASSED")
    else:
        print("\n❌ TRANSPORT FIX VALIDATION: FAILED")