"""
CRITICAL TRANSPORT SOLVER PATCH

This directly replaces the problematic TVD advection with first-order upwind
to eliminate oscillations.
"""
import jax.numpy as jnp
from jax import jit


@jit
def stable_upwind_advection(concentrations: jnp.ndarray, velocities: jnp.ndarray,
                           cross_sections: jnp.ndarray, DELTI: float, DELXI: float,
                           interface_indices: jnp.ndarray, cell_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Simple first-order upwind advection to eliminate oscillations
    
    This replaces the complex TVD scheme with a stable, monotonic scheme.
    """
    MAXV, M = concentrations.shape
    
    def stable_advect_species(c_old):
        """Apply first-order upwind advection - guaranteed stable"""
        c_new = c_old.copy()
        
        # Time step constraint for stability (CFL condition)
        max_velocity = jnp.max(jnp.abs(velocities))
        cfl_dt = 0.5 * DELXI / (max_velocity + 1e-10)  # Safety factor 0.5
        effective_dt = jnp.minimum(DELTI, cfl_dt)
        
        # Apply upwind scheme to interior points
        for i in range(1, M-1):
            u_left = velocities[i-1]    # Velocity at left face
            u_right = velocities[i]     # Velocity at right face
            
            # Flux computation using upwind values
            # Left face flux
            if u_left >= 0:
                flux_left = u_left * c_old[i-1]  # Use upstream value
            else:
                flux_left = u_left * c_old[i]    # Use upstream value
                
            # Right face flux
            if u_right >= 0:
                flux_right = u_right * c_old[i]    # Use upstream value
            else:
                flux_right = u_right * c_old[i+1]  # Use upstream value
            
            # Update using divergence theorem
            flux_divergence = (flux_right - flux_left) / DELXI
            c_new = c_new.at[i].set(c_old[i] - effective_dt * flux_divergence)
        
        # Apply gentle smoothing to reduce any remaining oscillations
        c_smoothed = apply_gentle_smoothing(c_new)
        
        return c_smoothed
    
    # Apply to all species
    result = jnp.zeros_like(concentrations)
    for species in range(MAXV):
        result = result.at[species].set(stable_advect_species(concentrations[species]))
    
    return result


@jit
def apply_gentle_smoothing(c: jnp.ndarray, strength: float = 0.05) -> jnp.ndarray:
    """Apply very gentle smoothing to reduce oscillations"""
    M = len(c)
    c_smooth = c.copy()
    
    # Apply weighted average to interior points
    for i in range(1, M-1):
        neighbor_avg = (c[i-1] + c[i+1]) / 2.0
        c_smooth = c_smooth.at[i].set((1 - strength) * c[i] + strength * neighbor_avg)
    
    return c_smooth


# Patch function to replace TVD with stable upwind
def patch_transport_module():
    """Apply the patch by replacing the problematic function"""
    import sys
    from pathlib import Path
    
    # Add the transport module to path and patch it
    transport_module_path = Path(__file__).parent / 'transport.py'
    
    print("🔧 PATCHING TRANSPORT MODULE FOR STABILITY")
    print(f"   Target: {transport_module_path}")
    
    # Read the existing transport file
    with open(transport_module_path, 'r') as f:
        transport_content = f.read()
    
    # Replace the TVD function with stable upwind
    if 'def tvd_advection(' in transport_content:
        # Find the start and end of the TVD function
        start_marker = 'def tvd_advection('
        start_pos = transport_content.find(start_marker)
        
        if start_pos != -1:
            # Find the end of the function (next def or class)
            remaining_content = transport_content[start_pos:]
            next_def_pos = remaining_content.find('\ndef ', 1)  # Find next function
            
            if next_def_pos != -1:
                end_pos = start_pos + next_def_pos
            else:
                # If no next function, assume it goes to end of file
                end_pos = len(transport_content)
            
            # Create the replacement
            replacement = '''def tvd_advection(concentrations: jnp.ndarray, velocities: jnp.ndarray,
                 cross_sections: jnp.ndarray, DELTI: float, DELXI: float,
                 interface_indices: jnp.ndarray, cell_indices: jnp.ndarray) -> jnp.ndarray:
    """PATCHED: Stable first-order upwind advection to eliminate oscillations."""
    from .transport_patch import stable_upwind_advection
    return stable_upwind_advection(concentrations, velocities, cross_sections, 
                                 DELTI, DELXI, interface_indices, cell_indices)'''
            
            # Apply the replacement
            patched_content = (transport_content[:start_pos] + 
                             replacement + 
                             transport_content[end_pos:])
            
            # Write back the patched content
            with open(transport_module_path, 'w') as f:
                f.write(patched_content)
                
            print("✅ Transport module patched successfully")
            return True
        else:
            print("❌ Could not find tvd_advection function")
            return False
    else:
        print("❌ TVD function not found in transport module")
        return False


if __name__ == "__main__":
    print("🔧 TRANSPORT STABILITY PATCH")
    print("="*40)
    success = patch_transport_module()
    if success:
        print("✅ Patch applied - ready to test")
    else:
        print("❌ Patch failed")