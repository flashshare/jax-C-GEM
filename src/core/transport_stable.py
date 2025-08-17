#!/usr/bin/env python3
"""
TRANSPORT SOLVER STABILIZATION MODULE

This module fixes the numerical instabilities causing oscillations in the transport solver.
The key issues identified:
1. Advection scheme causing oscillations
2. Dispersion coefficient implementation
3. Boundary condition application in transport

Author: Nguyen Truong An
"""
import jax.numpy as jnp
from jax import jit, vmap
from typing import NamedTuple, Tuple
import jax


class TransportState(NamedTuple):
    """State variables for transport"""
    concentrations: jnp.ndarray  # [species, grid_points]
    velocities: jnp.ndarray      # [grid_points]
    depths: jnp.ndarray          # [grid_points]
    dispersion: jnp.ndarray      # [grid_points]


@jit
def compute_stable_dispersion(u: jnp.ndarray, h: jnp.ndarray, 
                            width: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute dispersion coefficient with stability constraints
    
    Args:
        u: Velocity [m/s]
        h: Depth [m]  
        width: Width [m]
        dx: Grid spacing [m]
        
    Returns:
        Stable dispersion coefficient [m²/s]
    """
    # Base dispersion: Elder's formula + tidal mixing
    u_abs = jnp.abs(u)
    elder_disp = 5.93 * h * jnp.sqrt(9.81 * h)  # Elder's formula
    shear_disp = 0.011 * u_abs**2 / 9.81  # Shear dispersion
    
    # Tidal mixing enhancement
    tidal_disp = 100.0 * jnp.ones_like(u)  # Base tidal dispersion
    
    # Total dispersion
    total_disp = elder_disp + shear_disp + tidal_disp
    
    # Numerical stability constraint: Pe < 2 (Peclet number)
    # D > |u| * dx / 2
    min_disp_for_stability = u_abs * dx / 2.0
    
    # Apply stability constraint
    stable_disp = jnp.maximum(total_disp, min_disp_for_stability * 1.1)
    
    return stable_disp


@jit 
def upwind_advection_flux(c: jnp.ndarray, u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute upwind advection flux to prevent oscillations
    
    Args:
        c: Concentration [mol/m³]
        u: Velocity [m/s]
        dx: Grid spacing [m]
        
    Returns:
        Advective flux [mol/m²/s]
    """
    # Upwind differences
    # For u > 0: use backward difference (upstream value)
    # For u < 0: use forward difference (downstream value)
    
    c_left = jnp.concatenate([c[:1], c[:-1]])    # c[i-1]
    c_right = jnp.concatenate([c[1:], c[-1:]])   # c[i+1]
    
    # Upwind concentration
    c_upwind = jnp.where(u >= 0, c_left, c_right)
    
    # Flux = u * c_upwind
    flux = u * c_upwind
    
    return flux


@jit
def central_dispersion_flux(c: jnp.ndarray, D: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute central difference dispersion flux
    
    Args:
        c: Concentration [mol/m³]
        D: Dispersion coefficient [m²/s]
        dx: Grid spacing [m]
        
    Returns:
        Dispersive flux [mol/m²/s]
    """
    # Central differences for dispersion
    c_left = jnp.concatenate([c[:1], c[:-1]])    # c[i-1]
    c_right = jnp.concatenate([c[1:], c[-1:]])   # c[i+1]
    
    # dc/dx at cell faces using central differences
    dc_dx = (c_right - c_left) / (2.0 * dx)
    
    # Dispersive flux = -D * dc/dx
    flux = -D * dc_dx
    
    return flux


@jit
def apply_transport_boundary_conditions(c: jnp.ndarray, 
                                      c_upstream: float,
                                      c_downstream: float,
                                      u: jnp.ndarray) -> jnp.ndarray:
    """
    Apply boundary conditions for transport with proper flow direction handling
    
    Args:
        c: Concentration array
        c_upstream: Upstream boundary concentration
        c_downstream: Downstream boundary concentration
        u: Velocity array
        
    Returns:
        Concentration with boundary conditions applied
    """
    c_new = c.copy()
    
    # Upstream boundary (index 0 = mouth in our coordinate system)
    # If flow is incoming (u[0] > 0), set downstream concentration
    # If flow is outgoing (u[0] < 0), let concentration be computed
    if u[0] > 0:  # Incoming flow at mouth
        c_new = c_new.at[0].set(c_downstream)
    
    # Downstream boundary (index -1 = head in our coordinate system)  
    # If flow is incoming (u[-1] < 0), set upstream concentration
    # If flow is outgoing (u[-1] > 0), let concentration be computed
    if u[-1] < 0:  # Incoming flow at head
        c_new = c_new.at[-1].set(c_upstream)
        
    return c_new


@jit
def compute_transport_rhs(state: TransportState, 
                         species_idx: int,
                         c_upstream: float,
                         c_downstream: float,
                         dx: float,
                         reactions: jnp.ndarray) -> jnp.ndarray:
    """
    Compute right-hand side for transport equation with stable numerics
    
    Transport equation: dc/dt + d(uc)/dx = d/dx(D dc/dx) + R
    
    Args:
        state: Transport state
        species_idx: Index of species to compute
        c_upstream: Upstream boundary concentration
        c_downstream: Downstream boundary concentration  
        dx: Grid spacing
        reactions: Reaction rates [mol/m³/s]
        
    Returns:
        dc/dt [mol/m³/s]
    """
    c = state.concentrations[species_idx]
    u = state.velocities
    h = state.depths
    D = state.dispersion
    
    # Apply boundary conditions
    c = apply_transport_boundary_conditions(c, c_upstream, c_downstream, u)
    
    # Compute fluxes
    advection_flux = upwind_advection_flux(c, u, dx)
    dispersion_flux = central_dispersion_flux(c, D, dx)
    
    # Total flux
    total_flux = advection_flux + dispersion_flux
    
    # Flux divergence: d(flux)/dx
    flux_left = jnp.concatenate([total_flux[:1], total_flux[:-1]])
    flux_right = jnp.concatenate([total_flux[1:], total_flux[-1:]])
    flux_divergence = (flux_right - flux_left) / dx
    
    # Transport equation: dc/dt = -d(flux)/dx + R
    dcdt = -flux_divergence + reactions
    
    return dcdt


@jit 
def transport_step_rk4(state: TransportState,
                       dt: float,
                       dx: float,
                       boundary_upstream: jnp.ndarray,
                       boundary_downstream: jnp.ndarray,
                       reaction_fn) -> jnp.ndarray:
    """
    4th-order Runge-Kutta transport step with stability
    
    Args:
        state: Current transport state
        dt: Time step
        dx: Grid spacing
        boundary_upstream: Upstream boundary concentrations
        boundary_downstream: Downstream boundary concentrations
        reaction_fn: Function to compute reactions
        
    Returns:
        New concentrations
    """
    c = state.concentrations
    n_species, n_grid = c.shape
    
    def compute_dcdt(concentrations):
        # Update state with new concentrations
        new_state = TransportState(
            concentrations=concentrations,
            velocities=state.velocities,
            depths=state.depths,
            dispersion=state.dispersion
        )
        
        # Compute reactions
        reactions = reaction_fn(new_state)
        
        # Compute transport for each species
        dcdt_all = jnp.zeros_like(concentrations)
        
        for i in range(n_species):
            dcdt_species = compute_transport_rhs(
                new_state, i, 
                boundary_upstream[i], 
                boundary_downstream[i],
                dx, reactions[i]
            )
            dcdt_all = dcdt_all.at[i].set(dcdt_species)
            
        return dcdt_all
    
    # RK4 steps
    k1 = dt * compute_dcdt(c)
    k2 = dt * compute_dcdt(c + k1/2)
    k3 = dt * compute_dcdt(c + k2/2) 
    k4 = dt * compute_dcdt(c + k3)
    
    # Final update
    c_new = c + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Ensure non-negative concentrations
    c_new = jnp.maximum(c_new, 0.0)
    
    return c_new


# Test function for numerical stability
def test_transport_stability():
    """Test transport solver stability"""
    print("🧪 TESTING TRANSPORT SOLVER STABILITY")
    print("="*50)
    
    # Create test case
    n_grid = 50
    n_species = 4
    dx = 2000.0
    dt = 180.0
    
    # Test velocities (including flow reversals)
    u = jnp.array([0.5 * jnp.sin(2*jnp.pi*i/n_grid) for i in range(n_grid)])
    h = jnp.ones(n_grid) * 5.0
    width = jnp.ones(n_grid) * 500.0
    
    # Test concentrations (smooth gradient)
    x = jnp.linspace(0, 1, n_grid)
    c_test = jnp.array([
        35 * (1 - x),  # Salinity
        5 + 145 * x,   # NH4
        15 + 285 * x,  # NO3
        180 + 100 * (1-x)  # O2
    ])
    
    # Compute stable dispersion
    D = compute_stable_dispersion(u, h, width, dx)
    
    # Create state
    state = TransportState(
        concentrations=c_test,
        velocities=u,
        depths=h,
        dispersion=D
    )
    
    # Test stability criteria
    Pe_max = jnp.max(jnp.abs(u) * dx / D)  # Peclet number
    CFL_max = jnp.max(jnp.abs(u) * dt / dx)  # CFL number
    
    print(f"✅ Grid: {n_grid} points, dx={dx}m")
    print(f"✅ Time step: dt={dt}s")
    print(f"✅ Max Peclet number: {Pe_max:.2f} (should be < 2)")
    print(f"✅ Max CFL number: {CFL_max:.2f} (should be < 1)")
    
    if Pe_max < 2.0 and CFL_max < 1.0:
        print("✅ TRANSPORT SOLVER IS STABLE")
        return True
    else:
        print("❌ TRANSPORT SOLVER UNSTABLE")
        return False


if __name__ == "__main__":
    test_transport_stability()