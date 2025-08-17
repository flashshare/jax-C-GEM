#!/usr/bin/env python3
"""
TRANSPORT NUMERICAL STABILITY PATCH

This module patches the existing transport solver to fix oscillations
without breaking the existing infrastructure.

Author: Nguyen Truong An
"""
import jax.numpy as jnp
from jax import jit
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transport import compute_van_der_burgh_dispersion


@jit
def patch_dispersion_stability(disp_original: jnp.ndarray, 
                             u: jnp.ndarray, 
                             dx: float) -> jnp.ndarray:
    """
    Patch dispersion coefficient to ensure numerical stability
    
    Ensures Peclet number Pe = |u|*dx/D < 2 for stability
    """
    u_abs = jnp.abs(u)
    
    # Minimum dispersion for numerical stability
    min_disp_stability = u_abs * dx / 2.0 * 1.5  # Safety factor 1.5
    
    # Apply stability constraint
    disp_stable = jnp.maximum(disp_original, min_disp_stability)
    
    return disp_stable


@jit
def patch_concentration_smoothing(c: jnp.ndarray, smoothing_strength: float = 0.1) -> jnp.ndarray:
    """
    Apply mild spatial smoothing to reduce oscillations
    
    Uses a simple weighted average: c_new[i] = (1-α)*c[i] + α/2*(c[i-1] + c[i+1])
    """
    # Get neighboring values
    c_left = jnp.concatenate([c[:1], c[:-1]])    # c[i-1] 
    c_right = jnp.concatenate([c[1:], c[-1:]])   # c[i+1]
    
    # Weighted average (preserve boundaries)
    c_smooth = c.copy()
    
    # Apply smoothing to interior points only
    interior_smooth = (1 - smoothing_strength) * c + smoothing_strength * 0.5 * (c_left + c_right)
    
    # Keep boundaries unchanged, smooth interior
    c_smooth = c_smooth.at[1:-1].set(interior_smooth[1:-1])
    
    return c_smooth


@jit 
def patch_gradient_monotonicity(c: jnp.ndarray, enforce_monotonic: bool = True) -> jnp.ndarray:
    """
    Enforce monotonic gradients in estuarine species to prevent oscillations
    
    For estuarine systems:
    - Salinity should decrease monotonically upstream
    - Some nutrients should increase upstream
    """
    if not enforce_monotonic:
        return c
    
    # Simple gradient smoothing - remove local extrema
    c_smooth = c.copy()
    
    for i in range(1, len(c) - 1):
        # If this point is a local extremum, replace with average
        if (c[i] > c[i-1] and c[i] > c[i+1]) or (c[i] < c[i-1] and c[i] < c[i+1]):
            c_smooth = c_smooth.at[i].set((c[i-1] + c[i+1]) / 2.0)
    
    return c_smooth


def test_stability_patches():
    """Test the stability patches"""
    print("🔧 TESTING STABILITY PATCHES")
    print("="*40)
    
    # Test data with oscillations
    n = 20
    x = jnp.linspace(0, 1, n)
    
    # Oscillatory salinity (problematic)
    c_oscillatory = 35 * (1 - x) + 5 * jnp.sin(10 * jnp.pi * x)
    
    # Test dispersion stability
    u_test = jnp.ones(n) * 0.5
    dx_test = 2000.0
    disp_original = jnp.ones(n) * 50.0  # Too small
    
    disp_stable = patch_dispersion_stability(disp_original, u_test, dx_test)
    
    print(f"✅ Dispersion patch:")
    print(f"   Original: {disp_original[0]:.1f} m²/s")
    print(f"   Stable: {disp_stable[0]:.1f} m²/s")
    
    # Test concentration smoothing
    c_smooth = patch_concentration_smoothing(c_oscillatory, smoothing_strength=0.1)
    
    # Test monotonicity
    c_monotonic = patch_gradient_monotonicity(c_smooth)
    
    # Calculate oscillation metrics
    oscillation_original = jnp.sum(jnp.abs(jnp.diff(jnp.diff(c_oscillatory))))
    oscillation_patched = jnp.sum(jnp.abs(jnp.diff(jnp.diff(c_monotonic))))
    
    print(f"✅ Concentration smoothing:")
    print(f"   Original oscillation: {oscillation_original:.2f}")
    print(f"   Patched oscillation: {oscillation_patched:.2f}")
    print(f"   Improvement: {((oscillation_original - oscillation_patched)/oscillation_original*100):.1f}%")
    
    return True


if __name__ == "__main__":
    test_stability_patches()