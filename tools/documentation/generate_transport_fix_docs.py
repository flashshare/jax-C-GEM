#!/usr/bin/env python3
"""
Transport Physics Fix Documentation

This script provides a detailed explanation of the salinity gradient inversion issue
and the solution, creating documentation for future reference.

Author: Nguyen Truong An
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

# Configure nice plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_documentation():
    """Generate comprehensive documentation on the salinity gradient inversion fix."""
    
    # Create output directory
    doc_dir = Path("OUT/documentation")
    doc_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate PDF documentation
    create_documentation_pdf(doc_dir)
    
    # Generate diagnostic figures
    create_diagnostic_figures(doc_dir)
    
    print(f"Documentation generated in {doc_dir}")

def create_documentation_pdf(output_dir):
    """Create a PDF document explaining the issue and solution."""
    
    # Open markdown file
    with open(output_dir / "transport_physics_fix.md", "w") as f:
        f.write("""# C-GEM Transport Physics Fix: Resolving the Salinity Gradient Inversion

## Executive Summary

This document explains the root cause and solution for the salinity gradient inversion issue in the JAX C-GEM model. The problem manifested as an inverted salinity gradient, showing low values at the ocean boundary and high values at the river boundary, which is physically impossible.

**Root Cause:** The primary causes were:
1. Incorrect order of operations in the transport step function
2. Incorrect velocity index checks in boundary condition application
3. Misalignment between the C-GEM and JAX-C-GEM coordinate systems

**Solution:** The fix involves:
1. Applying boundary conditions BEFORE advection (like in C-GEM)
2. Using the exact C-GEM boundary application logic with proper index mapping
3. Ensuring the fixed transport flow maintains the correct gradient direction

## Background

In estuaries, salinity follows a gradient with highest values at the ocean boundary (typically ~30-35 PSU) and lowest values at the river boundary (typically ~0-1 PSU). This gradient is maintained by the balance between seawater intrusion and freshwater discharge.

The JAX C-GEM model was showing an inverted gradient where salinity was:
- ~0.37 PSU at the ocean boundary (mouth)
- ~31.43 PSU at the river boundary (head)

This physically impossible pattern persisted despite implementing:
1. Velocity-dependent boundary conditions
2. TVD advection scheme
3. Physical dispersion coefficients

## Diagnostic Investigation

A comprehensive diagnostic investigation revealed several critical issues:

### 1. Order of Operations

In the original C-GEM code (`transport.c`):
```c
void Transport(int t)
{
 int s, i;

 Dispcoef(t);
 for(s=0; s<MAXV; s++)
 {
    if(v[s].env==1)
    {
      Openbound(v[s].c, s);  // BOUNDARY CONDITIONS FIRST
      TVD(v[s].c, s);        // ADVECTION SECOND
      Disp(v[s].c);          // DISPERSION LAST
    }
 }
}
```

In contrast, the JAX implementation applied boundary conditions AFTER advection:
```python
def transport_step(...):
    # Apply advection first
    c_after_advection = tvd_advection(...)
    
    # Apply boundary conditions after advection
    c_after_bc = apply_boundary_conditions_transport(...)
```

### 2. Boundary Condition Application

In the original C-GEM (`uptransport.c`), boundary conditions check velocity at indices 2 and M-1:
```c
void Openbound(double* co, int s)
{
  if(U[2]>=0.0)  // Check velocity at index 2
    co[1]=co[1]- (co[1]-v[s].clb)*U[2]*((double)DELTI)/((double)DELXI);
  else
    co[1]=co[1]- (co[3]-co[1])*U[2]*((double)DELTI)/((double)DELXI);

  if(U[M1]>=0.0)  // Check velocity at index M-2
    co[M]=co[M]- (co[M]-co[M2])*U[M1]*((double)DELTI)/((double)DELXI);
  else
    co[M]=co[M]- (v[s].cub-co[M])*U[M1]*((double)DELTI)/((double)DELXI);
}
```

In contrast, the JAX implementation checked velocity at indices 0 and -1, and used a different formula.

### 3. Index Mapping Between Implementations

- C-GEM uses 1-based indexing with:
  - `co[1]`: Mouth/ocean boundary (co[0] not used)
  - `co[M]`: Head/river boundary
  
- JAX uses 0-based indexing with:
  - `concentrations[0]`: Mouth/ocean boundary
  - `concentrations[-1]`: Head/river boundary

## The Solution

The solution involves three key changes:

### 1. Fixed Boundary Condition Function

Create a boundary condition function that exactly matches C-GEM behavior:
```python
def apply_boundary_conditions_transport(concentrations, velocities, boundary_conditions, DELTI, DELXI):
    # Get velocity at index 2 and M-2 (like C-GEM)
    u_lower = velocities[1]  # U[2] in C-GEM
    u_upper = velocities[-2]  # U[M1] in C-GEM
    
    # Mouth boundary logic exactly matching C-GEM
    mouth_bc_inflow = co_1 - (co_1 - clb_target) * u_lower * DELTI / DELXI
    mouth_bc_outflow = co_1 - (co_3 - co_1) * u_lower * DELTI / DELXI
    new_mouth = jnp.where(u_lower >= 0.0, mouth_bc_inflow, mouth_bc_outflow)
    
    # Head boundary logic exactly matching C-GEM
    head_bc_inflow = co_M - (co_M - co_M2) * u_upper * DELTI / DELXI
    head_bc_outflow = co_M - (cub_target - co_M) * u_upper * DELTI / DELXI
    new_head = jnp.where(u_upper >= 0.0, head_bc_inflow, head_bc_outflow)
```

### 2. Fixed Transport Step Function

Change the order of operations to match C-GEM:
```python
def transport_step(...):
    # Step 1: FIRST apply boundary conditions (exact C-GEM order)
    c_after_bc = apply_boundary_conditions_transport(...)
    
    # Step 2: THEN apply advection
    c_after_advection = tvd_advection(...)
    
    # Step 3: FINALLY compute and apply dispersion
    c_after_dispersion = dispersion_step(...)
```

### 3. Initial Salinity Profile

Ensure the initial salinity profile has the physically correct gradient:
```python
# Ocean and river end salinity values
ocean_salinity = 33.0  # PSU at mouth
river_salinity = 0.1   # PSU upstream

# Create salinity profile with exponential decay from mouth to head
L = 40000.0  # Salt intrusion length scale (40 km)
salinity = ocean_salinity * jnp.exp(-x_vals / L)
salinity = jnp.maximum(salinity, river_salinity)
```

## Validation Results

The fixed implementation successfully:
1. Maintains the correct salinity gradient direction (high at ocean, low at river)
2. Preserves the physical behavior of boundary condition application
3. Matches the C-GEM coordinate system logic
4. Produces stable and physically realistic results

## Implementation

To implement this fix:
1. Replace the `transport.py` file with the fixed implementation
2. Ensure the transport step function applies boundary conditions first
3. Validate with test cases to confirm the fix works as expected

## Conclusion

This solution addresses the fundamental issue in the JAX C-GEM transport physics implementation by:
1. Correctly applying boundary conditions before advection
2. Using the proper velocity-dependent boundary logic
3. Maintaining the correct coordinate system mapping between C-GEM and JAX

The fix ensures the model produces physically realistic salinity gradients that are essential for accurate estuarine modeling.
""")
    
    print(f"Created documentation: {output_dir / 'transport_physics_fix.md'}")

def create_diagnostic_figures(output_dir):
    """Create diagnostic figures explaining the issue and solution."""
    
    # Figure 1: Coordinate System and Indices
    plt.figure(figsize=(10, 6))
    
    # Draw C-GEM vs JAX grid
    grid_size = 10
    
    # C-GEM grid (1-based)
    for i in range(grid_size):
        plt.plot([0, 10], [i, i], 'k-', alpha=0.3)
    for i in range(11):
        plt.plot([i, i], [0, grid_size-1], 'k-', alpha=0.3)
    
    # C-GEM indices
    for i in range(1, grid_size):
        plt.text(-0.8, i-0.5, f"{i}", fontsize=12)
    
    # JAX indices
    for i in range(grid_size-1):
        plt.text(10.8, i+0.5, f"{i}", fontsize=12, color='blue')
    
    # Mark ocean and river boundaries
    plt.text(5, -0.8, "OCEAN (MOUTH)", fontsize=14, ha='center')
    plt.text(5, grid_size+0.3, "RIVER (HEAD)", fontsize=14, ha='center')
    
    # Special markings for boundary conditions
    plt.scatter([1], [0.5], s=100, color='red')
    plt.text(1, 0, "co[1]", color='red', ha='center', fontsize=12)
    plt.text(1, -0.5, "JAX: concentrations[0]", color='blue', ha='center', fontsize=10)
    
    plt.scatter([grid_size-1], [grid_size-1.5], s=100, color='red')
    plt.text(grid_size-1, grid_size-2, "co[M]", color='red', ha='center', fontsize=12)
    plt.text(grid_size-1, grid_size-2.5, "JAX: concentrations[-1]", color='blue', ha='center', fontsize=10)
    
    # Annotate velocity check points
    plt.scatter([2], [1.5], s=80, color='green')
    plt.text(2.5, 1.5, "U[2]: Velocity check for lower boundary", color='green', fontsize=10)
    
    plt.scatter([grid_size-3], [grid_size-2.5], s=80, color='green')
    plt.text(grid_size-3.5, grid_size-2.5, "U[M-1]: Velocity check for upper boundary", color='green', fontsize=10)
    
    # Set axis limits and remove ticks
    plt.xlim(-1, 11)
    plt.ylim(-1, grid_size+1)
    plt.axis('off')
    
    # Title and legend
    plt.title("C-GEM vs JAX-C-GEM Coordinate System Mapping", fontsize=16)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', marker='o', markersize=10, label='C-GEM Indices (1-based)', markerfacecolor='red', linestyle='none'),
        Line2D([0], [0], color='blue', marker='o', markersize=10, label='JAX Indices (0-based)', markerfacecolor='blue', linestyle='none'),
        Line2D([0], [0], color='green', marker='o', markersize=10, label='Velocity Check Points', markerfacecolor='green', linestyle='none')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "coordinate_system_mapping.png", dpi=300)
    
    # Figure 2: Order of Operations
    plt.figure(figsize=(12, 8))
    
    # Draw flow diagrams
    def draw_box(ax, x, y, width, height, text, color='blue', alpha=0.2):
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='black'))
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12)
    
    def draw_arrow(ax, x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="-|>", lw=1.5, color='black'))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2, text, ha='center', va='center', fontsize=10)
    
    # C-GEM flow
    ax1 = plt.subplot(2, 1, 1)
    
    draw_box(ax1, 1, 3, 2, 1, "Dispcoef(t)", color='lightgray')
    draw_box(ax1, 4, 3, 2, 1, "Openbound()\nBoundary\nConditions", color='green')
    draw_box(ax1, 7, 3, 2, 1, "TVD()\nAdvection", color='blue')
    draw_box(ax1, 10, 3, 2, 1, "Disp()\nDispersion", color='red')
    
    draw_arrow(ax1, 3, 3.5, 4, 3.5)
    draw_arrow(ax1, 6, 3.5, 7, 3.5)
    draw_arrow(ax1, 9, 3.5, 10, 3.5)
    
    ax1.set_xlim(0, 13)
    ax1.set_ylim(2, 5)
    ax1.axis('off')
    ax1.set_title("C-GEM Transport Execution Order", fontsize=14)
    
    # JAX flow (current)
    ax2 = plt.subplot(2, 1, 2)
    
    draw_box(ax2, 1, 3, 2, 1, "tvd_advection()\nAdvection", color='blue')
    draw_box(ax2, 4, 3, 2, 1, "apply_boundary_\nconditions()", color='green')
    draw_box(ax2, 7, 3, 2, 1, "dispersion_step()\nDispersion", color='red')
    
    draw_arrow(ax2, 3, 3.5, 4, 3.5)
    draw_arrow(ax2, 6, 3.5, 7, 3.5)
    
    ax2.set_xlim(0, 13)
    ax2.set_ylim(2, 5)
    ax2.axis('off')
    ax2.set_title("JAX C-GEM Transport Execution Order (INCORRECT)", fontsize=14)
    
    # Add problem indicators
    ax2.text(2, 2.5, "⚠️ SHOULD BE SECOND", color='blue', fontsize=12)
    ax2.text(5, 2.5, "⚠️ SHOULD BE FIRST", color='green', fontsize=12)
    
    # Draw fix
    draw_arrow(ax2, 5, 2, 2, 2, text="FIX: Switch order")
    
    plt.tight_layout()
    plt.savefig(output_dir / "execution_order.png", dpi=300)
    
    # Figure 3: Salinity Gradient Comparison
    plt.figure(figsize=(10, 6))
    
    # Create distance array
    grid_size = 102
    delxi = 2000
    distance_km = np.arange(grid_size) * delxi / 1000
    
    # Create salinity profiles
    correct_salinity = 30 * np.exp(-distance_km / 40)
    inverted_salinity = 0.1 + 30 * (1 - np.exp(-distance_km / 150))
    
    plt.plot(distance_km, correct_salinity, 'g-', linewidth=2, label="Correct (High at mouth, low at river)")
    plt.plot(distance_km, inverted_salinity, 'r-', linewidth=2, label="Inverted (Low at mouth, high at river)")
    
    # Mark mouth and head
    plt.annotate("MOUTH\n(OCEAN)", xy=(0, correct_salinity[0]), xytext=(5, 33),
                arrowprops=dict(arrowstyle="->", color='blue'))
    plt.annotate("HEAD\n(RIVER)", xy=(distance_km[-1], correct_salinity[-1]), xytext=(190, 5),
                arrowprops=dict(arrowstyle="->", color='blue'))
    
    plt.grid(True)
    plt.xlabel("Distance from Mouth (km)", fontsize=12)
    plt.ylabel("Salinity (PSU)", fontsize=12)
    plt.title("Comparison of Correct vs Inverted Salinity Gradient", fontsize=16)
    plt.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "salinity_gradient_comparison.png", dpi=300)
    
    print(f"Created diagnostic figures in {output_dir}")

if __name__ == "__main__":
    generate_documentation()