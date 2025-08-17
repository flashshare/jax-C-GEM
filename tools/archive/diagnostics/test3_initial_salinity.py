#!/usr/bin/env python3
"""
Test 3: Initial Salinity Profile Test

This test examines how salinity is initialized in both JAX C-GEM and original
C-GEM to check for incorrect initialization causing the gradient inversion.

Potential Issue: Initial salinity might be set with an incorrect gradient direction.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_parser import parse_model_config
from core.data_loader import DataLoader
import jax.numpy as jnp

def test_initial_salinity():
    """Test initial salinity profile setup."""
    
    print("=== TEST 3: INITIAL SALINITY PROFILE ===\n")
    
    # 1. Load model config
    print("1. Loading model configuration...")
    model_config = parse_model_config('config/model_config.txt')
    
    M = model_config['M']
    DELXI = model_config['DELXI']
    
    # Calculate distance array
    x_vals = np.arange(M) * DELXI  # Distance from mouth (m)
    distance_km = x_vals / 1000.0
    
    # 2. Examine JAX C-GEM initialization
    print("\n2. Examining JAX C-GEM salinity initialization...")
    
    # We'll analyze the simulation_engine.py and transport.py files to find initialization
    print("Searching for salinity initialization in transport.py...")
    found_initialization = False
    
    try:
        with open('src/core/transport.py', 'r') as f:
            transport_code = f.read()
            
        # Look for initialization patterns
        init_patterns = [
            r'salinity\s*=\s*jnp\.linspace',
            r'concentrations\[\d+\].*=\s*jnp\.linspace',
            r'concentrations.*Sal.*=\s*'
        ]
        
        for pattern in init_patterns:
            matches = re.findall(pattern, transport_code)
            if matches:
                print(f"Found potential initialization: {matches}")
                found_initialization = True
                
        if not found_initialization:
            print("No direct salinity initialization found in transport.py")
    except Exception as e:
        print(f"Error examining transport.py: {e}")
        
    # Check simulation_engine.py
    print("\nSearching for salinity initialization in simulation_engine.py...")
    found_initialization = False
    
    try:
        with open('src/core/simulation_engine.py', 'r') as f:
            sim_code = f.read()
            
        # Look for initialization patterns
        init_patterns = [
            r'salinity\s*=\s*jnp\.linspace',
            r'concentrations\[\d+\].*=\s*jnp\.linspace',
            r'concentrations.*Sal.*=\s*'
        ]
        
        for pattern in init_patterns:
            matches = re.findall(pattern, sim_code)
            if matches:
                print(f"Found potential initialization: {matches}")
                found_initialization = True
                
        if not found_initialization:
            print("No direct salinity initialization found in simulation_engine.py")
    except Exception as e:
        print(f"Error examining simulation_engine.py: {e}")
    
    # 3. Examine C-GEM initialization
    print("\n3. Examining original C-GEM salinity initialization...")
    
    try:
        with open('deprecated/original-C-GEM/init.c', 'r') as f:
            init_c = f.read()
            
        # Look for salinity initialization in init.c
        sal_patterns = [
            r'v\[Sal\].c\[.*\]',
            r'v\[9\].c\[.*\]'
        ]
        
        found_c_init = False
        for pattern in sal_patterns:
            matches = re.findall(pattern, init_c)
            if matches:
                print(f"Found C salinity initialization: {matches[:5]}...")
                found_c_init = True
                
        if not found_c_init:
            print("No explicit salinity initialization found in init.c")
            
        # Check for a general initialization pattern that might include salinity
        general_init = re.search(r'for\s*\(\s*i\s*=\s*0\s*;\s*i\s*<\s*MAXV\s*;.*\)\s*{\s*.*for\s*\(\s*j\s*=\s*0\s*;\s*j\s*<=\s*M\s*;', init_c, re.DOTALL)
        if general_init:
            print("Found general species initialization loop in init.c")
            print(f"Pattern: {general_init.group(0)[:100]}...")
        else:
            print("No general species initialization loop found in init.c")
    except Exception as e:
        print(f"Error examining init.c: {e}")
    
    # 4. Create simple test initialization for comparison
    print("\n4. Creating test initializations for comparison...")
    
    # Case 1: Ocean to river gradient (correct - high at mouth, low at head)
    correct_gradient = np.linspace(30.0, 0.1, M)
    
    # Case 2: River to ocean gradient (inverted - low at mouth, high at head)
    inverted_gradient = np.linspace(0.1, 30.0, M)
    
    # Case 3: Exponential decay from ocean (more realistic)
    # S = S0 * exp(-x/b) where b is a decay constant
    b = 50000  # Decay length scale in meters
    x_array = np.arange(M) * DELXI
    exp_gradient = 30.0 * np.exp(-x_array / b)
    
    print(f"Correct gradient: mouth={correct_gradient[0]:.1f}, head={correct_gradient[-1]:.1f}")
    print(f"Inverted gradient: mouth={inverted_gradient[0]:.1f}, head={inverted_gradient[-1]:.1f}")
    print(f"Exponential gradient: mouth={exp_gradient[0]:.1f}, head={exp_gradient[-1]:.1f}")
    
    # 5. Create visualization
    print("\n5. Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(distance_km, correct_gradient, 'b-', linewidth=2, label='Correct (Ocean→River)')
    plt.plot(distance_km, inverted_gradient, 'r-', linewidth=2, label='Inverted (River→Ocean)')
    plt.plot(distance_km, exp_gradient, 'g-', linewidth=2, label='Exponential Decay (Realistic)')
    
    plt.title('Salinity Gradient Comparison')
    plt.xlabel('Distance from Mouth (km)')
    plt.ylabel('Salinity (PSU)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(distance_km, correct_gradient, 'b-', linewidth=2, label='Correct (Ocean→River)')
    plt.semilogy(distance_km, inverted_gradient, 'r-', linewidth=2, label='Inverted (River→Ocean)')
    plt.semilogy(distance_km, exp_gradient, 'g-', linewidth=2, label='Exponential Decay (Realistic)')
    
    plt.title('Salinity Gradient (Log Scale)')
    plt.xlabel('Distance from Mouth (km)')
    plt.ylabel('Salinity (PSU) - Log Scale')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    os.makedirs('OUT/diagnostics', exist_ok=True)
    plt.tight_layout()
    plt.savefig('OUT/diagnostics/test3_initial_salinity.png', dpi=300)
    print("Saved visualization to OUT/diagnostics/test3_initial_salinity.png")
    
    # 6. Check if initialization functions are available
    print("\n6. Searching for initialization functions...")
    try:
        from core.simulation_engine import create_initial_transport_state
        print("Found create_initial_transport_state function")
        
        # Try to call it with simple parameters to see what it does
        try:
            # Create minimal boundary conditions
            boundary_conditions = {
                'LB_Sal': 30.0,  # Ocean value
                'UB_Sal': 0.1    # River value
            }
            
            # Call the function
            initial_state = create_initial_transport_state(M, boundary_conditions)
            
            # Check the salinity gradient
            sal_index = 9  # From model_config.py Species enum
            salinity = initial_state[sal_index]
            
            print(f"\nInitial state from create_initial_transport_state:")
            print(f"  Shape: {salinity.shape}")
            print(f"  Mouth (index 0): {salinity[0]:.3f} PSU")
            print(f"  Head (index {M-1}): {salinity[M-1]:.3f} PSU")
            
            if salinity[0] > salinity[M-1]:
                print("  ✅ Gradient is correct (high at mouth, low at head)")
            else:
                print("  ❌ Gradient is inverted (low at mouth, high at head)")
                
            # Add to plot for comparison
            plt.figure(figsize=(12, 6))
            plt.plot(distance_km, salinity, 'b-', linewidth=2, label='Actual Initialization')
            plt.plot(distance_km, correct_gradient, 'g--', linewidth=1, label='Expected Correct')
            plt.plot(distance_km, inverted_gradient, 'r--', linewidth=1, label='Inverted')
            
            plt.title('Actual vs Expected Salinity Initialization')
            plt.xlabel('Distance from Mouth (km)')
            plt.ylabel('Salinity (PSU)')
            plt.legend()
            plt.grid(True)
            
            plt.savefig('OUT/diagnostics/test3_actual_initialization.png', dpi=300)
            print("Saved actual initialization to OUT/diagnostics/test3_actual_initialization.png")
        
        except Exception as e:
            print(f"Error calling create_initial_transport_state: {e}")
            
    except ImportError:
        print("create_initial_transport_state function not found")
    
    return {
        'grid_size': M,
        'distance_km': distance_km.tolist(),
        'correct_gradient': correct_gradient.tolist(),
        'inverted_gradient': inverted_gradient.tolist(),
        'exponential_gradient': exp_gradient.tolist(),
        'actual_initialization': salinity.tolist() if 'salinity' in locals() else None
    }

if __name__ == "__main__":
    test_initial_salinity()
    print("\nTest 3 complete.")