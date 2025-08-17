#!/usr/bin/env python3
"""
Test 5: Execution Flow Analysis

This test examines how data flows through the transport module and compares
with the C-GEM implementation to find where the inversion might occur.

Potential Issue: The order of operations or flow of data might be different
between the two implementations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def analyze_execution_flow():
    """Analyze the flow of execution in transport calculations."""
    
    print("=== TEST 5: EXECUTION FLOW ANALYSIS ===\n")
    
    # 1. Analyze C-GEM flow
    print("1. Analyzing original C-GEM execution flow...")
    c_flow = analyze_c_gem_flow()
    
    # 2. Analyze JAX C-GEM flow
    print("\n2. Analyzing JAX C-GEM execution flow...")
    jax_flow = analyze_jax_flow()
    
    # 3. Compare flows
    print("\n3. Comparing execution flows...")
    compare_flows(c_flow, jax_flow)
    
    # 4. Analyze boundary application
    print("\n4. Analyzing boundary condition application order...")
    analyze_boundary_application()
    
    return {
        'c_gem_flow': c_flow,
        'jax_flow': jax_flow
    }

def analyze_c_gem_flow():
    """Analyze the execution flow in original C-GEM."""
    
    flow = []
    
    # Parse main.c
    try:
        with open('deprecated/original-C-GEM/main.c', 'r') as f:
            main_c = f.read()
        
        # Extract main time loop
        time_loop_match = re.search(r'for\s*\(t=0;\s*t<=MAXT;\s*t\+=DELTI\)\s*{(.*?)}', main_c, re.DOTALL)
        if time_loop_match:
            time_loop = time_loop_match.group(1)
            steps = re.findall(r'(\w+)\s*\(.*?\)', time_loop)
            
            print(f"Found main time loop with steps: {steps}")
            flow.append(f"Main loop steps: {', '.join(steps)}")
        else:
            print("Could not find main time loop in main.c")
    except Exception as e:
        print(f"Error parsing main.c: {e}")
    
    # Parse transport.c
    try:
        with open('deprecated/original-C-GEM/transport.c', 'r') as f:
            transport_c = f.read()
        
        # Extract Transport function
        transport_match = re.search(r'void\s+Transport\s*\(.*?\)\s*{(.*?)}', transport_c, re.DOTALL)
        if transport_match:
            transport_func = transport_match.group(1)
            
            # Find function calls in transport
            transport_steps = re.findall(r'(\w+)\s*\(.*?\)', transport_func)
            
            print(f"Found Transport function with steps: {transport_steps}")
            flow.append(f"Transport function steps: {', '.join(transport_steps)}")
        else:
            print("Could not find Transport function in transport.c")
    except Exception as e:
        print(f"Error parsing transport.c: {e}")
    
    # Parse uptransport.c
    try:
        with open('deprecated/original-C-GEM/uptransport.c', 'r') as f:
            uptransport_c = f.read()
        
        # Extract Openbound function
        openbound_match = re.search(r'void\s+Openbound\s*\(.*?\)\s*{(.*?)}', uptransport_c, re.DOTALL)
        if openbound_match:
            openbound_func = openbound_match.group(1)
            
            # Look for key logic
            if 'if(U[2]>=0.0)' in openbound_func:
                flow.append("Openbound checks velocity at index 2")
                print("Openbound checks velocity at index 2")
                
                # Extract the if-else block
                if_else_block = re.search(r'if\s*\(\s*U\s*\[\s*2\s*\]\s*>=\s*0.0\s*\)(.*?)else(.*?)', openbound_func, re.DOTALL)
                if if_else_block:
                    if_block = if_else_block.group(1)
                    else_block = if_else_block.group(2)
                    
                    flow.append(f"Openbound if velocity >= 0: {if_block.strip()}")
                    flow.append(f"Openbound else: {else_block.strip()}")
                    
                    print(f"If block (velocity >= 0): {if_block.strip()}")
                    print(f"Else block: {else_block.strip()}")
            else:
                print("Could not find velocity check in Openbound")
        else:
            print("Could not find Openbound function in uptransport.c")
            
        # Extract TVD function
        tvd_match = re.search(r'void\s+TVD\s*\(.*?\)\s*{(.*?)}', uptransport_c, re.DOTALL)
        if tvd_match:
            flow.append("Found TVD function")
            print("Found TVD function in uptransport.c")
        else:
            print("Could not find TVD function in uptransport.c")
    except Exception as e:
        print(f"Error parsing uptransport.c: {e}")
    
    return flow

def analyze_jax_flow():
    """Analyze the execution flow in JAX C-GEM."""
    
    flow = []
    
    # Parse main.py
    try:
        with open('src/main.py', 'r') as f:
            main_py = f.read()
        
        # Look for main simulation loop
        sim_loop_match = re.search(r'def\s+run_simulation\s*\(.*?\).*?:.*?for\s+.*?in\s+.*?:', main_py, re.DOTALL)
        if sim_loop_match:
            flow.append("Found simulation loop in main.py")
            print("Found simulation loop in main.py")
        else:
            print("Could not find simulation loop in main.py")
    except Exception as e:
        print(f"Error parsing main.py: {e}")
    
    # Parse simulation_engine.py
    try:
        with open('src/core/simulation_engine.py', 'r') as f:
            sim_engine = f.read()
        
        # Look for jitted step function
        jitted_step_match = re.search(r'@jax\.jit.*?def\s+(\w+)\s*\(.*?\).*?:', sim_engine, re.DOTALL)
        if jitted_step_match:
            step_func_name = jitted_step_match.group(1)
            flow.append(f"Found jitted step function: {step_func_name}")
            print(f"Found jitted step function: {step_func_name}")
        else:
            print("Could not find jitted step function in simulation_engine.py")
    except Exception as e:
        print(f"Error parsing simulation_engine.py: {e}")
    
    # Parse transport.py
    try:
        with open('src/core/transport.py', 'r') as f:
            transport_py = f.read()
        
        # Look for transport_step function
        transport_step_match = re.search(r'def\s+transport_step\s*\(.*?\).*?:', transport_py, re.DOTALL)
        if transport_step_match:
            flow.append("Found transport_step function")
            print("Found transport_step function")
            
            # Look for function calls within transport_step
            transport_step = transport_step_match.group(0)
            function_calls = re.findall(r'(\w+)\s*\(', transport_step)
            if function_calls:
                flow.append(f"Transport step calls: {', '.join(function_calls)}")
                print(f"Transport step calls: {', '.join(function_calls)}")
        else:
            print("Could not find transport_step function")
        
        # Look for boundary condition application
        bc_match = re.search(r'def\s+apply_boundary_conditions_transport\s*\(.*?\).*?:', transport_py, re.DOTALL)
        if bc_match:
            bc_func = bc_match.group(0)
            
            # Look for velocity condition
            if 'u[0]' in bc_func or 'u[1]' in bc_func or 'u[2]' in bc_func:
                flow.append("Boundary condition checks velocity")
                print("Boundary condition checks velocity")
                
                # Find the specific index checked
                velocity_checks = re.findall(r'u\[(\d+)\]', bc_func)
                if velocity_checks:
                    flow.append(f"Checks velocity at indices: {', '.join(velocity_checks)}")
                    print(f"Checks velocity at indices: {', '.join(velocity_checks)}")
            else:
                print("Could not find velocity check in boundary condition function")
        else:
            print("Could not find boundary condition application function")
    except Exception as e:
        print(f"Error parsing transport.py: {e}")
    
    return flow

def compare_flows(c_flow, jax_flow):
    """Compare the execution flows between implementations."""
    
    print("\nExecution Flow Comparison:")
    print("-------------------------")
    
    print("\nOriginal C-GEM Flow:")
    for step in c_flow:
        print(f"  - {step}")
        
    print("\nJAX C-GEM Flow:")
    for step in jax_flow:
        print(f"  - {step}")
    
    # Check for key differences
    if any("Openbound checks velocity at index 2" in step for step in c_flow):
        print("\n⚠️ C-GEM checks velocity at index 2 for boundary conditions")
    
    # Check velocity indices in JAX
    jax_velocity_indices = []
    for step in jax_flow:
        if "Checks velocity at indices" in step:
            indices_match = re.search(r'indices:\s*(.*)', step)
            if indices_match:
                indices = indices_match.group(1).split(', ')
                jax_velocity_indices = [int(idx) for idx in indices]
                
    if jax_velocity_indices:
        print(f"\n⚠️ JAX checks velocity at indices: {jax_velocity_indices}")
        
        if 2 in jax_velocity_indices and 0 not in jax_velocity_indices:
            print("✅ Both implementations check velocity at index 2")
        elif 0 in jax_velocity_indices and 2 not in jax_velocity_indices:
            print("❌ Index mismatch: C-GEM uses index 2, JAX uses index 0")
        else:
            print("⚠️ Different velocity index checks between implementations")

def analyze_boundary_application():
    """Analyze boundary condition application order."""
    
    # Check C-GEM boundary application
    try:
        with open('deprecated/original-C-GEM/transport.c', 'r') as f:
            transport_c = f.read()
        
        transport_match = re.search(r'void\s+Transport\s*\(.*?\)\s*{(.*?)}', transport_c, re.DOTALL)
        if transport_match:
            transport_func = transport_match.group(1)
            
            # Find order of operations
            calls = re.findall(r'(\w+)\s*\((.*?)\)', transport_func)
            
            if calls:
                print("C-GEM Transport function call order:")
                for i, (func, args) in enumerate(calls):
                    print(f"  {i+1}. {func}({args})")
                    
                # Specifically check for Openbound/TVD order
                func_names = [func for func, _ in calls]
                if 'Openbound' in func_names and 'TVD' in func_names:
                    ob_index = func_names.index('Openbound')
                    tvd_index = func_names.index('TVD')
                    
                    if ob_index < tvd_index:
                        print("✅ C-GEM: Openbound is called BEFORE TVD")
                    else:
                        print("⚠️ C-GEM: Openbound is called AFTER TVD")
    except Exception as e:
        print(f"Error analyzing C-GEM boundary application: {e}")
    
    # Check JAX C-GEM boundary application
    try:
        with open('src/core/transport.py', 'r') as f:
            transport_py = f.read()
        
        # Find transport_step function
        transport_step_match = re.search(r'def\s+transport_step\s*\(.*?\).*?:(.*?)return', transport_py, re.DOTALL)
        if transport_step_match:
            transport_step = transport_step_match.group(1)
            
            # Find function calls
            calls = re.findall(r'(\w+)\s*\(', transport_step)
            
            if calls:
                print("\nJAX Transport step function call order:")
                for i, func in enumerate(calls):
                    print(f"  {i+1}. {func}")
                    
                # Check for boundary conditions vs. advection order
                bc_funcs = ['apply_boundary_conditions_transport']
                advection_funcs = ['tvd_advection', 'centered_difference_advection']
                
                bc_index = -1
                adv_index = -1
                
                for bc_func in bc_funcs:
                    if bc_func in calls:
                        bc_index = calls.index(bc_func)
                
                for adv_func in advection_funcs:
                    if adv_func in calls:
                        adv_index = calls.index(adv_func)
                
                if bc_index >= 0 and adv_index >= 0:
                    if bc_index < adv_index:
                        print("✅ JAX: Boundary conditions are applied BEFORE advection")
                    else:
                        print("⚠️ JAX: Boundary conditions are applied AFTER advection")
    except Exception as e:
        print(f"Error analyzing JAX boundary application: {e}")
    
    # Create visualization
    print("\n5. Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Visual representation of the execution flow
    c_flow_steps = [
        "Initial Conditions", 
        "Hydro Solve", 
        "Boundary Conditions", 
        "TVD Advection", 
        "Dispersion",
        "Biogeochemistry"
    ]
    
    jax_flow_steps = [
        "Initial Conditions", 
        "Hydro Solve", 
        "Transport Step", 
        "Biogeochemistry"
    ]
    
    # Plot C-GEM flow
    for i, step in enumerate(c_flow_steps):
        color = 'blue'
        if step == "Boundary Conditions":
            color = 'green'
        elif step == "TVD Advection":
            color = 'orange'
            
        ax.barh(0, 0.8, left=i, height=0.8, color=color, alpha=0.7, 
               label=f"C-GEM: {step}" if i == 0 else "")
        ax.text(i + 0.4, 0, step, ha='center', va='center', rotation=90, fontsize=10)
    
    # Plot JAX flow
    for i, step in enumerate(jax_flow_steps):
        color = 'red'
        if step == "Transport Step":
            color = 'purple'
            
        ax.barh(1, 0.8, left=i, height=0.8, color=color, alpha=0.7,
               label=f"JAX: {step}" if i == 0 else "")
        ax.text(i + 0.4, 1, step, ha='center', va='center', rotation=90, fontsize=10)
    
    # Expand the transport step
    transport_substeps = [
        "Apply BCs",
        "Advection",
        "Dispersion",
        "Return State"
    ]
    
    for i, step in enumerate(transport_substeps):
        ax.barh(2, 0.15, left=2 + i*0.2, height=0.4, color='purple', alpha=0.9)
        ax.text(2 + i*0.2 + 0.075, 2, step, ha='center', va='center', fontsize=8)
    
    # Add markers for potential issue areas
    ax.plot([2, 2], [1.5, 2], 'k--')
    ax.plot([2+0.8], [2], 'ro', markersize=10)
    ax.annotate('Potential\nIssue Area', xy=(2+0.4, 2), xytext=(2+0.4, 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', fontsize=10)
    
    # Customize plot
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['C-GEM Flow', 'JAX Flow', 'JAX Transport Detail'])
    ax.set_xlabel('Execution Order')
    ax.set_title('Execution Flow Comparison')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs('OUT/diagnostics', exist_ok=True)
    plt.savefig('OUT/diagnostics/test5_execution_flow.png', dpi=300)
    print("Saved visualization to OUT/diagnostics/test5_execution_flow.png")

if __name__ == "__main__":
    analyze_execution_flow()
    print("\nTest 5 complete.")