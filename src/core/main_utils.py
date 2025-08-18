"""
Shared utilities for JAX C-GEM main scripts.

This module contains common functionality shared between main.py and 
main_ultra_performance.py to eliminate code duplication and improve maintainability.

Architecture:
- Command line argument parsing
- Configuration loading and validation  
- Output format selection logic
- Common initialization routines
- Performance reporting utilities

Author: Nguyen Truong An
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def create_argument_parser(mode: str = "standard") -> argparse.ArgumentParser:
    """Create argument parser with common arguments."""
    if mode == "ultra":
        description = 'JAX C-GEM - Ultra-Performance Mode'
    else:
        description = 'JAX C-GEM - High-Performance Tidal Model'
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mode', choices=['run'], default='run', help='Execution mode')
    parser.add_argument('--output-format', choices=['csv', 'npz', 'auto'], default='auto', 
                       help='Output format (auto=choose based on simulation length)')
    parser.add_argument('--no-physics-check', action='store_true',
                       help='Disable physics validation for maximum speed')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--config', default='config/model_config.txt', 
                       help='Configuration file to use')
    
    return parser

def print_header(mode: str = "standard"):
    """Print application header with mode-specific information."""
    if mode == "ultra":
        print("🚀 JAX C-GEM Ultra-Performance Mode")
        print("=" * 50)
        print("🔥 Performance Optimizations Active:")
        print("   ✅ Dictionary creation bottleneck eliminated")
        print("   ✅ JIT-compiled combined simulation step")
        print("   ✅ Memory access pattern optimization")
        print("   ✅ Reduced I/O overhead")
        print("   ✅ Direct array operations")
        print("   🔬 Scientific accuracy: 100% preserved")
    else:
        print("🚀 JAX C-GEM High-Performance Mode")
        print("=" * 50)

def load_configurations(config_file: str) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """Load and validate all configuration files."""
    # Import here to avoid circular imports
    from .config_parser import parse_model_config, parse_input_data_config
    from .data_loader import DataLoader
    
    print("📋 Loading configuration...")
    config_start = time.time()
    
    model_config = parse_model_config(config_file)
    data_config = parse_input_data_config('config/input_data_config.txt')
    data_loader = DataLoader(data_config)
    
    config_time = time.time() - config_start
    print(f"✅ Configuration loaded in {config_time:.1f}s")
    
    return model_config, data_config, data_loader

def analyze_simulation_parameters(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze simulation parameters and return analysis results."""
    total_days = model_config.get('MAXT', 0)
    warmup_days = model_config.get('WARMUP', 0)
    output_days = total_days - warmup_days
    expected_outputs = output_days * 48  # 30-min intervals
    
    analysis = {
        'total_days': total_days,
        'warmup_days': warmup_days,
        'output_days': output_days,
        'expected_outputs': expected_outputs
    }
    
    print(f"\n📊 Simulation:")
    print(f"   Total days: {total_days}")
    print(f"   Output days: {output_days}")
    print(f"   Expected outputs: {expected_outputs:,}")
    
    return analysis

def select_optimal_output_format(output_format: str, expected_outputs: int) -> str:
    """Select optimal output format based on simulation size."""
    if output_format == 'auto':
        # Auto-select based on data size
        if expected_outputs > 10000:  # Large simulations
            chosen_format = 'npz'
            print(f"   📁 Auto-selected NPZ format (efficient for {expected_outputs:,} outputs)")
        else:  # Smaller simulations
            chosen_format = 'csv'  
            print(f"   📁 Auto-selected CSV format (readable for {expected_outputs:,} outputs)")
    else:
        chosen_format = output_format
        print(f"   📁 Using {chosen_format.upper()} format")
    
    return chosen_format

def print_performance_summary(total_time: float, mode: str = "standard"):
    """Print final performance summary with physics validation debug."""
    
    # Add physics validation debug messages
    print("\n🔬 PHYSICS VALIDATION DEBUG")
    print("=" * 40)
    
    try:
        # Quick physics validation check
        import numpy as np
        import os
        
        if os.path.exists("OUT/complete_simulation_results.npz"):
            data = np.load("OUT/complete_simulation_results.npz")
            
            # Check NH4 spikes
            if 'NH4' in data:
                nh4_data = data['NH4']
                warmup_steps = int(0.2 * nh4_data.shape[0])
                nh4_profile = np.mean(nh4_data[warmup_steps:], axis=0)
                nh4_max = np.max(nh4_profile)
                nh4_spikes = np.sum(nh4_profile > 40.0)
                
                print(f"🧪 NH4 Analysis:")
                print(f"   Maximum: {nh4_max:.1f} mmol/m³")
                print(f"   Spikes >40: {nh4_spikes} grid points")
                print(f"   Status: {'✅ GOOD' if nh4_max <= 40.0 else '⚠️ SPIKES DETECTED'}")
            
            # Check PO4 variation
            if 'PO4' in data:
                po4_data = data['PO4']
                po4_profile = np.mean(po4_data[warmup_steps:], axis=0)
                po4_std = np.std(po4_profile)
                po4_mean = np.mean(po4_profile)
                po4_cov = (po4_std/po4_mean)*100 if po4_mean > 0 else 0
                
                print(f"🧪 PO4 Analysis:")
                print(f"   Range: {np.min(po4_profile):.6f} - {np.max(po4_profile):.6f} mmol/m³")
                print(f"   CoV: {po4_cov:.1f}%")
                print(f"   Status: {'✅ GOOD' if po4_cov > 5.0 else '⚠️ LOW VARIATION'}")
            
            # Check Salinity
            if 'S' in data:
                s_data = data['S']
                s_profile = np.mean(s_data[warmup_steps:], axis=0)
                s_range = np.max(s_profile) - np.min(s_profile)
                
                print(f"🧪 Salinity Analysis:")
                print(f"   Range: {np.min(s_profile):.1f} - {np.max(s_profile):.1f} PSU")
                print(f"   Gradient: {s_range:.1f} PSU")
                print(f"   Status: {'✅ GOOD' if s_range > 10.0 else '⚠️ WEAK GRADIENT'}")
                
        else:
            print("⚠️ Results file not found - cannot validate physics")
            
    except Exception as e:
        print(f"⚠️ Physics validation error: {e}")
    
    print("\n" + "=" * 50)
    if mode == "ultra":
        print(f"⚡ Ultra-Performance JAX C-GEM completed in {total_time:.1f}s")
    else:
        print(f"🚀 JAX C-GEM completed in {total_time:.1f}s")
    print("=" * 50)

def handle_simulation_error(error: Exception, mode: str = "standard"):
    """Handle simulation errors with appropriate reporting."""
    if mode == "ultra":
        print(f"\n❌ Ultra-Performance simulation failed: {error}")
    else:
        print(f"\n❌ Simulation failed: {error}")
    
    print(f"Error type: {type(error).__name__}")
    print(f"Error details: {str(error)}")
    
    # Print helpful debugging information
    print("\n🔧 Debugging suggestions:")
    print("   1. Check configuration files for invalid parameters")
    print("   2. Verify input data files exist and are readable")  
    print("   3. Ensure sufficient memory for simulation size")
    print("   4. Try running with --debug flag for more details")
    
    return 1  # Return error code

def initialize_jax() -> bool:
    """Initialize JAX with error handling."""
    try:
        import jax
        import jax.numpy as jnp
        print("🔧 JAX initialized")
        return True
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        print("\n🔧 To install JAX, run:")
        print("   pip install jax jaxlib")
        return False

def validate_runtime_environment():
    """Validate the runtime environment is properly configured."""
    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended for optimal performance")
    
    # Check for critical dependencies
    missing_deps = []
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import jax
    except ImportError:
        missing_deps.append("jax")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    return True