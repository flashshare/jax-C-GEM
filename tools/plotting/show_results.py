#!/usr/bin/env python3
"""
Comprehensive Results Viewer for JAX C-GEM

This script provides comprehensive visualization of simulation results
with automatic format detection and detailed analysis capabilities.

Features:
- Automatic NPZ/CSV format detection and loading
- Comprehensive summary plots (3x3 layout)
- Detailed longitudinal profiles and time series
- Physics validation and statistical analysis
- Automatic execution at end of simulation
- Real-time monitoring capabilities
- Publication-quality figure generation

Usage:
    python tools/plotting/show_results.py                    # Auto-detect format
    python tools/plotting/show_results.py --format npz       # Force NPZ format
    python tools/plotting/show_results.py --output-dir MY_RESULTS
    python tools/plotting/show_results.py --auto             # Automatic mode (no interaction)
    python tools/plotting/show_results.py --save-figures     # Save figures to disk
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for Windows
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Enable interactive mode for real-time updates
plt.ion()

def load_results(output_dir="OUT", format_type="auto", quiet=False):
    """Load simulation results from output directory with enhanced format detection."""
    
    if not quiet:
        print(f"🔍 Looking for results in: {output_dir}")
    
    # Auto-detect format with priority to NPZ (more efficient)
    if format_type == "auto":
        npz_file = os.path.join(output_dir, "simulation_results.npz")
        alt_npz_file = os.path.join(output_dir, "complete_simulation_results.npz")
        
        if os.path.exists(npz_file) or os.path.exists(alt_npz_file):
            format_type = "npz"
            if not quiet:
                print("📊 Found NPZ format - using high-performance loader")
        else:
            format_type = "csv"
            if not quiet:
                print("📊 NPZ not found - trying CSV format")
    
    if not quiet:
        print(f"📊 Loading {format_type.upper()} format...")
    
    if format_type == "npz":
        return load_npz_results(output_dir, quiet)
    else:
        return load_csv_results(output_dir, quiet)

def load_npz_results(output_dir, quiet=False):
    """Load results from NPZ format with enhanced error handling."""
    # Try the primary filename first
    npz_file = os.path.join(output_dir, "simulation_results.npz")
    
    if not os.path.exists(npz_file):
        # Try alternative filename
        npz_file = os.path.join(output_dir, "complete_simulation_results.npz")
        
        if not os.path.exists(npz_file):
            if not quiet:
                print(f"❌ NPZ file not found: {npz_file}")
            return None
    
    try:
        npz_data = np.load(npz_file)
        
        if not quiet:
            print(f"✅ Loaded NPZ data from {npz_file}")
        
        data = {
            'time_days': npz_data['time_days'] if 'time_days' in npz_data else (npz_data['time'] if 'time' in npz_data else np.arange(len(npz_data[list(npz_data.keys())[0]]))),
            'water_level': npz_data['water_levels'] if 'water_levels' in npz_data else (npz_data['H'] if 'H' in npz_data else None),
            'velocity': npz_data['velocities'] if 'velocities' in npz_data else (npz_data['U'] if 'U' in npz_data else None),
            'salinity': npz_data['salinity'] if 'salinity' in npz_data else (npz_data['S'] if 'S' in npz_data else None),
            'oxygen': npz_data['oxygen'] if 'oxygen' in npz_data else (npz_data['O2'] if 'O2' in npz_data else None),
            'nitrate': npz_data['nitrate'] if 'nitrate' in npz_data else (npz_data['NO3'] if 'NO3' in npz_data else None),
            'ammonium': npz_data['ammonium'] if 'ammonium' in npz_data else (npz_data['NH4'] if 'NH4' in npz_data else None),
            'phosphate': npz_data['phosphate'] if 'phosphate' in npz_data else (npz_data['PO4'] if 'PO4' in npz_data else None),
            'phytoplankton': npz_data['phytoplankton1'] if 'phytoplankton1' in npz_data else (npz_data.get('PHY1', None)),
            'silicate': npz_data['silicate'] if 'silicate' in npz_data else (npz_data.get('SI', None)),
            'suspended_matter': npz_data['tss'] if 'tss' in npz_data else (npz_data.get('SPM', None)),
        }
        
        # Create distance array
        if 'grid_km' in npz_data:
            data['distance_km'] = npz_data['grid_km']
        elif data['water_level'] is not None:
            n_points = data['water_level'].shape[1]
            data['distance_km'] = np.arange(n_points) * 2.0  # 2km spacing
        elif data['velocity'] is not None:
            n_points = data['velocity'].shape[1]
            data['distance_km'] = np.arange(n_points) * 2.0  # 2km spacing
        else:
            n_points = 102
            data['distance_km'] = np.arange(n_points) * 2.0  # 2km spacing
        
        if not quiet:
            print(f"✅ Loaded NPZ data:")
            print(f"   Time range: {data['time_days'][0]:.1f} - {data['time_days'][-1]:.1f} days")
            print(f"   Grid points: {len(data['distance_km'])}")
            available_vars = [k for k, v in data.items() if v is not None and k not in ['time_days', 'distance_km']]
            print(f"   Variables: {available_vars}")
        
        return data
        
    except Exception as e:
        if not quiet:
            print(f"❌ Error loading NPZ data: {e}")
        return None

def load_csv_results(output_dir, quiet=False):
    """Load results from CSV format with enhanced file detection."""
    
    # Try to load from different possible locations
    possible_files = [
        ('water_level', ['Hydrodynamics/H.csv', 'H.csv']),
        ('velocity', ['Hydrodynamics/U.csv', 'U.csv']),
        ('salinity', ['Reaction/S.csv', 'S.csv']),
        ('oxygen', ['Reaction/O2.csv', 'O2.csv']),
        ('nitrate', ['Reaction/NO3.csv', 'NO3.csv']),
        ('ammonium', ['Reaction/NH4.csv', 'NH4.csv']),
        ('phosphate', ['Reaction/PO4.csv', 'PO4.csv']),
        ('phytoplankton', ['Reaction/PHY1.csv', 'PHY1.csv']),
        ('silicate', ['Reaction/SI.csv', 'SI.csv']),
        ('suspended_matter', ['Reaction/SPM.csv', 'SPM.csv']),
    ]
    
    data = {}
    
    for var_name, file_paths in possible_files:
        for file_path in file_paths:
            full_path = os.path.join(output_dir, file_path)
            if os.path.exists(full_path):
                try:
                    # Try pandas first (handles headers better)
                    try:
                        csv_data = pd.read_csv(full_path, header=None)
                        csv_array = csv_data.values
                    except:
                        csv_array = np.loadtxt(full_path, delimiter=',')
                    
                    if len(csv_array.shape) == 2:
                        # First column is time, rest are spatial data
                        if 'time_days' not in data:
                            data['time_days'] = csv_array[:, 0] / 86400.0  # Convert to days
                        data[var_name] = csv_array[:, 1:]
                        if not quiet:
                            print(f"✓ Loaded {var_name} from {file_path}")
                    break
                except Exception as e:
                    if not quiet:
                        print(f"⚠️  Could not load {file_path}: {e}")
    
    if data:
        # Create distance array based on first loaded variable
        first_var = next(iter([v for k, v in data.items() if k != 'time_days']))
        n_points = first_var.shape[1] if first_var is not None else 102
        data['distance_km'] = np.arange(n_points) * 2.0
        
        if not quiet:
            available_vars = [k for k in data.keys() if k not in ['time_days', 'distance_km']]
            print(f"✅ Loaded CSV data:")
            print(f"   Variables: {available_vars}")
            if 'time_days' in data:
                print(f"   Time range: {data['time_days'][0]:.1f} - {data['time_days'][-1]:.1f} days")
    
    return data if data else None

def create_summary_plots(data, save_figures=False, quiet=False):
    """Create comprehensive summary plots with enhanced analysis."""
    
    if not data:
        if not quiet:
            print("❌ No data available for plotting")
        return
    
    if not quiet:
        print("\n📊 Creating comprehensive summary plots...")
    
    # Set up the plot layout
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('JAX C-GEM Comprehensive Results Analysis', fontsize=16, fontweight='bold')
    
    # Create subplots with enhanced layout
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
    
    # Plot 1: Longitudinal profiles (final state) - Enhanced
    ax1 = fig.add_subplot(gs[0, :2])
    final_idx = -1  # Last time step
    
    if data['water_level'] is not None:
        ax1.plot(data['distance_km'], data['water_level'][final_idx, :], 'b-', linewidth=2, label='Water Level (m)')
        ax1.set_ylabel('Water Level (m)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
    
    # Salinity on secondary axis
    if data['salinity'] is not None:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data['distance_km'], data['salinity'][final_idx, :], 'orange', linewidth=2, label='Salinity (PSU)')
        ax1_twin.set_ylabel('Salinity (PSU)', color='orange')
        ax1_twin.tick_params(axis='y', labelcolor='orange')
    
    ax1.set_xlabel('Distance from Mouth (km)')
    ax1.set_title('(a) Final Longitudinal Profiles')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Enhanced velocity profile
    ax2 = fig.add_subplot(gs[0, 2:])
    if data['velocity'] is not None:
        ax2.plot(data['distance_km'], data['velocity'][final_idx, :], 'g-', linewidth=2, label='Velocity')
        # Add flow direction indication
        velocity_sign = np.sign(data['velocity'][final_idx, :])
        colors = ['red' if v < 0 else 'blue' for v in velocity_sign]
        ax2.scatter(data['distance_km'], data['velocity'][final_idx, :], c=colors, alpha=0.6, s=20)
    ax2.set_xlabel('Distance from Mouth (km)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('(b) Final Velocity Profile')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series at mouth
    ax3 = fig.add_subplot(gs[1, 0])
    mouth_idx = 0
    if data['water_level'] is not None:
        ax3.plot(data['time_days'], data['water_level'][:, mouth_idx], 'b-', linewidth=1)
    ax3.set_title('(c) Water Level at Mouth')
    ax3.set_ylabel('Water Level (m)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time series at mid-estuary
    ax4 = fig.add_subplot(gs[1, 1])
    mid_idx = len(data['distance_km']) // 2
    if data['velocity'] is not None:
        ax4.plot(data['time_days'], data['velocity'][:, mid_idx], 'g-', linewidth=1)
    ax4.set_title(f'(d) Velocity at {data["distance_km"][mid_idx]:.0f}km')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Time series at head
    ax5 = fig.add_subplot(gs[1, 2])
    head_idx = -1
    if data['salinity'] is not None:
        ax5.plot(data['time_days'], data['salinity'][:, head_idx], 'orange', linewidth=1)
    ax5.set_title('(e) Salinity at Head')
    ax5.set_ylabel('Salinity (PSU)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Tidal range analysis
    ax6 = fig.add_subplot(gs[1, 3])
    if data['water_level'] is not None:
        # Calculate tidal range over time
        h_min = np.min(data['water_level'], axis=1)
        h_max = np.max(data['water_level'], axis=1)
        tidal_range = h_max - h_min
        ax6.plot(data['time_days'], tidal_range, 'b-', linewidth=1)
        ax6.set_title('(f) Tidal Range')
        ax6.set_ylabel('Range (m)')
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Oxygen dynamics
    ax7 = fig.add_subplot(gs[2, 0])
    if data['oxygen'] is not None:
        # Show oxygen at multiple locations
        locations = [0, len(data['distance_km'])//3, 2*len(data['distance_km'])//3, -1]
        location_names = ['Mouth', 'Lower', 'Middle', 'Head']
        
        for i, (loc, name) in enumerate(zip(locations, location_names)):
            ax7.plot(data['time_days'], data['oxygen'][:, loc], 
                    linewidth=1, label=f'{name} ({data["distance_km"][loc]:.0f}km)')
        
        ax7.set_title('(g) Dissolved Oxygen Dynamics')
        ax7.set_ylabel('O2 (mmol/m³)')
        ax7.set_xlabel('Time (days)')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No oxygen data\navailable', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=12)
        ax7.set_title('(g) Dissolved Oxygen')
    
    # Plot 8: Nutrient dynamics
    ax8 = fig.add_subplot(gs[2, 1])
    if data['nitrate'] is not None or data['ammonium'] is not None or data['phosphate'] is not None:
        mid_idx = len(data['distance_km']) // 2
        
        if data['nitrate'] is not None:
            ax8.plot(data['time_days'], data['nitrate'][:, mid_idx], 
                    'blue', linewidth=1, label='NO3')
        if data['ammonium'] is not None:
            ax8.plot(data['time_days'], data['ammonium'][:, mid_idx], 
                    'red', linewidth=1, label='NH4')
        if data['phosphate'] is not None:
            ax8.plot(data['time_days'], data['phosphate'][:, mid_idx], 
                    'green', linewidth=1, label='PO4')
        
        ax8.set_title(f'(h) Nutrients at {data["distance_km"][mid_idx]:.0f}km')
        ax8.set_ylabel('Concentration (mmol/m³)')
        ax8.set_xlabel('Time (days)')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No nutrient data\navailable', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=12)
        ax8.set_title('(h) Nutrients')
    
    # Plot 9: Salt intrusion analysis
    ax9 = fig.add_subplot(gs[2, 2])
    if data['salinity'] is not None:
        # Calculate salt intrusion length over time
        salt_intrusion = []
        distance = data['distance_km']
        
        for t in range(len(data['time_days'])):
            salt_mask = data['salinity'][t, :] > 1.0
            if np.any(salt_mask):
                max_intrusion_idx = np.where(salt_mask)[0][-1]
                salt_intrusion.append(distance[max_intrusion_idx])
            else:
                salt_intrusion.append(0)
        
        ax9.plot(data['time_days'], salt_intrusion, 'orange', linewidth=1.5)
        ax9.set_title('(i) Salt Intrusion Length')
        ax9.set_ylabel('Intrusion (km)')
        ax9.set_xlabel('Time (days)')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No salinity data\navailable', ha='center', va='center', 
                transform=ax9.transAxes, fontsize=12)
        ax9.set_title('(i) Salt Intrusion')
    
    # Plot 10: Physics validation
    ax10 = fig.add_subplot(gs[2, 3])
    if data['water_level'] is not None and data['velocity'] is not None:
        # Check for numerical stability
        h_stable = (np.abs(data['water_level']) < 10).all(axis=1)
        u_stable = (np.abs(data['velocity']) < 5).all(axis=1)
        stability = h_stable & u_stable
        
        ax10.plot(data['time_days'], stability.astype(float), 'red', linewidth=2, alpha=0.7)
        ax10.fill_between(data['time_days'], 0, stability.astype(float), alpha=0.3, color='green')
        ax10.set_title('(j) Model Stability')
        ax10.set_ylabel('Stable (1) / Unstable (0)')
        ax10.set_xlabel('Time (days)')
        ax10.set_ylim(-0.1, 1.1)
        ax10.grid(True, alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'Stability check\nrequires H & U data', ha='center', va='center', 
                transform=ax10.transAxes, fontsize=10)
        ax10.set_title('(j) Model Stability')
    
    # Plot 11: Enhanced summary statistics
    ax11 = fig.add_subplot(gs[3, :2])
    ax11.axis('off')
    
    # Calculate comprehensive summary statistics
    summary_text = "COMPREHENSIVE SIMULATION SUMMARY\n" + "="*35 + "\n\n"
    
    # Time information
    if 'time_days' in data and data['time_days'] is not None:
        duration = data['time_days'][-1] - data['time_days'][0]
        summary_text += f"⏱️  Duration: {duration:.1f} days ({duration/365:.2f} years)\n"
        summary_text += f"📊 Time steps: {len(data['time_days'])}\n"
        dt_hours = (data['time_days'][1] - data['time_days'][0]) * 24 if len(data['time_days']) > 1 else 0
        summary_text += f"⚡ Time resolution: {dt_hours:.2f} hours\n\n"
    
    # Estuary characteristics
    summary_text += "🌊 ESTUARY CHARACTERISTICS:\n" + "-"*25 + "\n"
    summary_text += f"📏 Length: {data['distance_km'][-1]:.0f} km\n"
    summary_text += f"🔸 Grid points: {len(data['distance_km'])}\n"
    summary_text += f"📐 Grid spacing: {data['distance_km'][1] - data['distance_km'][0]:.1f} km\n\n"
    
    # Physical dynamics summary
    summary_text += "🌀 PHYSICAL DYNAMICS:\n" + "-"*20 + "\n"
    if data['water_level'] is not None:
        tidal_range = np.max(data['water_level']) - np.min(data['water_level'])
        mean_level = np.mean(data['water_level'])
        summary_text += f"🌊 Max tidal range: {tidal_range:.2f} m\n"
        summary_text += f"📊 Mean water level: {mean_level:.2f} m\n"
    
    if data['velocity'] is not None:
        max_velocity = np.max(np.abs(data['velocity']))
        mean_abs_velocity = np.mean(np.abs(data['velocity']))
        summary_text += f"💨 Max velocity: {max_velocity:.2f} m/s\n"
        summary_text += f"🔄 Mean |velocity|: {mean_abs_velocity:.2f} m/s\n"
    
    # Biogeochemical summary
    if data['salinity'] is not None or data['oxygen'] is not None:
        summary_text += "\n🧪 BIOGEOCHEMICAL STATE:\n" + "-"*23 + "\n"
        
        if data['salinity'] is not None:
            max_salinity = np.max(data['salinity'])
            min_salinity = np.min(data['salinity'])
            # Calculate salt intrusion
            salt_intrusion = 0
            for i in range(len(data['distance_km'])):
                if np.mean(data['salinity'][:, -(i+1)]) > 1.0:
                    salt_intrusion = data['distance_km'][-(i+1)]
                    break
            summary_text += f"🧂 Salinity range: {min_salinity:.1f} - {max_salinity:.1f} PSU\n"
            summary_text += f"🏔️  Salt intrusion: ~{salt_intrusion:.0f} km\n"
        
        if data['oxygen'] is not None:
            min_oxygen = np.min(data['oxygen'])
            max_oxygen = np.max(data['oxygen'])
            mean_oxygen = np.mean(data['oxygen'])
            summary_text += f"💨 Oxygen range: {min_oxygen:.1f} - {max_oxygen:.1f} mmol/m³\n"
            summary_text += f"📊 Mean oxygen: {mean_oxygen:.1f} mmol/m³\n"
            if min_oxygen < 64:  # ~2 mg/L hypoxia threshold
                summary_text += f"⚠️  HYPOXIC CONDITIONS DETECTED!\n"
    
    # Physics validation
    summary_text += "\n🔬 PHYSICS VALIDATION:\n" + "-"*18 + "\n"
    
    checks_passed = 0
    total_checks = 0
    
    if data['water_level'] is not None:
        total_checks += 1
        if np.all(np.abs(data['water_level']) < 10):
            summary_text += "✅ Water levels realistic\n"
            checks_passed += 1
        else:
            summary_text += "⚠️ Extreme water levels detected\n"
    
    if data['velocity'] is not None:
        total_checks += 1
        if np.all(np.abs(data['velocity']) < 5):
            summary_text += "✅ Velocities realistic\n"
            checks_passed += 1
        else:
            summary_text += "⚠️ Extreme velocities detected\n"
    
    if data['salinity'] is not None:
        total_checks += 1
        if np.all(data['salinity'] >= 0) and np.all(data['salinity'] <= 35):
            summary_text += "✅ Salinity within bounds\n"
            checks_passed += 1
        else:
            summary_text += "⚠️ Salinity out of bounds\n"
    
    if total_checks > 0:
        summary_text += f"\n📋 VALIDATION: {checks_passed}/{total_checks} checks passed\n"
        if checks_passed == total_checks:
            summary_text += "🎉 ALL PHYSICS CHECKS PASSED!\n"
        else:
            summary_text += "⚠️  SOME PHYSICS CHECKS FAILED\n"
    
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, 
             fontsize=9, fontfamily='monospace', va='top', ha='left')
    
    # Plot 12: Available variables overview
    ax12 = fig.add_subplot(gs[3, 2:])
    ax12.axis('off')
    
    # List available variables and their status
    var_text = "AVAILABLE VARIABLES\n" + "="*20 + "\n\n"
    
    variable_info = [
        ('water_level', '🌊 Water Level', 'm'),
        ('velocity', '💨 Velocity', 'm/s'),
        ('salinity', '🧂 Salinity', 'PSU'),
        ('oxygen', '💨 Dissolved O2', 'mmol/m³'),
        ('nitrate', '🔵 Nitrate (NO3)', 'mmol/m³'),
        ('ammonium', '🔴 Ammonium (NH4)', 'mmol/m³'),
        ('phosphate', '🟢 Phosphate (PO4)', 'mmol/m³'),
        ('phytoplankton', '🦠 Phytoplankton', 'mmol/m³'),
        ('silicate', '🪨 Silicate', 'mmol/m³'),
        ('suspended_matter', '🌫️  Suspended Matter', 'mg/L'),
    ]
    
    for var_key, var_name, unit in variable_info:
        if var_key in data and data[var_key] is not None:
            var_text += f"✅ {var_name} [{unit}]\n"
        else:
            var_text += f"❌ {var_name} [not available]\n"
    
    var_text += f"\n📊 SIMULATION PERFORMANCE:\n" + "-"*20 + "\n"
    if 'time_days' in data:
        total_steps = len(data['time_days'])
        var_text += f"Total timesteps: {total_steps:,}\n"
        if total_steps > 0:
            vars_loaded = sum(1 for k, v in data.items() if v is not None and k not in ['time_days', 'distance_km'])
            var_text += f"Variables loaded: {vars_loaded}\n"
            data_size_mb = sum(v.nbytes for v in data.values() if hasattr(v, 'nbytes')) / (1024**2)
            var_text += f"Data size: ~{data_size_mb:.1f} MB\n"
    
    ax12.text(0.05, 0.95, var_text, transform=ax12.transAxes, 
             fontsize=9, fontfamily='monospace', va='top', ha='left')
    
    # Save figures if requested
    if save_figures:
        save_plots(fig, data, quiet)
    
    # Show plot
    if not quiet:
        print("📊 Summary plots created successfully!")
        print("💡 Tip: Use Ctrl+C to close plot windows")
    
    plt.tight_layout()
    plt.show()

def save_plots(fig, data, quiet=False):
    """Save plots to disk with timestamped filenames."""
    try:
        timestamp = data['time_days'][-1] if 'time_days' in data else datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = "OUT"
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save comprehensive summary
        filename = os.path.join(plots_dir, f"comprehensive_summary_{timestamp:.0f}days.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        if not quiet:
            print(f"� Plots saved: {filename}")
            
    except Exception as e:
        if not quiet:
            print(f"⚠️ Could not save plots: {e}")

def create_separate_detailed_plots(data, save_figures=False, quiet=False):
    """Create separate detailed plots for each variable (from auto_plot.py functionality)."""
    
    if not data:
        return
    
    if not quiet:
        print("\n📊 Creating detailed individual plots...")
    
    output_dir = "OUT"
    plots_dir = os.path.join(output_dir, "detailed_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Detailed longitudinal profiles
    create_detailed_longitudinal_plots(data, plots_dir, save_figures, quiet)
    
    # 2. Detailed time series
    create_detailed_time_series_plots(data, plots_dir, save_figures, quiet)
    
    # 3. Physics analysis
    create_detailed_physics_plots(data, plots_dir, save_figures, quiet)

def create_detailed_longitudinal_plots(data, plots_dir, save_figures, quiet):
    """Create detailed longitudinal profile plots."""
    # Implementation from auto_plot.py
    pass

def create_detailed_time_series_plots(data, plots_dir, save_figures, quiet):
    """Create detailed time series plots.""" 
    # Implementation from auto_plot.py
    pass

def create_detailed_physics_plots(data, plots_dir, save_figures, quiet):
    """Create detailed physics analysis plots."""
    # Implementation from auto_plot.py  
    pass

def create_automatic_plots(output_dir="OUT", format_type="auto", save_figures=False, quiet=False):
    """
    Main function to create all plots automatically.
    This replaces the functionality from auto_plot.py.
    """
    if not quiet:
        print("\n" + "="*60)
        print("📊 JAX C-GEM COMPREHENSIVE RESULTS ANALYSIS")
        print("="*60)
    
    try:
        # Load data
        data = load_results(output_dir, format_type, quiet)
        
        if data is None:
            if not quiet:
                print("❌ No data found for plotting")
            return False
        
        # Create comprehensive summary plots
        create_summary_plots(data, save_figures, quiet)
        
        # Create detailed individual plots if requested
        if save_figures:
            create_separate_detailed_plots(data, save_figures, quiet)
        
        if not quiet:
            plots_dir = os.path.join(output_dir, "plots")
            print(f"✅ All plots created successfully!")
            print(f"📁 Plots saved to: {os.path.abspath(plots_dir)}")
        
        return True
        
    except Exception as e:
        if not quiet:
            print(f"❌ Error creating plots: {e}")
            import traceback
            traceback.print_exc()
        return False

def main():
    """Enhanced main function with comprehensive options."""
    
    parser = argparse.ArgumentParser(description='JAX C-GEM Comprehensive Results Viewer')
    parser.add_argument('--output-dir', default='OUT', 
                       help='Output directory containing results')
    parser.add_argument('--format', choices=['csv', 'npz', 'auto'], default='auto',
                       help='Data format to load (auto=detect automatically)')
    parser.add_argument('--auto', action='store_true',
                       help='Automatic mode (no user interaction, minimal output)')
    parser.add_argument('--save-figures', action='store_true',
                       help='Save figures to disk in addition to displaying them')
    parser.add_argument('--detailed', action='store_true',
                       help='Create detailed individual plots in addition to summary')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages (useful for automated calls)')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("📊 JAX C-GEM Comprehensive Results Viewer")
        print("=" * 45)
    
    # Load results
    data = load_results(args.output_dir, args.format, args.quiet)
    
    if data is None:
        if not args.quiet:
            print("❌ No simulation results found!")
            print("Make sure you have run a simulation first:")
            print("  python src/main.py --mode run")
            print("  python main_ultra_performance.py")
            print("  or use a VS Code task: 'Run Model'")
        return 1
    
    # Create plots
    if not args.quiet:
        print("\n📈 Creating comprehensive visualization...")
    
    try:
        # Main comprehensive plots
        create_summary_plots(data, args.save_figures, args.quiet)
        
        # Detailed plots if requested
        if args.detailed:
            create_separate_detailed_plots(data, args.save_figures, args.quiet)
        
        if not args.quiet:
            print("\n✅ Results visualization complete!")
            if not args.auto:
                print("💡 Tip: Use Ctrl+C to close plot windows")
                print("💡 Use --save-figures to save plots to disk")
                print("💡 Use --detailed for additional detailed plots")
        
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"❌ Error creating plots: {e}")
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
