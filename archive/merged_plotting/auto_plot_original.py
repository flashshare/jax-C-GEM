"""
Automatic plotting for JAX C-GEM model results.
Creates summary plots automatically after simulation completion.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for Windows
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json

# Enable interactive mode
plt.ion()

def create_summary_plots(output_dir="OUT", format_type="csv", model_config=None):
    """
    Create automatic summary plots showing:
    1. Longitudinal profiles (final state)
    2. Time series at key stations
    3. Model performance summary
    
    Args:
        output_dir: Directory containing simulation results
        format_type: "csv" or "npz" format
        model_config: Model configuration dictionary
    """
    
    print("\n" + "="*60)
    print("📊 CREATING AUTOMATIC SUMMARY PLOTS")
    print("="*60)
    
    try:
        if format_type == "csv":
            data = load_csv_results(output_dir)
        else:
            data = load_npz_results(output_dir)
        
        if data is None:
            print("❌ No data found for plotting")
            return
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, "Summary_Plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate all summary plots
        create_longitudinal_profile_plot(data, plots_dir)
        create_time_series_plot(data, plots_dir)
        create_physics_summary_plot(data, plots_dir)
        
        print(f"✅ Summary plots saved to: {os.path.abspath(plots_dir)}")
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

def load_csv_results(output_dir):
    """Load results from CSV files."""
    try:
        # Load hydrodynamic data
        hydro_dir = os.path.join(output_dir, "Hydrodynamics")
        reaction_dir = os.path.join(output_dir, "Reaction")
        
        if not os.path.exists(hydro_dir):
            print(f"❌ Hydrodynamics directory not found: {hydro_dir}")
            return None
        
        data = {}
        
        # Load water level
        h_file = os.path.join(hydro_dir, "H.csv")
        if os.path.exists(h_file):
            h_data = pd.read_csv(h_file, header=None)
            data['time'] = h_data.iloc[:, 0].values  # First column is time
            data['water_level'] = h_data.iloc[:, 1:].values  # Rest are spatial data
        
        # Load velocity
        u_file = os.path.join(hydro_dir, "U.csv")
        if os.path.exists(u_file):
            u_data = pd.read_csv(u_file, header=None)
            data['velocity'] = u_data.iloc[:, 1:].values
        
        # Load salinity
        s_file = os.path.join(reaction_dir, "S.csv")
        if os.path.exists(s_file):
            s_data = pd.read_csv(s_file, header=None)
            data['salinity'] = s_data.iloc[:, 1:].values
        
        # Load oxygen
        o2_file = os.path.join(reaction_dir, "O2.csv")
        if os.path.exists(o2_file):
            o2_data = pd.read_csv(o2_file, header=None)
            data['oxygen'] = o2_data.iloc[:, 1:].values
        
        print(f"✅ Loaded CSV data from {output_dir}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")
        return None

def load_npz_results(output_dir):
    """Load results from NPZ files."""
    try:
        # Look for the main results file
        main_file = os.path.join(output_dir, "complete_simulation_results.npz")
        
        if not os.path.exists(main_file):
            print(f"❌ NPZ file not found: {main_file}")
            return None
        
        npz_data = np.load(main_file)
        
        data = {
            'time': npz_data['time'],
            'water_level': npz_data['H'] if 'H' in npz_data else None,
            'velocity': npz_data['U'] if 'U' in npz_data else None,
            'salinity': npz_data['S'] if 'S' in npz_data else None,
            'oxygen': npz_data['O2'] if 'O2' in npz_data else None
        }
        
        print(f"✅ Loaded NPZ data from {main_file}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading NPZ data: {e}")
        return None

def create_longitudinal_profile_plot(data, plots_dir):
    """Create longitudinal profile plots (final state)."""
    
    # Create distance array (assuming 2km grid spacing, 101 points)
    distance = np.arange(101) * 2.0  # km from mouth
    
    # Use final time step
    final_idx = -1
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Final Longitudinal Profiles (Downstream → Upstream)', fontsize=16, fontweight='bold')
    
    # Water level profile
    if data['water_level'] is not None:
        axes[0,0].plot(distance, data['water_level'][final_idx, :], 'b-', linewidth=2)
        axes[0,0].set_ylabel('Water Level (m)')
        axes[0,0].set_title('Water Surface Elevation')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add station markers
        stations = {'Mouth': 0, 'Station 1': 50, 'Station 2': 100, 'Station 3': 150, 'Head': 200}
        for name, km in stations.items():
            if km <= distance.max():
                axes[0,0].axvline(x=km, color='red', linestyle='--', alpha=0.7)
                axes[0,0].text(km, axes[0,0].get_ylim()[1]*0.9, name, rotation=90, fontsize=8)
    
    # Velocity profile
    if data['velocity'] is not None:
        axes[0,1].plot(distance, data['velocity'][final_idx, :], 'g-', linewidth=2)
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].set_title('Flow Velocity')
        axes[0,1].grid(True, alpha=0.3)
    
    # Salinity profile
    if data['salinity'] is not None:
        axes[1,0].plot(distance, data['salinity'][final_idx, :], 'orange', linewidth=2)
        axes[1,0].set_ylabel('Salinity (PSU)')
        axes[1,0].set_title('Salt Intrusion')
        axes[1,0].set_xlabel('Distance from Mouth (km)')
        axes[1,0].grid(True, alpha=0.3)
    
    # Oxygen profile
    if data['oxygen'] is not None:
        axes[1,1].plot(distance, data['oxygen'][final_idx, :], 'purple', linewidth=2)
        axes[1,1].set_ylabel('Oxygen (mmol/m³)')
        axes[1,1].set_title('Dissolved Oxygen')
        axes[1,1].set_xlabel('Distance from Mouth (km)')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(plots_dir, f"longitudinal_profiles_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Longitudinal profiles saved: {filename}")

def create_time_series_plot(data, plots_dir):
    """Create time series plots at key stations."""
    
    # Station indices (approximate)
    stations = {
        'Mouth (0 km)': 0,
        'Station 1 (50 km)': 25,
        'Station 2 (100 km)': 50,
        'Station 3 (150 km)': 75,
        'Head (200 km)': 100
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Time Series at Key Stations', fontsize=16, fontweight='bold')
    
    time_days = data['time']
    
    # Water level time series
    if data['water_level'] is not None:
        for name, idx in stations.items():
            if idx < data['water_level'].shape[1]:
                axes[0,0].plot(time_days, data['water_level'][:, idx], label=name, linewidth=1.5)
        axes[0,0].set_ylabel('Water Level (m)')
        axes[0,0].set_title('Water Level Variation')
        axes[0,0].legend(fontsize=8)
        axes[0,0].grid(True, alpha=0.3)
    
    # Velocity time series
    if data['velocity'] is not None:
        for name, idx in stations.items():
            if idx < data['velocity'].shape[1]:
                axes[0,1].plot(time_days, data['velocity'][:, idx], label=name, linewidth=1.5)
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].set_title('Flow Velocity')
        axes[0,1].legend(fontsize=8)
        axes[0,1].grid(True, alpha=0.3)
    
    # Salinity time series
    if data['salinity'] is not None:
        for name, idx in stations.items():
            if idx < data['salinity'].shape[1]:
                axes[1,0].plot(time_days, data['salinity'][:, idx], label=name, linewidth=1.5)
        axes[1,0].set_ylabel('Salinity (PSU)')
        axes[1,0].set_title('Salinity Variation')
        axes[1,0].set_xlabel('Time (days)')
        axes[1,0].legend(fontsize=8)
        axes[1,0].grid(True, alpha=0.3)
    
    # Oxygen time series
    if data['oxygen'] is not None:
        for name, idx in stations.items():
            if idx < data['oxygen'].shape[1]:
                axes[1,1].plot(time_days, data['oxygen'][:, idx], label=name, linewidth=1.5)
        axes[1,1].set_ylabel('Oxygen (mmol/m³)')
        axes[1,1].set_title('Dissolved Oxygen')
        axes[1,1].set_xlabel('Time (days)')
        axes[1,1].legend(fontsize=8)
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(plots_dir, f"time_series_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Time series plots saved: {filename}")

def create_physics_summary_plot(data, plots_dir):
    """Create physics summary and validation plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Physics Summary and Model Performance', fontsize=16, fontweight='bold')
    
    # Statistics calculations
    time_days = data['time']
    
    # 1. Water level statistics
    if data['water_level'] is not None:
        h_min = np.min(data['water_level'], axis=1)
        h_max = np.max(data['water_level'], axis=1)
        h_range = h_max - h_min
        
        axes[0,0].plot(time_days, h_range, 'b-', linewidth=2)
        axes[0,0].set_ylabel('Tidal Range (m)')
        axes[0,0].set_title('Tidal Range Over Time')
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Velocity statistics
    if data['velocity'] is not None:
        u_max = np.max(np.abs(data['velocity']), axis=1)
        
        axes[0,1].plot(time_days, u_max, 'g-', linewidth=2)
        axes[0,1].set_ylabel('Max Velocity (m/s)')
        axes[0,1].set_title('Maximum Flow Velocity')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Salt intrusion length
    if data['salinity'] is not None:
        # Find salt intrusion (where salinity > 1 PSU)
        salt_intrusion = []
        distance = np.arange(data['salinity'].shape[1]) * 2.0  # km
        
        for t in range(len(time_days)):
            salt_mask = data['salinity'][t, :] > 1.0
            if np.any(salt_mask):
                max_intrusion_idx = np.where(salt_mask)[0][-1]
                salt_intrusion.append(distance[max_intrusion_idx])
            else:
                salt_intrusion.append(0)
        
        axes[0,2].plot(time_days, salt_intrusion, 'orange', linewidth=2)
        axes[0,2].set_ylabel('Salt Intrusion (km)')
        axes[0,2].set_title('Salt Intrusion Length')
        axes[0,2].grid(True, alpha=0.3)
    
    # 4. Oxygen depletion
    if data['oxygen'] is not None:
        o2_min = np.min(data['oxygen'], axis=1)
        
        axes[1,0].plot(time_days, o2_min, 'purple', linewidth=2)
        axes[1,0].set_ylabel('Min Oxygen (mmol/m³)')
        axes[1,0].set_title('Minimum Oxygen Concentration')
        axes[1,0].set_xlabel('Time (days)')
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Model stability check
    if data['water_level'] is not None and data['velocity'] is not None:
        # Check for extreme values
        h_stable = (np.abs(data['water_level']) < 10).all(axis=1)  # No extreme water levels
        u_stable = (np.abs(data['velocity']) < 5).all(axis=1)     # No extreme velocities
        
        stability = h_stable & u_stable
        
        axes[1,1].plot(time_days, stability.astype(float), 'r-', linewidth=2)
        axes[1,1].set_ylabel('Model Stable (0/1)')
        axes[1,1].set_title('Numerical Stability Check')
        axes[1,1].set_xlabel('Time (days)')
        axes[1,1].set_ylim(-0.1, 1.1)
        axes[1,1].grid(True, alpha=0.3)
    
    # 6. Summary statistics text
    axes[1,2].axis('off')
    summary_text = "SIMULATION SUMMARY\n" + "="*20 + "\n\n"
    
    if data['time'] is not None:
        summary_text += f"Duration: {time_days[-1]:.1f} days\n"
        summary_text += f"Time steps: {len(time_days)}\n\n"
    
    if data['water_level'] is not None:
        h_overall_range = np.max(data['water_level']) - np.min(data['water_level'])
        summary_text += f"Max tidal range: {h_overall_range:.2f} m\n"
    
    if data['velocity'] is not None:
        u_max_overall = np.max(np.abs(data['velocity']))
        summary_text += f"Max velocity: {u_max_overall:.2f} m/s\n"
    
    if data['salinity'] is not None:
        s_max_intrusion = np.max(salt_intrusion) if 'salt_intrusion' in locals() else 0
        summary_text += f"Max salt intrusion: {s_max_intrusion:.0f} km\n"
    
    if data['oxygen'] is not None:
        o2_min_overall = np.min(data['oxygen'])
        summary_text += f"Min oxygen: {o2_min_overall:.1f} mmol/m³\n"
    
    # Physics validation
    summary_text += "\nPHYSICS VALIDATION:\n" + "-"*15 + "\n"
    
    if data['water_level'] is not None:
        if np.all(np.abs(data['water_level']) < 10):
            summary_text += "✓ Water levels realistic\n"
        else:
            summary_text += "⚠ Extreme water levels detected\n"
    
    if data['velocity'] is not None:
        if np.all(np.abs(data['velocity']) < 5):
            summary_text += "✓ Velocities realistic\n"
        else:
            summary_text += "⚠ Extreme velocities detected\n"
    
    if data['salinity'] is not None:
        if np.all(data['salinity'] >= 0) and np.all(data['salinity'] <= 35):
            summary_text += "✓ Salinity within bounds\n"
        else:
            summary_text += "⚠ Salinity out of bounds\n"
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=10, fontfamily='monospace', va='top', ha='left')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(plots_dir, f"physics_summary_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Physics summary saved: {filename}")

def create_real_time_monitoring_plot():
    """Create a real-time monitoring plot that updates during simulation."""
    
    plt.ion()  # Turn on interactive mode
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Real-Time Model Monitoring', fontsize=14, fontweight='bold')
    
    # Initialize empty plots
    line1, = axes[0,0].plot([], [], 'b-', linewidth=2)
    axes[0,0].set_ylabel('Water Level (m)')
    axes[0,0].set_title('Water Surface Profile')
    axes[0,0].grid(True, alpha=0.3)
    
    line2, = axes[0,1].plot([], [], 'g-', linewidth=2)
    axes[0,1].set_ylabel('Velocity (m/s)')
    axes[0,1].set_title('Velocity Profile')
    axes[0,1].grid(True, alpha=0.3)
    
    line3, = axes[1,0].plot([], [], 'orange', linewidth=2)
    axes[1,0].set_ylabel('Salinity (PSU)')
    axes[1,0].set_title('Salinity Profile')
    axes[1,0].set_xlabel('Distance from Mouth (km)')
    axes[1,0].grid(True, alpha=0.3)
    
    line4, = axes[1,1].plot([], [], 'purple', linewidth=2)
    axes[1,1].set_ylabel('Oxygen (mmol/m³)')
    axes[1,1].set_title('Oxygen Profile')
    axes[1,1].set_xlabel('Distance from Mouth (km)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Distance array
    distance = np.arange(101) * 2.0  # km
    
    def update_plot():
        """Update plot with latest snapshot data."""
        snapshot_file = "OUT/live_plot_data.json"
        
        if os.path.exists(snapshot_file):
            try:
                with open(snapshot_file, 'r') as f:
                    data = json.load(f)
                
                # Update plots
                if 'water_level' in data and len(data['water_level']) == len(distance):
                    line1.set_data(distance, data['water_level'])
                    axes[0,0].relim()
                    axes[0,0].autoscale_view()
                
                if 'velocity' in data and len(data['velocity']) == len(distance):
                    line2.set_data(distance, data['velocity'])
                    axes[0,1].relim()
                    axes[0,1].autoscale_view()
                
                if 'salinity' in data and len(data['salinity']) == len(distance):
                    line3.set_data(distance, data['salinity'])
                    axes[1,0].relim()
                    axes[1,0].autoscale_view()
                
                if 'oxygen' in data and len(data['oxygen']) == len(distance):
                    line4.set_data(distance, data['oxygen'])
                    axes[1,1].relim()
                    axes[1,1].autoscale_view()
                
                # Update title with current time
                if 'time' in data:
                    fig.suptitle(f'Real-Time Monitoring (Day {data["time"]:.2f})', 
                               fontsize=14, fontweight='bold')
                
                plt.draw()
                plt.show(block=False)  # Show plot without blocking
                
                # Save plot every few updates
                if hasattr(update_plot, 'call_count'):
                    update_plot.call_count += 1
                else:
                    update_plot.call_count = 1
                
                # Save snapshot every 10 updates
                if update_plot.call_count % 10 == 0:
                    output_dir = os.path.dirname(snapshot_file) if snapshot_file else "OUT"
                    plots_dir = os.path.join(output_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%H%M%S")
                    plt.savefig(os.path.join(plots_dir, f"realtime_{timestamp}.png"), 
                               dpi=150, bbox_inches='tight')
                    print(f"📊 Real-time plot saved: realtime_{timestamp}.png")
                
            except Exception as e:
                print(f"Warning: Could not update real-time plot: {e}")
    
    return update_plot

if __name__ == "__main__":
    # Test the plotting functions
    create_summary_plots("OUT", "csv")
