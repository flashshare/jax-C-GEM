#!/usr/bin/env python3
"""
Biogeochemistry Diagnosis Tool
Analyzes the JAX C-GEM biogeochemical model output to understand validation issues
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_biogeochemistry():
    """Analyze biogeochemical model output to diagnose validation issues."""
    
    print("🔬 JAX C-GEM Biogeochemistry Diagnosis")
    print("=" * 50)
    
    # Load results
    print("📊 Loading simulation results...")
    try:
        results = np.load('OUT/complete_simulation_results.npz')
        print(f"✅ Loaded results with {len(results.files)} variables")
        
        # Extract key variables
        time = results['time']
        time_days = time / (24 * 3600)  # Convert to days
        salinity = results['S']  # Use actual keys from NPZ file
        oxygen = results['O2'] 
        nitrate = results['NO3']
        ammonium = results['NH4']
        phosphate = results['PO4']
        toc = results['TOC']
        
        # Get PC station (around 116km) - cell index ~58
        pc_index = 58
        print(f"📍 Analyzing PC station at grid cell {pc_index}")
        
        # Extract time series at PC station
        sal_ts = salinity[:, pc_index]
        o2_ts = oxygen[:, pc_index] 
        no3_ts = nitrate[:, pc_index]
        nh4_ts = ammonium[:, pc_index]
        po4_ts = phosphate[:, pc_index]
        toc_ts = toc[:, pc_index]
        
        print("\n📊 TEMPORAL VARIATION ANALYSIS:")
        print("-" * 40)
        print(f"Salinity:  {sal_ts.min():.3f} - {sal_ts.max():.3f} PSU (range: {sal_ts.max()-sal_ts.min():.3f})")
        print(f"Oxygen:    {o2_ts.min():.1f} - {o2_ts.max():.1f} mmol/m³ (range: {o2_ts.max()-o2_ts.min():.1f})")
        print(f"Nitrate:   {no3_ts.min():.1f} - {no3_ts.max():.1f} mmol/m³ (range: {no3_ts.max()-no3_ts.min():.1f})")
        print(f"Ammonium:  {nh4_ts.min():.3f} - {nh4_ts.max():.3f} mmol/m³ (range: {nh4_ts.max()-nh4_ts.min():.3f})")
        print(f"Phosphate: {po4_ts.min():.3f} - {po4_ts.max():.3f} mmol/m³ (range: {po4_ts.max()-po4_ts.min():.3f})")
        print(f"TOC:       {toc_ts.min():.3f} - {toc_ts.max():.3f} mmol/m³ (range: {toc_ts.max()-toc_ts.min():.3f})")
        
        # Check if variables are static (no seasonal variation)
        print("\n🔍 SEASONAL RESPONSIVENESS CHECK:")
        print("-" * 40)
        
        # Calculate coefficients of variation (CV = std/mean)
        variables = {
            'Salinity': sal_ts,
            'Oxygen': o2_ts,
            'Nitrate': no3_ts,
            'Ammonium': nh4_ts,
            'Phosphate': po4_ts,
            'TOC': toc_ts
        }
        
        for name, ts in variables.items():
            cv = np.std(ts) / np.mean(ts) * 100 if np.mean(ts) > 0 else 0
            status = "🔴 STATIC" if cv < 1 else "🟡 WEAK" if cv < 5 else "🟢 RESPONSIVE"
            print(f"{name:10s}: CV = {cv:5.2f}% {status}")
        
        # Analyze spatial gradients
        print("\n📏 SPATIAL GRADIENT ANALYSIS:")
        print("-" * 40)
        
        # Use final time step for spatial analysis
        final_step = -1
        
        print("Final spatial gradients (mouth → head):")
        print(f"Salinity:  {salinity[final_step, 0]:.3f} → {salinity[final_step, -1]:.3f} PSU")
        print(f"Oxygen:    {oxygen[final_step, 0]:.1f} → {oxygen[final_step, -1]:.1f} mmol/m³") 
        print(f"Nitrate:   {nitrate[final_step, 0]:.1f} → {nitrate[final_step, -1]:.1f} mmol/m³")
        print(f"Ammonium:  {ammonium[final_step, 0]:.3f} → {ammonium[final_step, -1]:.3f} mmol/m³")
        print(f"Phosphate: {phosphate[final_step, 0]:.3f} → {phosphate[final_step, -1]:.3f} mmol/m³")
        print(f"TOC:       {toc[final_step, 0]:.3f} → {toc[final_step, -1]:.3f} mmol/m³")
        
        # Create diagnostic plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('JAX C-GEM Biogeochemistry Diagnosis', fontsize=16, fontweight='bold')
        
        # Plot time series at PC station
        axes[0, 0].plot(time_days[2400:] - time_days[2400], sal_ts[2400:])  # Skip warmup
        axes[0, 0].set_title('Salinity Time Series (PC Station)')
        axes[0, 0].set_ylabel('Salinity (PSU)')
        
        axes[0, 1].plot(time_days[2400:] - time_days[2400], o2_ts[2400:])
        axes[0, 1].set_title('Oxygen Time Series (PC Station)')
        axes[0, 1].set_ylabel('O₂ (mmol/m³)')
        
        axes[0, 2].plot(time_days[2400:] - time_days[2400], no3_ts[2400:])
        axes[0, 2].set_title('Nitrate Time Series (PC Station)')
        axes[0, 2].set_ylabel('NO₃ (mmol/m³)')
        
        # Plot spatial profiles (final time step)
        grid_km = np.arange(102) * 2.0  # 2 km spacing
        
        axes[1, 0].plot(grid_km, salinity[final_step, :])
        axes[1, 0].set_title('Salinity Spatial Profile (Final)')
        axes[1, 0].set_xlabel('Distance from mouth (km)')
        axes[1, 0].set_ylabel('Salinity (PSU)')
        
        axes[1, 1].plot(grid_km, oxygen[final_step, :])
        axes[1, 1].set_title('Oxygen Spatial Profile (Final)')
        axes[1, 1].set_xlabel('Distance from mouth (km)')
        axes[1, 1].set_ylabel('O₂ (mmol/m³)')
        
        axes[1, 2].plot(grid_km, nitrate[final_step, :])
        axes[1, 2].set_title('Nitrate Spatial Profile (Final)')
        axes[1, 2].set_xlabel('Distance from mouth (km)')
        axes[1, 2].set_ylabel('NO₃ (mmol/m³)')
        
        plt.tight_layout()
        plt.savefig('biogeochemistry_diagnosis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Diagnostic plot saved: biogeochemistry_diagnosis.png")
        
        # Load field data for comparison
        print("\n🔬 FIELD DATA COMPARISON:")
        print("-" * 40)
        
        try:
            care_data = pd.read_csv('INPUT/Calibration/CARE_2017-2018.csv')
            print(f"✅ CARE field data: {len(care_data)} observations")
            
            # Show field data ranges for comparison
            if 'NH4_mg_per_L' in care_data.columns:
                nh4_field = care_data['NH4_mg_per_L'].dropna()
                print(f"Field NH4: {nh4_field.min():.3f} - {nh4_field.max():.3f} mg/L")
                
            if 'NO3_mg_per_L' in care_data.columns:
                no3_field = care_data['NO3_mg_per_L'].dropna()
                print(f"Field NO3: {no3_field.min():.3f} - {no3_field.max():.3f} mg/L")
                
        except Exception as e:
            print(f"❌ Could not load field data: {e}")
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return

if __name__ == "__main__":
    analyze_biogeochemistry()