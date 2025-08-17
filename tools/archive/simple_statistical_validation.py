"""
Simple Statistical Validation for JAX C-GEM Model

This is a simplified version that focuses on the core functionality
without the complex JAX/pandas compatibility issues.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_model_results(results_dir="OUT"):
    """Load model results from NPZ format."""
    try:
        # Import the working loader from publication_output
        import sys
        sys.path.append(str(Path(__file__).parent.parent / "plotting"))
        from publication_output import load_model_results
        return load_model_results(results_dir)
    except Exception as e:
        print(f"❌ Error loading model results: {e}")
        return {}

def load_field_data(data_dir="INPUT/Calibration"):
    """Load field data for validation."""
    field_data = {}
    data_path = Path(data_dir)
    
    # Load salinity data
    care_file = data_path / "CARE_2017-2018.csv"
    if care_file.exists():
        care_data = pd.read_csv(care_file)
        if 'Salinity' in care_data.columns:
            # Extract numeric salinity values only
            salinity_vals = care_data['Salinity'].dropna()
            salinity_numeric = pd.to_numeric(salinity_vals, errors='coerce').dropna()
            if len(salinity_numeric) > 0:
                field_data['salinity'] = salinity_numeric.values
                print(f"✅ Loaded CARE salinity data: {len(salinity_numeric)} points")
    
    # Load oxygen data
    if care_file.exists():
        care_data = pd.read_csv(care_file)
        if 'DO' in care_data.columns:
            # Extract numeric oxygen values only
            oxygen_vals = care_data['DO'].dropna()
            oxygen_numeric = pd.to_numeric(oxygen_vals, errors='coerce').dropna()
            if len(oxygen_numeric) > 0:
                field_data['oxygen'] = oxygen_numeric.values
                print(f"✅ Loaded CARE oxygen data: {len(oxygen_numeric)} points")
    
    return field_data

def simple_validation_metrics(observed, predicted):
    """Compute basic validation metrics."""
    if len(observed) == 0 or len(predicted) == 0:
        return {'rmse': float('nan'), 'r2': float('nan'), 'n_obs': 0}
    
    # Take minimum length
    min_len = min(len(observed), len(predicted))
    obs = np.array(observed[:min_len])
    pred = np.array(predicted[:min_len])
    
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    obs_clean = obs[valid_mask]
    pred_clean = pred[valid_mask]
    
    if len(obs_clean) == 0:
        return {'rmse': float('nan'), 'r2': float('nan'), 'n_obs': 0}
    
    # Basic metrics
    rmse = np.sqrt(np.mean((obs_clean - pred_clean)**2))
    
    # R-squared
    obs_mean = np.mean(obs_clean)
    ss_res = np.sum((obs_clean - pred_clean)**2)
    ss_tot = np.sum((obs_clean - obs_mean)**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0
    
    return {
        'rmse': rmse,
        'r2': r2,
        'n_obs': len(obs_clean),
        'obs_mean': np.mean(obs_clean),
        'pred_mean': np.mean(pred_clean)
    }

def main():
    """Run simplified statistical validation."""
    print("🔬 JAX C-GEM Simple Statistical Validation")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading model results...")
    model_results = load_model_results("OUT")
    if not model_results:
        print("❌ No model results found")
        return 1
    
    print(f"✅ Model results loaded: {len(model_results)} variables")
    print(f"   Available variables: {list(model_results.keys())[:5]}...")
    
    print("\n🔍 Loading field data...")
    field_data = load_field_data("INPUT/Calibration")
    if not field_data:
        print("❌ No field data found")
        return 1
    
    print(f"✅ Field data loaded: {len(field_data)} datasets")
    
    # Create output directory
    output_dir = Path("OUT/Statistical_Validation")
    output_dir.mkdir(exist_ok=True)
    
    # Validate common variables
    validation_results = {}
    
    for field_var in field_data.keys():
        if field_var in model_results:
            print(f"\n📊 Validating {field_var}...")
            
            model_data = model_results[field_var]
            field_obs = field_data[field_var]
            
            # For 2D model data, take spatial average or first location
            if hasattr(model_data, 'shape') and len(model_data.shape) > 1:
                model_values = np.mean(model_data, axis=1)  # Average over space
            else:
                model_values = np.array(model_data)
                
            # Sample model data to match field data size roughly
            if len(model_values) > len(field_obs) * 2:
                step = len(model_values) // len(field_obs)
                model_values = model_values[::step]
            
            # Compute metrics
            metrics = simple_validation_metrics(field_obs, model_values)
            validation_results[field_var] = metrics
            
            if not np.isnan(metrics['rmse']):
                print(f"   ✅ RMSE: {metrics['rmse']:.3f}")
                print(f"   ✅ R²: {metrics['r2']:.3f}")
                print(f"   ✅ Observations: {metrics['n_obs']}")
                print(f"   ✅ Obs mean: {metrics['obs_mean']:.3f}, Model mean: {metrics['pred_mean']:.3f}")
            else:
                print(f"   ❌ Validation failed - insufficient valid data")
    
    # Generate simple report
    print(f"\n📄 Generating validation report...")
    report_file = output_dir / "validation_summary.md"
    
    with open(report_file, 'w') as f:
        f.write("# JAX C-GEM Validation Summary\\n\\n")
        f.write(f"Generated: {pd.Timestamp.now()}\\n\\n")
        
        f.write("## Validation Results\\n\\n")
        for var, metrics in validation_results.items():
            f.write(f"### {var.title()}\\n")
            if not np.isnan(metrics['rmse']):
                f.write(f"- **RMSE**: {metrics['rmse']:.4f}\\n")
                f.write(f"- **R²**: {metrics['r2']:.4f}\\n")
                f.write(f"- **Observations**: {metrics['n_obs']}\\n")
                f.write(f"- **Field mean**: {metrics['obs_mean']:.3f}\\n")
                f.write(f"- **Model mean**: {metrics['pred_mean']:.3f}\\n")
            else:
                f.write("- **Status**: Insufficient valid data for validation\\n")
            f.write("\\n")
    
    print(f"✅ Validation report saved: {report_file}")
    print(f"✅ Validation completed for {len(validation_results)} variables")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
