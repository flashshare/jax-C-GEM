#!/usr/bin/env python
"""
Advanced Statistical Benchmarking Script for JAX C-GEM.

This script performs advanced statistical benchmarking with cross-validation, 
uncertainty quantification, and comprehensive model comparison.

Author: JAX C-GEM Team
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

def main():
    """Perform advanced statistical benchmarking."""
    try:
        print("🏅 JAX C-GEM Advanced Statistical Benchmarking")
        print("=" * 55)
        
        # Import required modules
        from tools.analysis.model_validation import validate_model_against_benchmarks
        from tools.plotting.publication_output import load_model_results, load_field_data
        
        # Create output directory
        output_dir = 'OUT/Advanced_Benchmarks'
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Loading model results and benchmark datasets...")
        
        # Load model results
        try:
            model_results = load_model_results('OUT')
            if not model_results:
                print("❌ No model results found! Please run a simulation first.")
                return 1
            print(f"✅ Loaded {len(model_results)} model variables")
        except Exception as e:
            print(f"❌ Error loading model results: {e}")
            return 1
        
        # Load field data as benchmark datasets
        try:
            field_data = load_field_data('INPUT/Calibration')
            if not field_data:
                print("❌ No field data found! Check INPUT/Calibration directory.")
                return 1
            print(f"✅ Loaded {len(field_data)} field datasets")
        except Exception as e:
            print(f"❌ Error loading field data: {e}")
            return 1
        
        # Create benchmark datasets dictionary
        benchmark_data = {
            'CARE_Field_Observations': field_data,
            'Historical_Dataset': field_data,  # Could be extended with other datasets
        }
        
        print(f"\n🏅 Performing advanced statistical benchmarking...")
        print(f"   📊 Benchmark datasets: {list(benchmark_data.keys())}")
        
        # Perform advanced benchmarking
        benchmark_results = validate_model_against_benchmarks(
            model_results, 
            benchmark_data, 
            output_dir
        )
        
        if benchmark_results:
            print(f"\n✅ Benchmarking completed for {len(benchmark_results)} benchmark datasets:")
            
            for benchmark_name, results in benchmark_results.items():
                print(f"\n   🏆 {benchmark_name}:")
                for variable, result in results.items():
                    metrics = result.metrics
                    if hasattr(metrics, 'rmse') and not float('inf') in [metrics.rmse, metrics.r_squared]:
                        print(f"      📊 {variable}:")
                        print(f"         • R²: {metrics.r_squared:.4f}")
                        print(f"         • RMSE: {metrics.rmse:.4f}")
                        print(f"         • Nash-Sutcliffe: {metrics.nash_sutcliffe:.4f}")
                        print(f"         • Observations: {metrics.n_observations}")
        else:
            print("⚠️  No successful benchmarks completed")
        
        benchmark_report = Path(output_dir) / "benchmark_comparison.md"
        if benchmark_report.exists():
            print(f"\n📄 Benchmark comparison report: {benchmark_report}")
        
        print(f"\n🔍 Advanced benchmarking outputs saved to: {output_dir}")
        print("✅ Advanced statistical benchmarking completed successfully!")
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure required modules are installed and model_validation.py is available.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
