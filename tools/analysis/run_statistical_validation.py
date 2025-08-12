#!/usr/bin/env python
"""
Statistical Model Validation Script for JAX C-GEM.

This script performs comprehensive statistical validation of model results against field data
using the ModelValidator class with advanced statistical metrics.

Author: Nguyen Truong An
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

def main():
    """Perform comprehensive statistical model validation."""
    try:
        print("🔬 JAX C-GEM Statistical Model Validation")
        print("=" * 50)
        
        # Import required modules
        from tools.analysis.model_validation import ModelValidator
        from tools.plotting.publication_output import load_model_results, load_field_data
        
        # Create output directory
        output_dir = 'OUT/Statistical_Validation'
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Loading model results and field data...")
        
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
        
        # Load field data
        try:
            field_data = load_field_data('INPUT/Calibration')
            if not field_data:
                print("❌ No field data found! Check INPUT/Calibration directory.")
                return 1
            print(f"✅ Loaded {len(field_data)} field datasets")
        except Exception as e:
            print(f"❌ Error loading field data: {e}")
            return 1
        
        print("\n🔬 Performing comprehensive statistical validation...")
        
        # Initialize validator
        validator = ModelValidator(output_dir)
        
        # Perform validation against observations
        validation_results = validator.validate_against_observations(
            model_results, 
            field_data, 
            create_plots=True
        )
        
        if validation_results:
            print(f"\n✅ Validation completed for {len(validation_results)} variables:")
            
            for variable, result in validation_results.items():
                metrics = result.metrics
                if not float('inf') in [metrics.rmse, metrics.r_squared, metrics.nash_sutcliffe]:
                    print(f"   📊 {variable}:")
                    print(f"      • RMSE: {metrics.rmse:.4f}")
                    print(f"      • R²: {metrics.r_squared:.4f}")
                    print(f"      • Nash-Sutcliffe: {metrics.nash_sutcliffe:.4f}")
                    print(f"      • Observations: {metrics.n_observations}")
                else:
                    print(f"   ⚠️  {variable}: Insufficient valid data")
        else:
            print("⚠️  No successful validations completed")
        
        # Generate comprehensive report
        print(f"\n📄 Generating comprehensive validation report...")
        try:
            report_path = validator.generate_validation_report()
            print(f"✅ Statistical validation report saved: {report_path}")
        except Exception as e:
            print(f"❌ Error generating report: {e}")
        
        print(f"\n🔍 Validation outputs saved to: {output_dir}")
        print("✅ Statistical validation completed successfully!")
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
