"""
JAX C-GEM Tools Entry Point

This script provides a unified command-line interface for accessing 
the various tools and utilities in the JAX C-GEM ecosystem.

Usage examples:
    python tools_cli.py visualization --plot-profiles
    python tools_cli.py calibration --optimize
    python tools_cli.py analysis --sensitivity
    python tools_cli.py validation --benchmark

Author: JAX C-GEM Team
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

def setup_paths():
    """Add source directories to path."""
    # Add tools directory to Python path
    tools_dir = Path(__file__).parent.absolute()
    src_dir = tools_dir.parent / 'src'
    
    sys.path.append(str(tools_dir))
    sys.path.append(str(src_dir))

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='JAX C-GEM Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualization', help='Visualization tools')
    viz_parser.add_argument('--plot-profiles', action='store_true', help='Plot longitudinal profiles')
    viz_parser.add_argument('--plot-timeseries', action='store_true', help='Plot time series')
    viz_parser.add_argument('--show-dashboard', action='store_true', help='Show interactive dashboard')
    viz_parser.add_argument('--results-dir', default='OUT', help='Results directory')
    viz_parser.add_argument('--output-dir', default='OUT/Plots', help='Output directory for plots')
    viz_parser.add_argument('--publication-quality', action='store_true', help='Generate publication-quality figures')
    
    # Calibration command
    cal_parser = subparsers.add_parser('calibration', help='Calibration tools')
    cal_parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    cal_parser.add_argument('--config', default='config/model_config.txt', help='Model configuration file')
    cal_parser.add_argument('--output', default='config/calibrated_model_config.txt', help='Output configuration file')
    cal_parser.add_argument('--iterations', type=int, default=50, help='Number of optimization iterations')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='Analysis tools')
    analysis_parser.add_argument('--process-results', action='store_true', help='Process model results')
    analysis_parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    analysis_parser.add_argument('--sensitivity-mode', choices=['oat', 'sobol'], default='oat', 
                                help='Sensitivity analysis mode (one-at-a-time or Sobol)')
    analysis_parser.add_argument('--results-dir', default='OUT', help='Results directory')
    analysis_parser.add_argument('--output-dir', default='OUT/Analysis', help='Output directory for analysis')
    
    # Validation command
    val_parser = subparsers.add_parser('validation', help='Validation tools')
    val_parser.add_argument('--benchmark', action='store_true', help='Run comprehensive benchmark')
    val_parser.add_argument('--verify-physics', action='store_true', help='Verify physical consistency')
    val_parser.add_argument('--profile-benchmark', action='store_true', help='Run longitudinal profile benchmark')
    val_parser.add_argument('--output-dir', default='OUT/Validation', help='Output directory for validation results')
    
    # Monitoring command
    mon_parser = subparsers.add_parser('monitoring', help='Monitoring tools')
    mon_parser.add_argument('--realtime', action='store_true', help='Run model with real-time monitoring')
    mon_parser.add_argument('--snapshot-interval', type=float, default=0.1, help='Snapshot interval in days')
    
    return parser.parse_args()

def run_visualization(args: argparse.Namespace) -> int:
    """Run visualization tools."""
    from visualization.result_visualizer import ResultVisualizer
    
    visualizer = ResultVisualizer(args.results_dir)
    
    if args.plot_profiles or (not args.plot_timeseries and not args.show_dashboard):
        # Default to profiles if nothing specified
        print("📊 Plotting longitudinal profiles...")
        visualizer.create_summary_dashboard(args.output_dir)
        print(f"✅ Plots saved to {args.output_dir}")
    
    if args.plot_timeseries:
        print("📊 Plotting time series...")
        # Add time series plotting code here
    
    if args.show_dashboard:
        print("📊 Launching interactive dashboard...")
        # Add dashboard launching code here
    
    return 0

def run_calibration(args: argparse.Namespace) -> int:
    """Run calibration tools."""
    if args.optimize:
        from calibration.gradient_calibrator import JAXCalibrator
        
        print("🎯 Running gradient-based parameter optimization...")
        calibrator = JAXCalibrator(args.config)
        
        # Look for observation files
        obs_files = {}
        calibration_dir = Path('INPUT/Calibration')
        
        if (calibration_dir / 'longitudinal_profiles.csv').exists():
            obs_files['longitudinal'] = str(calibration_dir / 'longitudinal_profiles.csv')
        
        if (calibration_dir / 'seasonal_cycles.csv').exists():
            obs_files['seasonal'] = str(calibration_dir / 'seasonal_cycles.csv')
        
        if (calibration_dir / 'variability.csv').exists():
            obs_files['variability'] = str(calibration_dir / 'variability.csv')
        
        calibrator.load_observations(obs_files)
        
        # Run optimization
        result = calibrator.optimize_parameters(
            max_iterations=args.iterations,
            learning_rate=0.01
        )
        
        # Save optimized parameters
        calibrator.save_optimized_parameters(result, args.output)
        
        print(f"✅ Optimization complete! Results saved to {args.output}")
    else:
        print("🎯 Please specify a calibration action (--optimize)")
        return 1
    
    return 0

def run_analysis(args: argparse.Namespace) -> int:
    """Run analysis tools."""
    if args.process_results:
        from analysis.result_processor import ResultProcessor
        
        print("🔍 Processing model results...")
        processor = ResultProcessor(args.results_dir)
        
        processor.export_aggregated_csv(args.output_dir)
        processor.export_to_netcdf(f"{args.output_dir}/model_results.nc")
        
        print(f"✅ Processed results saved to {args.output_dir}")
    
    if args.sensitivity:
        from analysis.sensitivity_analyzer import SensitivityAnalyzer
        
        print(f"🔍 Running {args.sensitivity_mode} sensitivity analysis...")
        analyzer = SensitivityAnalyzer('config/model_config.txt')
        
        if args.sensitivity_mode == 'oat':
            results = analyzer.one_at_a_time_sensitivity()
            analyzer.plot_one_at_a_time_results(results, args.output_dir)
        else:
            results = analyzer.global_sensitivity_analysis(n_samples=10)  # Small sample for demo
            analyzer.plot_sobol_indices(results, args.output_dir)
        
        print(f"✅ Sensitivity analysis results saved to {args.output_dir}")
    
    if not args.process_results and not args.sensitivity:
        print("🔍 Please specify an analysis action (--process-results or --sensitivity)")
        return 1
    
    return 0

def run_validation(args: argparse.Namespace) -> int:
    """Run validation tools."""
    if args.benchmark:
        print("🧪 Running comprehensive benchmark...")
        # Import here to avoid circular imports
        from validation.comprehensive_benchmark import run_benchmark
        run_benchmark(args.output_dir)
        print(f"✅ Benchmark results saved to {args.output_dir}")
    
    if args.verify_physics:
        print("🔬 Verifying physics consistency...")
        # Import physics validator
        from diagnostics.physics_validator import check_estuary_physics
        # Simplified example - in practice, would need model results
        print("✅ Physics verification requires model state - use physics_validator directly")
    
    if args.profile_benchmark:
        print("📊 Running longitudinal profile benchmark...")
        # Import profile benchmark
        from validation.longitudinal_profile_benchmark import main as run_profile_benchmark
        run_profile_benchmark()
        print(f"✅ Profile benchmark completed")
    
    if not args.benchmark and not args.verify_physics and not args.profile_benchmark:
        print("🧪 Please specify a validation action")
        return 1
    
    return 0

def run_monitoring(args: argparse.Namespace) -> int:
    """Run monitoring tools."""
    if args.realtime:
        print("📊 Running model with real-time monitoring...")
        # Import monitoring module
        from monitoring.run_model_with_realtime_monitor import main as run_with_monitor
        run_with_monitor()
        print("✅ Monitoring complete")
    else:
        print("📊 Please specify a monitoring action (--realtime)")
        return 1
    
    return 0

def main() -> int:
    """Main entry point."""
    # Setup paths
    setup_paths()
    
    # Parse arguments
    args = parse_arguments()
    
    if not args.command:
        print("❌ No command specified. Use --help for usage information.")
        return 1
    
    # Run appropriate command
    try:
        if args.command == 'visualization':
            return run_visualization(args)
        elif args.command == 'calibration':
            return run_calibration(args)
        elif args.command == 'analysis':
            return run_analysis(args)
        elif args.command == 'validation':
            return run_validation(args)
        elif args.command == 'monitoring':
            return run_monitoring(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    except ImportError as e:
        print(f"❌ Error: Required module not found: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
