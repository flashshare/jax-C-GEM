import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import shared utilities
from core.main_utils import (
    create_argument_parser, print_header, load_configurations,
    analyze_simulation_parameters, select_optimal_output_format,
    print_performance_summary, handle_simulation_error,
    initialize_jax, validate_runtime_environment
)

# JAX imports (needed for simulation logic)
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    # Will be handled by initialize_jax()
    pass

def main():
    # Validate runtime environment
    if not validate_runtime_environment():
        return 1
    
    # Initialize JAX
    if not initialize_jax():
        return 1
    
    # Parse arguments
    parser = create_argument_parser("standard")
    args = parser.parse_args()
    
    # Print header
    print_header("standard")
    
    total_start = time.time()
    
    try:
        # Load configurations using shared utility
        model_config, data_config, data_loader = load_configurations(args.config)
        
        # Analyze simulation parameters
        analysis = analyze_simulation_parameters(model_config)
        
        # Select optimal output format
        output_format = select_optimal_output_format(args.output_format, analysis['expected_outputs'])
        
        # Run simulation with minimal overhead
        success = run_simulation_main(model_config, data_config, data_loader, args, output_format)
            
        if not success:
            print("❌ Simulation failed")
            return 1
        
        # Comprehensive automated physics validation
        if not args.no_physics_check:
            try:
                print("\n" + "="*80)
                print("🔬 AUTOMATED PHYSICS VALIDATION WITH LONGITUDINAL PROFILES")
                print("="*80)
                from core.automated_physics_validation import run_automated_physics_validation
                
                # Determine results file based on output format
                if output_format == 'npz' or output_format == 'auto':
                    results_file = "OUT/complete_simulation_results.npz"
                else:
                    results_file = "OUT/complete_simulation_results.npz"  # Will fall back to CSV
                
                validation_results = run_automated_physics_validation(results_file)
                
                if validation_results and validation_results['validation_results']:
                    overall_status = validation_results['validation_results'].get('overall_status', 'UNKNOWN')
                    
                    print(f"\n🎯 AUTOMATED VALIDATION SUMMARY:")
                    print(f"   Overall Physics Status: {overall_status}")
                    
                    if overall_status == 'EXCELLENT':
                        print("   🎉 EXCELLENT! Model exhibits proper estuarine physics")
                        print("   ✅ All gradients are correct and profiles are smooth")
                        print("   � RECOMMENDED: Proceed to 3-phase verification with observed data")
                        print("   � Next commands:")
                        print("      python tools/verification/phase1_longitudinal_profiles.py")
                        print("      python tools/verification/phase2_tidal_dynamics.py")
                        print("      python tools/verification/phase3_seasonal_cycles.py")
                        
                    elif overall_status in ['GOOD', 'FAIR']:
                        print(f"   ⚠️  {overall_status} status - Some physics issues detected")
                        print("   📊 Check longitudinal profiles and validation figures")
                        print("   � Review recommendations above for specific fixes")
                        print("   📈 Validation plots saved to: OUT/Validation/")
                        
                    else:
                        print("   ❌ POOR physics validation - Major issues detected")
                        print("   🚨 Model does NOT exhibit realistic estuarine behavior")
                        print("   📋 DO NOT proceed to field data verification yet")
                        print("   🔧 Follow debugging checklist above")
                        print("   � Check boundary conditions, parameters, and solver stability")
                    
                    # Always show file locations
                    if validation_results.get('figure_path'):
                        print(f"   📊 Validation figure: {validation_results['figure_path']}")
                    if validation_results.get('csv_path'):
                        print(f"   📊 Mean profiles CSV: {validation_results['csv_path']}")
                        
                else:
                    print("   ❌ Physics validation encountered errors")
                    print("   📋 Check simulation outputs and try manual validation")
                    
            except Exception as e:
                print(f"⚠️ Physics validation failed: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                print("   💡 Try manual validation: python src/comprehensive_debug.py")
            
        print("="*80)
        
        # Print performance summary
        total_time = time.time() - total_start
        print_performance_summary(total_time, "standard")
        return 0
        
    except Exception as e:
        return handle_simulation_error(e, "standard")

def run_simulation_main(model_config, data_config, data_loader, args, output_format):
    """Run JAX C-GEM simulation with high performance."""
    try:
        # Import high-performance simulation engine
        from core.simulation_engine import run_simulation
        from core.hydrodynamics import create_hydro_params, create_initial_hydro_state
        from core.transport import create_transport_params, create_initial_transport_state
        from core.biogeochemistry import create_biogeo_params
        
        print("\n🔧 Initializing simulation components...")
        init_start = time.time()
        
        # Create parameters and state
        hydro_params = create_hydro_params(model_config)
        transport_params = create_transport_params(model_config)
        biogeo_params = create_biogeo_params(model_config)
        hydro_state = create_initial_hydro_state(model_config, hydro_params)
        transport_state = create_initial_transport_state(model_config)
        
        # Create indices
        M = model_config.get('M', 100)
        all_indices = jnp.arange(M)
        even_mask = (all_indices % 2 == 0) & (all_indices >= 2) & (all_indices < M-1)
        odd_mask = (all_indices % 2 == 1) & (all_indices >= 3) & (all_indices < M-2)
        even_indices = jnp.arange(2, M-1, 2)
        odd_indices = jnp.arange(3, M-2, 2)
        grid_indices = jnp.arange(M)
        
        model_state = {
            'hydro_state': hydro_state,
            'transport_state': transport_state,
            'hydro_params': hydro_params,
            'transport_params': transport_params,
            'biogeo_params': biogeo_params,
            'config': model_config,
            'data_loader': data_loader,
            'simulation_config': {
                'start_time': 0,  # Always start at time 0, warmup is part of total simulation
                'end_time': model_config['MAXT_seconds'], 
                'dt': model_config['DELTI'],
                'output_interval': 30  # Output every 30 steps for performance
            },
            'M': model_config['M'],
            'hydro_indices': {
                'even_mask': even_mask,
                'odd_mask': odd_mask,
                'even_indices': even_indices,
                'odd_indices': odd_indices
            },
            'grid_indices': grid_indices,
            'transport_indices': {
                'interface_indices': jnp.arange(2, M-2),  # Interior interfaces: avoid boundaries
                'cell_indices': jnp.arange(2, M-3)        # Interior cells: size M-5 = 97, matches flux sizes
            }
        }
        
        init_time = time.time() - init_start
        print(f"✅ Initialization completed in {init_time:.1f}s")
        
        # Run high-performance simulation
        print("🔥 Starting simulation...")
        sim_start = time.time()
        
        results = run_simulation(model_state)
        
        sim_time = time.time() - sim_start
        print(f"✅ Simulation completed in {sim_time:.1f}s")
        
        # Save results using centralized result writer
        print(f"💾 Saving results in {output_format.upper()} format...")
        save_start = time.time()
        
        from core.result_writer import save_results_as_npz, save_results_as_csv
        
        if output_format == 'npz':
            save_results_as_npz(results)
        else:
            save_results_as_csv(results)
        
        save_time = time.time() - save_start
        print(f"✅ Results saved in {save_time:.1f}s")
        
        # Automatic results visualization
        try:
            print("\n📊 Creating automatic results visualization...")
            
            # Import the plotting function
            sys.path.append(str(Path(__file__).parent.parent / "tools" / "plotting"))
            from show_results import create_automatic_plots
            
            # Create plots automatically (quiet mode, save figures)
            plot_success = create_automatic_plots(
                output_dir="OUT", 
                format_type=output_format,
                save_figures=True,
                quiet=True
            )
            
            if plot_success:
                print("✅ Results visualization created successfully!")
                print("📁 Plots saved to: OUT/plots/")
            else:
                print("⚠️ Results visualization had some issues, but simulation completed")
                
        except Exception as plot_error:
            print(f"⚠️ Could not create automatic plots: {plot_error}")
            print("💡 You can manually create plots with: python tools/plotting/show_results.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(main())
    """Save results in NPZ format."""
    try:
        print("🔍 Debug: Starting NPZ save...")
        print(f"🔍 Debug: Results keys: {list(results.keys())}")
        
        from pathlib import Path
        output_dir = Path("OUT")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_dict = {}
        
        # Handle time data
        print("🔍 Debug: Processing time data...")
        if 'time' in results and results['time'] is not None:
            time_data = results['time']
            print(f"🔍 Debug: Time data type: {type(time_data)}, shape: {getattr(time_data, 'shape', 'no shape')}")
            # Convert to numpy array first to avoid JAX array boolean issues
            time_array = np.array(time_data)
            if time_array.size > 0:
                save_dict['time'] = time_array
                print(f"🔍 Debug: Time array saved, shape: {save_dict['time'].shape}")
            else:
                save_dict['time'] = np.array([0])
        else:
            print("🔍 Debug: No time data found, using fallback...")
            # Create a simple time array if missing
            if 'hydro' in results and results['hydro'] and 'time' in results['hydro']:
                time_data = results['hydro']['time']
                save_dict['time'] = np.array(time_data)
            elif len(results) > 0:
                # Fallback: create time array based on data length
                save_dict['time'] = np.arange(1000)  # Default reasonable size
            else:
                save_dict['time'] = np.array([0])

        # Save hydro data from nested structure
        print("🔍 Debug: Processing hydro data...")
        if 'hydro' in results and results['hydro']:
            for key in ['H', 'U', 'D', 'PROF']:
                if key in results['hydro'] and results['hydro'][key] is not None:
                    hydro_array = np.array(results['hydro'][key])
                    if hydro_array.size > 0:  # Check if array has data
                        save_dict[key] = hydro_array
                        print(f"🔍 Debug: Hydro {key} saved, shape: {hydro_array.shape}")

        # Save species data from nested transport structure  
        print("🔍 Debug: Processing transport data...")
        if 'transport' in results and results['transport']:
            species_names = ['PHY1', 'PHY2', 'SI', 'NO3', 'NH4', 'PO4', 'PIP', 
                             'O2', 'TOC', 'S', 'SPM', 'DIC', 'AT', 'HS', 'PH', 'ALKC', 'CO2']
            
            for species in species_names:
                if species in results['transport'] and results['transport'][species] is not None:
                    species_array = np.array(results['transport'][species])
                    if species_array.size > 0:  # Check if array has data
                        save_dict[species] = species_array
                        print(f"🔍 Debug: Species {species} saved, shape: {species_array.shape}")
        
        print(f"🔍 Debug: About to save {len(save_dict)} datasets...")
        np.savez_compressed(output_dir / "simulation_results.npz", **save_dict)
        print(f"💾 Results saved to {output_dir}/simulation_results.npz ({len(save_dict)} datasets)")
        
    except Exception as e:
        print(f"⚠️  NPZ saving failed: {e}")
        print(f"🔍 Debug: Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

def save_results_csv(results):
    """Save results in CSV format."""
    try:
        from pathlib import Path
        output_dir = Path("OUT")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if results has time data
        if 'time' not in results or results['time'] is None:
            print("⚠️  No time data in results - saving available data only")
            # Try to get time from hydro structure
            if 'hydro' in results and results['hydro'] and 'time' in results['hydro']:
                time_array = np.array(results['hydro']['time'])
            elif len(results) > 0:
                # Create a simple time array based on available data length
                time_array = np.arange(1000)  # Default reasonable size
            else:
                time_array = np.array([0])
        else:
            time_array = np.array(results['time'])
        
        # Save key species from transport structure  
        key_species = ['S', 'O2', 'NO3', 'NH4', 'PO4']
        saved_count = 0
        
        # Check if we have transport data structure
        if 'transport' in results and results['transport']:
            for species in key_species:
                if species in results['transport'] and results['transport'][species] is not None:
                    species_array = np.array(results['transport'][species])
                    if species_array.size > 0:
                        filename = output_dir / f"{species}.csv"
                        
                        with open(filename, 'w') as f:
                            # Write header
                            M = species_array.shape[1] if len(species_array.shape) > 1 else len(species_array)
                            f.write("Time," + ",".join([f"X{i}" for i in range(M)]) + "\n")
                            
                            # Write data
                            for t_idx, time_val in enumerate(time_array):
                                if t_idx < len(species_array):
                                    if len(species_array.shape) > 1:
                                        row_data = species_array[t_idx]
                                    else:
                                        row_data = [species_array[t_idx]]
                                    f.write(f"{time_val}," + ",".join([f"{val:.6f}" for val in row_data]) + "\n")
                        saved_count += 1
        else:
            # Fallback: check for species directly in results
            for species in key_species:
                if species in results and results[species] is not None:
                    species_array = np.array(results[species])
                    if species_array.size > 0:
                        filename = output_dir / f"{species}.csv"
                        
                        with open(filename, 'w') as f:
                            # Write header
                            M = species_array.shape[1] if len(species_array.shape) > 1 else len(species_array)
                            f.write("Time," + ",".join([f"X{i}" for i in range(M)]) + "\n")
                            
                            # Write data  
                            for t_idx, time_val in enumerate(time_array):
                                if t_idx < len(species_array):
                                    if len(species_array.shape) > 1:
                                        row_data = species_array[t_idx]
                                    else:
                                        row_data = [species_array[t_idx]]
                                    f.write(f"{time_val}," + ",".join([f"{val:.6f}" for val in row_data]) + "\n")
                        saved_count += 1
        
        print(f"💾 CSV results saved to {output_dir}/ ({saved_count} species files)")
        
    except Exception as e:
        print(f"⚠️  CSV saving failed: {e}")
        raise

if __name__ == "__main__":
    sys.exit(main())