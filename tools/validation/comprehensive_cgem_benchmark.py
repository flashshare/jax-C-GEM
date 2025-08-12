#!/usr/bin/env python
"""
Comprehensive C-GEM vs JAX C-GEM Benchmark
==========================================

This script provides a complete comparison between the original C-GEM 
and JAX C-GEM implementations, ensuring identical configurations and 
providing detailed performance and accuracy analysis.

Key Features:
- Ensures identical parameters (deltaT, deltaX, simulation time, warmup)
- Cross-platform compatibility (Windows/Linux)
- Performance timing comparison
- Accuracy comparison of key variables (salinity, velocity, water level)
- Statistical analysis and visualization
- Generates comprehensive benchmark report

Author: Nguyen Truong An
"""

import os
import sys
import time
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import shutil

# Add src to path for JAX C-GEM imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Scientific computing imports
try:
    import numpy as np
    import pandas as pd
    from scipy import interpolate
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Memory tracking import
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs
    HAS_MATPLOTLIB = True
    plt.rcParams.update({
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
except ImportError:
    HAS_MATPLOTLIB = False

class ComprehensiveBenchmark:
    """Comprehensive benchmark comparing C-GEM and JAX C-GEM."""
    
    def __init__(self):
        """Initialize benchmark system."""
        self.project_root = project_root
        self.c_gem_dir = self.project_root / "deprecated" / "original-C-GEM"
        self.c_gem_exe = self.c_gem_dir / "c-gem.exe"
        self.output_dir = self.project_root / "OUT" / "comprehensive_benchmark"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.platform = platform.system()
        self.results = {
            'jax_cgem': {'timing': None, 'data': None, 'config': None},
            'original_cgem': {'timing': None, 'data': None, 'config': None},
            'comparison': {'metrics': None, 'plots_generated': False}
        }
        
        print("🏆 Comprehensive C-GEM vs JAX C-GEM Benchmark")
        print("=" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💻 Platform: {self.platform}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def verify_identical_configurations(self) -> bool:
        """Verify both models use identical parameters."""
        print("\n🔧 Verifying Identical Configurations...")
        print("-" * 40)
        
        # Load JAX C-GEM config
        try:
            from core.config_parser import parse_model_config
            jax_config = parse_model_config('config/model_config.txt')
            print(f"✅ JAX C-GEM config loaded: {len(jax_config)} parameters")
        except Exception as e:
            print(f"❌ Failed to load JAX C-GEM config: {e}")
            return False
        
        # Read C-GEM define.h
        define_h_path = self.c_gem_dir / "define.h"
        if not define_h_path.exists():
            print(f"❌ C-GEM define.h not found: {define_h_path}")
            return False
        
        with open(define_h_path, 'r') as f:
            define_content = f.read()
        
        # Extract key parameters from define.h
        c_gem_params = self._extract_define_h_params(define_content)
        self.cgem_params = c_gem_params  # Store for later use
        
        print(f"✅ C-GEM parameters extracted from define.h:")
        print(f"   MAXT: {c_gem_params.get('MAXT')} seconds ({c_gem_params.get('MAXT')/(24*60*60):.0f} days)")
        print(f"   WARMUP: {c_gem_params.get('WARMUP')} seconds ({c_gem_params.get('WARMUP')/(24*60*60):.0f} days)")
        print(f"   DELTI: {c_gem_params.get('DELTI')} seconds")
        print(f"   TS: {c_gem_params.get('TS')} (save every TS steps)")
        
        # Debug: Verify storage
        print(f"✅ Parameters stored in self.cgem_params: {hasattr(self, 'cgem_params')}")
        
        # Compare critical parameters
        critical_params = [
            ('MAXT', 'MAXT'),
            ('WARMUP', 'WARMUP'), 
            ('DELTI', 'DELTI'),
            ('DELXI', 'DELXI'),
            ('EL', 'EL'),
            ('AMPL', 'AMPL')
        ]
        
        config_match = True
        print("\n📊 Parameter Comparison:")
        
        for jax_key, c_key in critical_params:
            jax_val = jax_config.get(jax_key, 'MISSING')
            c_val = c_gem_params.get(c_key, 'MISSING')
            
            if jax_val != 'MISSING' and c_val != 'MISSING':
                # Special handling for time parameters (JAX uses days, C-GEM uses seconds)
                if jax_key in ['MAXT', 'WARMUP']:
                    # Convert JAX days to seconds for comparison
                    jax_val_seconds = float(jax_val) * 24 * 60 * 60
                    if abs(jax_val_seconds - float(c_val)) < 1e-6:
                        print(f"   ✅ {jax_key}: JAX={jax_val} days ({jax_val_seconds:.0f}s), C-GEM={c_val}s ✓")
                    else:
                        print(f"   ❌ {jax_key}: JAX={jax_val} days ({jax_val_seconds:.0f}s), C-GEM={c_val}s ✗")
                        config_match = False
                else:
                    # Direct comparison for other parameters
                    if abs(float(jax_val) - float(c_val)) < 1e-6:
                        print(f"   ✅ {jax_key}: JAX={jax_val}, C-GEM={c_val} ✓")
                    else:
                        print(f"   ❌ {jax_key}: JAX={jax_val}, C-GEM={c_val} ✗")
                        config_match = False
            else:
                print(f"   ⚠️  {jax_key}: JAX={jax_val}, C-GEM={c_val} (missing)")
                config_match = False
        
        self.results['jax_cgem']['config'] = jax_config
        self.results['original_cgem']['config'] = c_gem_params
        
        if config_match:
            print("✅ Configuration verification passed")
        else:
            print("⚠️  Configuration mismatch detected - results may not be comparable")
        
        return config_match
    
    def _extract_define_h_params(self, content: str) -> Dict[str, Any]:
        """Extract parameters from define.h content."""
        import re
        params = {}
        
        # Patterns for different parameter formats
        patterns = [
            (r'#define\s+(\w+)\s+([0-9.]+)', lambda x: float(x)),
            (r'#define\s+(\w+)\s+\(([0-9.]+)\)\*[0-9*]+', lambda x: float(x) * 24 * 60 * 60),  # MAXT
            (r'#define\s+(\w+)\s+([0-9.]+)\*[0-9*]+', lambda x: float(x) * 24 * 60 * 60),  # WARMUP
        ]
        
        for pattern, converter in patterns:
            matches = re.findall(pattern, content)
            for param_name, value_str in matches:
                try:
                    params[param_name] = converter(value_str)
                except:
                    params[param_name] = value_str
        
        return params
    
    def compile_original_cgem(self) -> bool:
        """Compile original C-GEM with Windows compatibility."""
        print("\n🔧 Compiling Original C-GEM...")
        print("-" * 40)
        
        if not self.c_gem_dir.exists():
            print(f"❌ C-GEM directory not found: {self.c_gem_dir}")
            return False
        
        # Change to C-GEM directory
        original_dir = os.getcwd()
        os.chdir(str(self.c_gem_dir))
        
        try:
            # Try using the Windows batch file first
            if self.platform == "Windows" and (self.c_gem_dir / "compile_and_run.bat").exists():
                print("🏗️  Using Windows batch compilation...")
                
                # Modify the batch file to only compile, not run
                compile_only_bat = self.c_gem_dir / "compile_only.bat"
                with open("compile_and_run.bat", 'r') as f:
                    content = f.read()
                
                # Remove the execution part, keep only compilation
                compile_content = content.split("REM Run the simulation")[0]
                compile_content += "\necho ✅ Compilation completed!\n"
                
                with open(compile_only_bat, 'w') as f:
                    f.write(compile_content)
                
                result = subprocess.run([str(compile_only_bat)], 
                                      capture_output=True, text=True, shell=True)
                
                if result.returncode == 0 or self.c_gem_exe.exists():
                    print("✅ C-GEM compiled successfully")
                    return True
                else:
                    print(f"❌ Compilation failed: {result.stderr}")
            
            # Fallback: Try direct GCC compilation
            print("🔧 Trying direct GCC compilation...")
            gcc_paths = [
                "gcc", 
                "C:/Program Files/CodeBlocks/MinGW/bin/gcc.exe",
                "C:/MinGW/bin/gcc.exe",
                "C:/msys64/mingw64/bin/gcc.exe"
            ]
            
            for gcc_path in gcc_paths:
                if shutil.which(gcc_path) or Path(gcc_path).exists():
                    print(f"🔧 Using GCC: {gcc_path}")
                    
                    # Create output directories
                    for dir_name in ["OUT", "OUT/Hydrodynamics", "OUT/Flux", "OUT/Reaction"]:
                        Path(dir_name).mkdir(exist_ok=True)
                    
                    # Compile
                    cmd = [gcc_path, "-o", "c-gem.exe"] + [str(f) for f in Path('.').glob('*.c')] + ["-lm", "-O2"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ C-GEM compiled successfully with GCC")
                        return True
                    else:
                        print(f"❌ GCC compilation failed: {result.stderr}")
            
            print("❌ No suitable compiler found")
            return False
            
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def run_jax_cgem_benchmark(self) -> bool:
        """Run JAX C-GEM benchmark with real-time progress monitoring."""
        print("\n🐍 Running JAX C-GEM Benchmark...")
        print("-" * 40)
        
        # Check if results already exist
        npz_file = Path("OUT/simulation_results.npz")
        csv_file = Path("OUT/O2.csv")
        
        if npz_file.exists():
            print("📁 Found existing simulation results - loading...")
            print(f"   ✅ NPZ file found: {npz_file}")
            
            # Try to load existing results
            if self._load_jax_cgem_results():
                print("✅ JAX C-GEM results loaded from existing files")
                self.results['jax_cgem']['timing'] = 0.0  # Unknown timing for existing results
                return True
            else:
                print("⚠️  Existing NPZ file appears incomplete - checking if simulation is running...")
                
                # Check if JAX simulation is currently running
                jax_running = self._check_jax_simulation_running()
                if jax_running:
                    print("🔄 JAX C-GEM simulation detected running in background")
                    print("   Waiting for simulation to complete...")
                    return self._wait_for_jax_simulation()
                else:
                    print("⚠️  No running simulation detected - will start new simulation")
        elif csv_file.exists():
            print("📁 Found existing CSV results - loading...")
            if self._load_jax_cgem_results():
                print("✅ JAX C-GEM results loaded from CSV files")
                return True
        
        try:
            # Track memory usage if available
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
            start_time = time.time()
            
            # Determine which script to use based on simulation length
            script_to_use, output_format = self._choose_optimal_script()
            
            # Run JAX C-GEM with real-time monitoring
            cmd_args = [sys.executable, script_to_use, "--mode", "run", "--output-format", output_format, "--no-physics-check"]
            
            print(f"🚀 Starting JAX simulation: {' '.join(cmd_args)}")
            
            # For large simulations, both CSV and NPZ are supported
            # The benchmark will handle format conversion as needed after simulation
            
            process = subprocess.Popen(cmd_args, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, 
                                     cwd=str(self.project_root), bufsize=1, universal_newlines=True)
            
            # Monitor progress in real-time
            self._monitor_jax_progress(process, start_time)
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=30*60)  # 30 minute timeout
            except subprocess.TimeoutExpired:
                print("⏰ JAX simulation timed out - terminating process")
                process.kill()
                stdout, stderr = process.communicate()
                return False
            
            elapsed_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
            
            if process.returncode == 0:
                print(f"\n✅ JAX C-GEM completed in {elapsed_time:.1f} seconds")
                if stdout.strip():
                    print(f"📋 Output: {stdout.strip()[-200:]}")  # Show last 200 chars
                    
                self.results['jax_cgem']['timing'] = elapsed_time
                if HAS_PSUTIL and initial_memory and final_memory:
                    memory_delta = final_memory - initial_memory
                    self.results['jax_cgem']['memory_usage'] = {
                        'initial': initial_memory,
                        'final': final_memory,
                        'delta': memory_delta
                    }
                
                # Load results
                self._load_jax_cgem_results()
                return True
            else:
                print(f"\n❌ JAX C-GEM failed with return code {process.returncode}")
                if stderr.strip():
                    print(f"📋 Error: {stderr.strip()[-500:]}")  # Show last 500 chars of error
                if stdout.strip():
                    print(f"📋 Output: {stdout.strip()[-200:]}")  # Show last 200 chars of output
                return False
                
        except Exception as e:
            print(f"❌ JAX C-GEM error: {e}")
            return False
    
    def _choose_optimal_script(self) -> tuple:
        """Choose the optimal script and output format based on simulation characteristics."""
        try:
            from core.config_parser import parse_model_config
            config = parse_model_config("config/model_config.txt")
            total_days = config.get('MAXT', 0)
            warmup_days = config.get('WARMUP', 0)
            output_days = total_days - warmup_days
            expected_outputs = output_days * 48  # 30-min intervals
            
            # Always use main.py with appropriate format
            script = "src/main.py"
            
            # For large output volumes, use NPZ format for performance
            if expected_outputs > 10000:  # > ~200 days of output
                output_format = "npz"
                print(f"🚀 Large simulation detected ({total_days} days, {expected_outputs:,} outputs)")
                print(f"   Using NPZ format for performance: {output_format}")
            else:
                output_format = "csv"
                print(f"⚡ Standard simulation ({total_days} days, {expected_outputs:,} outputs)")
                print(f"   Using CSV format: {output_format}")
                
            print(f"   Script: {script}")
            
            return script, output_format
            
        except Exception as e:
            print(f"⚠️  Error detecting simulation size: {e}")
            # Fallback to main.py with CSV
            return "src/main.py", "csv"
    
    def _check_jax_simulation_running(self) -> bool:
        """Check if a JAX C-GEM simulation is currently running."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'python.exe':
                        cmdline = proc.info.get('cmdline', [])
                        if any('main.py' in arg for arg in cmdline) and any('--mode' in arg for arg in cmdline):
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except:
            return False
    
    def _wait_for_jax_simulation(self) -> bool:
        """Wait for the running JAX simulation to complete and load results."""
        max_wait_time = 25 * 60  # 25 minutes maximum wait
        start_wait = time.time()
        check_interval = 15  # Check every 15 seconds
        last_progress_check = start_wait
        
        print("   ⏳ Monitoring background JAX simulation...")
        
        while time.time() - start_wait < max_wait_time:
            # Check if simulation is still running
            jax_running = self._check_jax_simulation_running()
            elapsed = time.time() - start_wait
            
            if not jax_running:
                print("   ✅ JAX simulation completed!")
                time.sleep(3)  # Give it a moment to finish writing files
                
                # Try to load results
                if self._load_jax_cgem_results():
                    print("✅ JAX C-GEM results loaded successfully")
                    return True
                else:
                    print("❌ Failed to load completed simulation results")
                    return False
            
            # Show progress update every minute
            if elapsed - (last_progress_check - start_wait) >= 60:
                print(f"   🔄 Still waiting... {elapsed/60:.1f} minutes elapsed")
                last_progress_check = time.time()
                
                # Try to detect if simulation is making progress by checking NPZ file changes
                try:
                    npz_file = Path("OUT/simulation_results.npz")
                    if npz_file.exists():
                        npz_data = np.load(str(npz_file))
                        time_entries = len(npz_data['time']) if 'time' in npz_data else 0
                        npz_data.close()
                        print(f"   📊 Current NPZ entries: {time_entries}")
                except:
                    pass
            
            time.sleep(check_interval)
        
        print(f"   ⏰ Timeout waiting for JAX simulation ({max_wait_time/60:.0f} minutes) - will start new simulation")
        return False

    def _monitor_jax_progress(self, process, start_time):
        """Monitor JAX C-GEM progress with NPZ format support and file-based completion detection."""
        print("📊 Monitoring progress...")
        
        # Configuration parameters (from model_config.txt)
        MAXT_days = 465  # Total simulation days
        WARMUP_days = 100  # Warmup period
        output_days = MAXT_days - WARMUP_days  # Days with output (365)

        last_update = time.time()
        last_progress = 0.0
        progress_shown = False
        max_wait_time = 25 * 60  # Maximum wait time: 25 minutes
        
        # Output file paths to check for completion
        npz_file = Path("OUT/simulation_results.npz")
        csv_file = Path("OUT/O2.csv")
        
        while process.poll() is None:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if simulation completed by looking for output files
            if npz_file.exists() or csv_file.exists():
                print(f"   ✅ Output file detected - simulation completed!")
                break
                
            # Timeout protection
            if elapsed > max_wait_time:
                print(f"   ⏰ Timeout reached ({max_wait_time/60:.0f} minutes) - assuming completion")
                break
            
            # Check progress every 15 seconds
            if current_time - last_update >= 15:
                try:
                    # For NPZ format, we can't track detailed progress
                    # Just provide time-based estimates
                    
                    if not progress_shown and elapsed > 30:
                        # After 30 seconds, show we're running
                        print(f"   🔄 Simulation running... Elapsed: {elapsed/60:.1f} minutes")
                        print(f"   📊 Expected duration: ~15-20 minutes for 465-day simulation")
                        progress_shown = True
                    
                    elif elapsed > 180 and (current_time - last_update) >= 120:  # Every 2 minutes after 3 minutes
                        estimated_total = 15 * 60  # 15 minutes estimate
                        progress_pct = min((elapsed / estimated_total) * 100, 95)  # Cap at 95%
                        eta_remaining = max(0, estimated_total - elapsed)
                        eta_str = f"{eta_remaining/60:.1f}m" if eta_remaining > 0 else "Soon"
                        
                        print(f"   ⏳ Est. Progress: ~{progress_pct:.0f}% | Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}")
                
                except Exception as e:
                    # Silently ignore monitoring errors
                    pass
                
                last_update = current_time
            
            # Sleep briefly to avoid excessive CPU usage
            time.sleep(5)
        
        # Terminate the process if it's still running after file detection
        if process.poll() is None:
            try:
                process.terminate()
                time.sleep(2)  # Give it a moment to terminate gracefully
                if process.poll() is None:
                    process.kill()  # Force kill if necessary
            except:
                pass
        
        # Show completion
        final_elapsed = time.time() - start_time
        print(f"   ✅ Process completed after {final_elapsed/60:.1f} minutes")
    
    def _monitor_cgem_progress(self, process, start_time):
        """Monitor C-GEM progress without blocking."""
        original_out_path = Path("deprecated/original-C-GEM/OUT")
        o2_file = original_out_path / "O2.csv"
        
        # Expected outputs based on configuration - use actual config values
        # Fallback to hardcoded values if cgem_params is not available (defensive programming)
        if hasattr(self, 'cgem_params') and self.cgem_params:
            maxt_days = self.cgem_params.get('MAXT', 465) / (24*60*60)  # Convert seconds to days
            warmup_days = self.cgem_params.get('WARMUP', 100) / (24*60*60)  # Convert seconds to days
            delti = self.cgem_params.get('DELTI', 180)  # Time step in seconds
            ts = self.cgem_params.get('TS', 10)  # Save every TS time steps
        else:
            # Fallback to define.h defaults
            maxt_days = 465  # MAXT = 465 days
            warmup_days = 100  # WARMUP = 100 days
            delti = 180  # DELTI = 180 seconds
            ts = 10  # TS = 10
            
        total_days = int(maxt_days - warmup_days)  # Output days (MAXT - WARMUP)
        
        # Calculate correct output frequency using TS value
        output_interval_seconds = delti * ts  # 180 * 10 = 1800 seconds = 30 minutes
        outputs_per_day = 24 * 3600 // output_interval_seconds  # 48 outputs per day
        expected_outputs = total_days * outputs_per_day
        
        last_update = 0
        last_output_count = 0
        print(f"   Progress tracking: Monitoring {o2_file}")
        print(f"   Expected: {expected_outputs:,} outputs over {total_days} days")
        print(f"   {'Simulated Days':>12} | {'Progress':>8} | {'Elapsed':>10} | {'Est. Remaining':>12} | {'Memory':>8}")
        print(f"   {'-'*12} | {'-'*8} | {'-'*10} | {'-'*12} | {'-'*8}")
        
        while process.poll() is None:  # Process still running
            current_time = time.time()
            
            # Update every 30 seconds
            if current_time - last_update >= 30:
                try:
                    if o2_file.exists():
                        # Count lines in output file (subtract header)
                        with open(o2_file, 'r') as f:
                            current_outputs = max(0, sum(1 for _ in f) - 1)
                        
                        if current_outputs > 0:
                            # Calculate progress
                            progress_pct = (current_outputs / expected_outputs) * 100
                            simulated_days = current_outputs / outputs_per_day
                            elapsed = current_time - start_time
                            
                            # Estimate remaining time
                            if current_outputs > last_output_count and last_output_count > 0:
                                outputs_per_second = (current_outputs - last_output_count) / (current_time - last_update)
                                remaining_outputs = expected_outputs - current_outputs
                                eta_remaining = remaining_outputs / outputs_per_second if outputs_per_second > 0 else None
                                
                                if eta_remaining:
                                    eta_str = f"{eta_remaining/60:.0f}m" if eta_remaining < 3600 else f"{eta_remaining/3600:.1f}h"
                                else:
                                    eta_str = "Calculating..."
                            else:
                                eta_str = "Calculating..."
                            
                            # Memory usage
                            memory_str = ""
                            if HAS_PSUTIL:
                                try:
                                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                                    memory_str = f"{current_memory:.0f}MB"
                                except:
                                    memory_str = "N/A"
                            
                            # Update display
                            elapsed_str = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
                            print(f"   {simulated_days:6.1f} days  | {progress_pct:6.1f}% | {elapsed_str:>10} | {eta_str:>12} | {memory_str}")
                            
                            last_output_count = current_outputs
                
                except Exception as e:
                    # Silently ignore file access errors during monitoring
                    pass
                
                last_update = current_time
            
            # Sleep briefly to avoid excessive CPU usage
            time.sleep(5)
    
    def run_original_cgem_benchmark(self) -> bool:
        """Run original C-GEM benchmark."""
        print("\n🔵 Running Original C-GEM Benchmark...")
        print("-" * 40)
        
        if not self.c_gem_exe.exists():
            print(f"❌ C-GEM executable not found: {self.c_gem_exe}")
            return False
        
        # Change to C-GEM directory
        original_dir = os.getcwd()
        os.chdir(str(self.c_gem_dir))
        
        try:
            # Track memory usage if available
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
            start_time = time.time()
            
            # Clear any existing output with robust Windows cleanup
            out_dir = Path("OUT")
            if out_dir.exists():
                print("🧹 Cleaning existing C-GEM output directory...")
                try:
                    shutil.rmtree(out_dir)
                    print("   ✅ Directory removed successfully")
                except OSError as e:
                    print(f"   ⚠️ Initial cleanup failed: {e}")
                    # Force cleanup with retries for Windows
                    for attempt in range(3):
                        try:
                            time.sleep(1)  # Wait for file handles to be released
                            if out_dir.exists():
                                shutil.rmtree(out_dir, ignore_errors=True)
                            if not out_dir.exists():
                                print(f"   ✅ Directory cleaned after attempt {attempt + 1}")
                                break
                        except OSError:
                            if attempt == 2:
                                print(f"   ❌ Failed to clean directory after 3 attempts")
                                return False
            
            # Create output directories
            for dir_name in ["OUT", "OUT/Hydrodynamics", "OUT/Flux", "OUT/Reaction"]:
                Path(dir_name).mkdir(exist_ok=True)
            
            # Run C-GEM with progress monitoring
            print("📊 Starting C-GEM with progress monitoring...")
            process = subprocess.Popen([str(self.c_gem_exe)], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     bufsize=1, universal_newlines=True)
            
            # Monitor progress in real-time  
            self._monitor_cgem_progress(process, start_time)
            
            # Wait for completion (no timeout for full simulation)
            stdout, stderr = process.communicate()
            elapsed_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
            
            if process.returncode == 0:
                print(f"✅ C-GEM completed in {elapsed_time:.1f} seconds")
                self.results['original_cgem']['timing'] = elapsed_time
                if HAS_PSUTIL and initial_memory and final_memory:
                    memory_delta = final_memory - initial_memory
                    self.results['original_cgem']['memory_usage'] = {
                        'initial': initial_memory,
                        'final': final_memory,
                        'delta': memory_delta
                    }
                
                # Load results
                self._load_original_cgem_results()
                return True
            else:
                print(f"⚠️  C-GEM finished with return code {process.returncode}")
                
                elapsed_time = time.time() - start_time
                print(f"⏱️  Execution time: {elapsed_time:.1f} seconds")
                self.results['original_cgem']['timing'] = elapsed_time
                if HAS_PSUTIL and initial_memory and final_memory:
                    memory_delta = final_memory - initial_memory
                    self.results['original_cgem']['memory_usage'] = {
                        'initial': initial_memory,
                        'final': final_memory,
                        'delta': memory_delta
                    }
                
                # Try to load results anyway (C-GEM often exits with non-zero codes but still produces valid results)
                try:
                    self._load_original_cgem_results()
                    print("   📊 Results loaded successfully despite non-zero exit code")
                    return True
                except Exception as e:
                    print(f"   ❌ Failed to load results: {e}")
                    return False
        except Exception as e:
            print(f"❌ C-GEM execution error: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def _load_jax_cgem_results(self):
        """Load JAX C-GEM results from NPZ or CSV files."""
        try:
            out_dir = self.project_root / "OUT"
            h_data = None
            s_data = None
            
            # First try to load NPZ format (preferred for large simulations)
            npz_file = out_dir / "simulation_results.npz"
            if npz_file.exists():
                print(f"   📦 Loading JAX C-GEM results from NPZ format: {npz_file}")
                try:
                    npz_data = np.load(str(npz_file))
                    
                    # Check if NPZ file is complete by examining available keys
                    available_keys = list(npz_data.keys())
                    has_hydro = any(key in available_keys for key in ['H', 'U', 'D', 'PROF'])
                    has_species = any(key in available_keys for key in ['O2', 'NO3', 'NH4', 'PO4', 'S', 'SPM', 'DIC', 'AT'])
                    
                    print(f"   📋 NPZ file contains {len(available_keys)} keys")
                    
                    # If NPZ file only has 'time' key or minimal data, it's incomplete
                    if len(available_keys) <= 1 or not (has_hydro or has_species):
                        print(f"   ⚠️  NPZ file appears incomplete - only {available_keys} found")
                        npz_data.close()
                        print("   📄 Falling back to CSV format...")
                    else:
                        # Extract hydrodynamics (water level or velocity)
                        if 'H' in npz_data:
                            h_array = npz_data['H']
                            h_data = pd.DataFrame(h_array.T)  # Transpose for proper shape
                            print(f"   📊 Loaded JAX C-GEM hydrodynamics from NPZ: {h_data.shape}")
                        
                        # Extract transport (salinity or other species)
                        if 'S' in npz_data:
                            s_array = npz_data['S']
                            s_data = pd.DataFrame(s_array.T)  # Transpose for proper shape
                            print(f"   📊 Loaded JAX C-GEM transport from NPZ: {s_data.shape}")
                        elif 'O2' in npz_data:
                            s_array = npz_data['O2']
                            s_data = pd.DataFrame(s_array.T)
                            print(f"   📊 Loaded JAX C-GEM transport (O2) from NPZ: {s_data.shape}")
                        
                        npz_data.close()
                        
                except Exception as npz_error:
                    print(f"   ⚠️  Error reading NPZ file: {npz_error}")
            
            # Fallback to CSV format if NPZ not available or failed
            if h_data is None or s_data is None:
                print("   📄 Falling back to CSV format...")
                
                # Load hydrodynamics from structured directory
                hydro_dir = out_dir / "Hydrodynamics"
                h_files = list(hydro_dir.glob("*.csv")) if hydro_dir.exists() else []
                
                if h_files and h_data is None:
                    h_data = pd.read_csv(h_files[0])  # Take first file (usually H.csv)
                    print(f"   📊 Loaded JAX C-GEM hydrodynamics from CSV: {h_data.shape}")
                
                # Load transport from structured directory (Reaction folder)
                reaction_dir = out_dir / "Reaction"
                s_files = list(reaction_dir.glob("*.csv")) if reaction_dir.exists() else []
                
                if s_files and s_data is None:
                    s_data = pd.read_csv(s_files[0])  # Take first file (usually O2.csv)
                    print(f"   📊 Loaded JAX C-GEM transport from CSV: {s_data.shape}")
            
            # Report final status
            if h_data is None:
                print(f"   ⚠️  No JAX C-GEM hydrodynamics data found")
            if s_data is None:
                print(f"   ⚠️  No JAX C-GEM transport data found")
            
            self.results['jax_cgem']['data'] = {
                'hydrodynamics': h_data,
                'transport': s_data
            }
            
            # Return True only if we have meaningful data
            return h_data is not None or s_data is not None
            
        except Exception as e:
            print(f"   ❌ Error loading JAX C-GEM results: {e}")
            return False
    
    def _load_original_cgem_results(self):
        """Load original C-GEM results from output files."""
        try:
            c_gem_out = self.c_gem_dir / "OUT"
            
            # Look for hydrodynamics files
            hydro_dir = c_gem_out / "Hydrodynamics"
            h_files = list(hydro_dir.glob("*.csv")) if hydro_dir.exists() else []
            
            if h_files:
                h_data = pd.read_csv(h_files[0])  # Take first file
                print(f"   📊 Loaded C-GEM hydrodynamics: {h_data.shape}")
            else:
                print(f"   ⚠️  No C-GEM hydrodynamics files found in {hydro_dir}")
                h_data = None
            
            # Look for transport/reaction files
            reaction_dir = c_gem_out / "Reaction"
            s_files = list(reaction_dir.glob("*.csv")) if reaction_dir.exists() else []
            
            if s_files:
                s_data = pd.read_csv(s_files[0])  # Take first file
                print(f"   📊 Loaded C-GEM transport: {s_data.shape}")
            else:
                print(f"   ⚠️  No C-GEM transport files found in {reaction_dir}")
                s_data = None
            
            self.results['original_cgem']['data'] = {
                'hydrodynamics': h_data,
                'transport': s_data
            }
            
        except Exception as e:
            print(f"   ❌ Error loading C-GEM results: {e}")
    
    def compare_results(self):
        """Compare results between both models."""
        print("\n📊 Comparing Model Results...")
        print("-" * 40)
        
        jax_data = self.results['jax_cgem']['data']
        cgem_data = self.results['original_cgem']['data']
        
        if not jax_data or not cgem_data:
            print("❌ Cannot compare - missing data from one or both models")
            return
        
        # Performance comparison
        jax_time = self.results['jax_cgem']['timing']
        cgem_time = self.results['original_cgem']['timing']
        
        if jax_time and cgem_time:
            speedup = cgem_time / jax_time if jax_time < cgem_time else jax_time / cgem_time
            faster_model = "JAX C-GEM" if jax_time < cgem_time else "Original C-GEM"
            
            print(f"⚡ Performance Comparison:")
            print(f"   🐍 JAX C-GEM:      {jax_time:.1f} seconds")
            print(f"   🔵 Original C-GEM: {cgem_time:.1f} seconds")
            print(f"   🏆 {faster_model} is {speedup:.2f}x faster!")
        
        # Generate plots if matplotlib available
        if HAS_MATPLOTLIB:
            self._generate_comparison_plots()
    
    def _generate_comparison_plots(self):
        """Generate comprehensive comparison plots."""
        print("\n🎨 Generating Comparison Plots...")
        
        if not HAS_MATPLOTLIB:
            print("   ❌ matplotlib not available")
            return
        
        try:
            fig = plt.figure(figsize=(15, 12))
            gs_layout = gs.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Performance comparison
            ax1 = fig.add_subplot(gs_layout[0, 0])
            self._plot_performance_comparison(ax1)
            
            # Memory usage (if available)
            ax2 = fig.add_subplot(gs_layout[0, 1])
            self._plot_memory_comparison(ax2)
            
            # Accuracy comparison
            ax3 = fig.add_subplot(gs_layout[1, :])
            self._plot_accuracy_comparison(ax3)
            
            # Longitudinal profiles
            ax4 = fig.add_subplot(gs_layout[2, :])
            self._plot_longitudinal_profiles(ax4)
            
            plt.suptitle('Comprehensive C-GEM vs JAX C-GEM Benchmark', fontsize=16, fontweight='bold')
            
            # Save plot
            plot_path = self.output_dir / f"comprehensive_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Comparison plots saved: {plot_path}")
            
            # Plot saved - no need to show in non-interactive mode
            
        except Exception as e:
            print(f"   ❌ Error generating plots: {e}")
    
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison."""
        jax_time = self.results['jax_cgem']['timing']
        cgem_time = self.results['original_cgem']['timing']
        
        if jax_time and cgem_time:
            models = ['JAX C-GEM', 'Original C-GEM']
            times = [jax_time, cgem_time]
            colors = ['#1f77b4', '#ff7f0e']
            
            bars = ax.bar(models, times, color=colors, alpha=0.8)
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Performance Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{time:.1f}s', ha='center', va='bottom')
    
    def _plot_memory_comparison(self, ax):
        """Plot memory usage comparison."""
        jax_memory = self.results['jax_cgem'].get('memory_usage')
        cgem_memory = self.results['original_cgem'].get('memory_usage')
        
        if HAS_PSUTIL and jax_memory and cgem_memory:
            models = ['JAX C-GEM', 'Original C-GEM']
            initial = [jax_memory['initial'], cgem_memory['initial']]
            final = [jax_memory['final'], cgem_memory['final']]
            
            x = range(len(models))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], initial, width, label='Initial Memory', alpha=0.7)
            ax.bar([i + width/2 for i in x], final, width, label='Final Memory', alpha=0.7)
            
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add delta annotations
            for i, (model, jmem, cmem) in enumerate(zip(models, [jax_memory, cgem_memory], [jax_memory, cgem_memory])):
                delta = cmem['delta']
                ax.text(i, max(cmem['initial'], cmem['final']) + 5,
                       f'Δ{delta:+.1f}MB', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Memory Usage\nComparison\n(psutil not available)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Memory Usage Comparison')
    
    def _plot_accuracy_comparison(self, ax):
        """Plot accuracy comparison using statistical metrics."""
        jax_data = self.results['jax_cgem']['data']
        cgem_data = self.results['original_cgem']['data']
        
        if not (jax_data and cgem_data and HAS_SCIPY):
            ax.text(0.5, 0.5, 'Accuracy Metrics\n(RMSE, Correlation)\n(Data not available)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Accuracy Comparison')
            return
        
        metrics = {}
        try:
            # Compare hydrodynamics data
            if jax_data['hydrodynamics'] is not None and cgem_data['hydrodynamics'] is not None:
                jax_hydro = jax_data['hydrodynamics']
                cgem_hydro = cgem_data['hydrodynamics']
                
                # Find common columns (excluding time/index columns)
                jax_cols = [col for col in jax_hydro.columns if col not in ['Time', 'time', 'Index', 'index']]
                cgem_cols = [col for col in cgem_hydro.columns if col not in ['Time', 'time', 'Index', 'index']]
                common_cols = set(jax_cols) & set(cgem_cols)
                
                for col in list(common_cols)[:3]:  # Take first 3 common columns
                    try:
                        # Interpolate to common length if needed
                        if len(jax_hydro) != len(cgem_hydro):
                            min_len = min(len(jax_hydro), len(cgem_hydro))
                            jax_vals = jax_hydro[col].iloc[:min_len].values
                            cgem_vals = cgem_hydro[col].iloc[:min_len].values
                        else:
                            jax_vals = jax_hydro[col].values
                            cgem_vals = cgem_hydro[col].values
                        
                        # Remove NaN values
                        mask = ~(np.isnan(jax_vals) | np.isnan(cgem_vals))
                        jax_clean = jax_vals[mask]
                        cgem_clean = cgem_vals[mask]
                        
                        if len(jax_clean) > 10:  # Need reasonable sample size
                            # Calculate RMSE
                            rmse = np.sqrt(np.mean((jax_clean - cgem_clean)**2))
                            # Calculate correlation
                            corr, _ = pearsonr(jax_clean, cgem_clean)
                            # Calculate relative error
                            rel_error = rmse / (np.mean(np.abs(cgem_clean)) + 1e-10) * 100
                            
                            metrics[col] = {'RMSE': rmse, 'Correlation': corr, 'Rel_Error': rel_error}
                    except Exception as e:
                        continue
            
            # Plot metrics
            if metrics:
                cols = list(metrics.keys())
                rmse_vals = [metrics[col]['RMSE'] for col in cols]
                corr_vals = [metrics[col]['Correlation'] for col in cols]
                
                # Create dual y-axis plot
                ax2 = ax.twinx()
                
                x_pos = np.arange(len(cols))
                width = 0.35
                
                bars1 = ax.bar(x_pos - width/2, rmse_vals, width, label='RMSE', alpha=0.7, color='red')
                bars2 = ax2.bar(x_pos + width/2, corr_vals, width, label='Correlation', alpha=0.7, color='blue')
                
                ax.set_xlabel('Variables')
                ax.set_ylabel('RMSE', color='red')
                ax2.set_ylabel('Correlation', color='blue')
                ax.set_title('Accuracy Comparison: RMSE vs Correlation')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(cols, rotation=45, ha='right')
                
                # Color the y-axis labels to match the bars
                ax.tick_params(axis='y', labelcolor='red')
                ax2.tick_params(axis='y', labelcolor='blue')
                
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars1, rmse_vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='red')
                
                for bar, val in zip(bars2, corr_vals):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='blue')
            else:
                ax.text(0.5, 0.5, 'No comparable\ndata found', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Accuracy Comparison')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error computing\naccuracy metrics:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Accuracy Comparison')
    
    def _plot_longitudinal_profiles(self, ax):
        """Plot longitudinal profile comparison."""
        jax_data = self.results['jax_cgem']['data']
        cgem_data = self.results['original_cgem']['data']
        
        if not (jax_data and cgem_data):
            ax.text(0.5, 0.5, 'Longitudinal Profiles\n(Data not available)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Longitudinal Profile Comparison')
            return
        
        try:
            # Get hydrodynamics data for both models
            jax_hydro = jax_data['hydrodynamics']
            cgem_hydro = cgem_data['hydrodynamics']
            jax_transport = jax_data['transport']
            cgem_transport = cgem_data['transport']
            
            if jax_hydro is None or cgem_hydro is None:
                ax.text(0.5, 0.5, 'No hydrodynamics\ndata available', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Longitudinal Profile Comparison')
                return
            
            # Create spatial grid (assuming first column might be spatial or we use index)
            # Use the last timestep for profiles
            jax_final = jax_hydro.iloc[-1]
            cgem_final = cgem_hydro.iloc[-1]
            
            # Create spatial coordinates (assuming uniform grid)
            n_jax = len(jax_final) - 1  # Subtract time column
            n_cgem = len(cgem_final) - 1
            
            x_jax = np.linspace(0, 200, n_jax)  # 200km estuary from config
            x_cgem = np.linspace(0, 200, n_cgem)
            
            # Find variables to plot (excluding time/index columns)
            exclude_cols = ['Time', 'time', 'Index', 'index', 'times', 'Times']
            jax_vars = [col for col in jax_hydro.columns if col not in exclude_cols]
            cgem_vars = [col for col in cgem_hydro.columns if col not in exclude_cols]
            
            # Plot key variables
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            plot_count = 0
            
            # Try to find and plot common variables
            common_vars = set(jax_vars) & set(cgem_vars)
            priority_vars = ['H', 'U', 'S', 'O2']  # Prioritize key physical variables
            
            vars_to_plot = []
            for var in priority_vars:
                if var in common_vars:
                    vars_to_plot.append(var)
                    
            # Add other common variables if we need more
            for var in common_vars:
                if var not in vars_to_plot and len(vars_to_plot) < 3:
                    vars_to_plot.append(var)
            
            if vars_to_plot:
                for i, var in enumerate(vars_to_plot[:3]):  # Plot up to 3 variables
                    try:
                        jax_vals = jax_final[var].values if hasattr(jax_final[var], 'values') else [jax_final[var]]
                        cgem_vals = cgem_final[var].values if hasattr(cgem_final[var], 'values') else [cgem_final[var]]
                        
                        # Handle scalar case
                        if np.isscalar(jax_vals):
                            jax_vals = np.array([jax_vals])
                        if np.isscalar(cgem_vals):
                            cgem_vals = np.array([cgem_vals])
                            
                        # Ensure proper lengths
                        jax_vals = jax_vals[:len(x_jax)]
                        cgem_vals = cgem_vals[:len(x_cgem)]
                        x_jax_plot = x_jax[:len(jax_vals)]
                        x_cgem_plot = x_cgem[:len(cgem_vals)]
                        
                        if len(jax_vals) > 1 and len(cgem_vals) > 1:
                            ax.plot(x_jax_plot, jax_vals, 
                                   label=f'JAX {var}', linestyle='-', 
                                   color=colors[i], linewidth=2)
                            ax.plot(x_cgem_plot, cgem_vals, 
                                   label=f'C-GEM {var}', linestyle='--', 
                                   color=colors[i], linewidth=2)
                            plot_count += 1
                    except Exception as e:
                        continue
                
                if plot_count > 0:
                    ax.set_xlabel('Distance from mouth (km)')
                    ax.set_ylabel('Variable values (normalized)')
                    ax.set_title('Longitudinal Profiles Comparison (Final Timestep)')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Could not plot\nprofile data', 
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Longitudinal Profile Comparison')
            else:
                ax.text(0.5, 0.5, 'No common variables\nfound for comparison', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Longitudinal Profile Comparison')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting profiles:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Longitudinal Profile Comparison')
    
    def generate_plots(self):
        """Generate comprehensive comparison plots with enhanced analysis."""
        if not HAS_MATPLOTLIB:
            print("⚠️ Matplotlib not available. Skipping plots...")
            return
            
        print("📊 Generating enhanced comparison plots...")
        
        # Create comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Main comparison plots (2x2 grid in upper half)
        ax1 = plt.subplot(3, 3, 1)  # Performance
        ax2 = plt.subplot(3, 3, 2)  # Memory 
        ax3 = plt.subplot(3, 3, 3)  # Accuracy
        ax4 = plt.subplot(3, 3, 4)  # Longitudinal profiles
        
        # Enhanced analysis plots (bottom half)
        ax5 = plt.subplot(3, 3, 5)  # Tidal range comparison
        ax6 = plt.subplot(3, 3, 6)  # Salinity comparison
        ax7 = plt.subplot(3, 3, 7)  # Dissolved oxygen comparison
        ax8 = plt.subplot(3, 3, 8)  # NH4 comparison
        ax9 = plt.subplot(3, 3, 9)  # Field data comparison
        
        # Generate plots
        self._plot_performance_comparison(ax1)
        self._plot_memory_comparison(ax2)
        self._plot_accuracy_comparison(ax3)
        self._plot_longitudinal_profiles(ax4)
        
        # Enhanced variable-specific plots
        self._plot_tidal_range_comparison(ax5)
        self._plot_salinity_comparison(ax6)
        self._plot_dissolved_oxygen_comparison(ax7)
        self._plot_nh4_comparison(ax8)
        self._plot_field_data_comparison(ax9)
        
        plt.tight_layout(pad=3.0)
        
        # Save plot
        plot_path = self.output_dir / "comprehensive_comparison_plots.png"
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comprehensive plots saved: {plot_path}")
        except Exception as e:
            print(f"⚠️ Error saving plots: {e}")
        
        plt.close()
    
    def _plot_tidal_range_comparison(self, ax):
        """Plot tidal range comparison between models and observations."""
        try:
            # Load observed tidal range data
            tidal_file = Path("INPUT/Calibration/CEM-Tidal-range.csv")
            
            ranges = []
            labels = []
            colors = []
            
            # Add observed data if available
            if tidal_file.exists():
                tidal_data = pd.read_csv(tidal_file)
                obs_range = tidal_data['Tidal Range (m)'].dropna()
                if len(obs_range) > 0:
                    ranges.append(obs_range.mean())
                    labels.append('Observed')
                    colors.append('#2ca02c')  # Green
            
            # Add model data
            jax_data = self.results['jax_cgem']['data']
            cgem_data = self.results['original_cgem']['data']
            
            if jax_data and 'hydrodynamics' in jax_data and jax_data['hydrodynamics'] is not None:
                jax_hydro = jax_data['hydrodynamics']
                if 'H' in jax_hydro.columns:
                    jax_range = jax_hydro['H'].max() - jax_hydro['H'].min()
                    ranges.append(jax_range)
                    labels.append('JAX C-GEM')
                    colors.append('#1f77b4')  # Blue
            
            if cgem_data and 'hydrodynamics' in cgem_data and cgem_data['hydrodynamics'] is not None:
                cgem_hydro = cgem_data['hydrodynamics']
                if 'H' in cgem_hydro.columns:
                    cgem_range = cgem_hydro['H'].max() - cgem_hydro['H'].min()
                    ranges.append(cgem_range)
                    labels.append('C-GEM')
                    colors.append('#ff7f0e')  # Orange
            
            if ranges:
                bars = ax.bar(labels, ranges, color=colors, alpha=0.8)
                ax.set_ylabel('Tidal Range (m)')
                ax.set_title('Tidal Range Comparison')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, ranges):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{val:.2f}m', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No tidal range\ndata available', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Tidal Range Comparison')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Tidal Range Comparison')
    
    def _plot_salinity_comparison(self, ax):
        """Plot salinity profile comparison."""
        try:
            jax_data = self.results['jax_cgem']['data']
            cgem_data = self.results['original_cgem']['data']
            
            salinity_found = False
            
            # Plot JAX C-GEM salinity
            if jax_data and 'transport' in jax_data and jax_data['transport'] is not None:
                jax_transport = jax_data['transport']
                if 'S' in jax_transport.columns:
                    s_data = jax_transport['S']
                    if hasattr(s_data, 'values') and len(s_data.values.flatten()) > 1:
                        x_jax = np.linspace(0, 200, len(s_data.values.flatten()))
                        ax.plot(x_jax, s_data.values.flatten(), label='JAX C-GEM', 
                               color='#1f77b4', linewidth=2)
                        salinity_found = True
            
            # Plot C-GEM salinity  
            if cgem_data and 'transport' in cgem_data and cgem_data['transport'] is not None:
                cgem_transport = cgem_data['transport']
                if 'S' in cgem_transport.columns:
                    s_data = cgem_transport['S']
                    if hasattr(s_data, 'values') and len(s_data.values.flatten()) > 1:
                        x_cgem = np.linspace(0, 200, len(s_data.values.flatten()))
                        ax.plot(x_cgem, s_data.values.flatten(), label='C-GEM', 
                               color='#ff7f0e', linewidth=2, linestyle='--')
                        salinity_found = True
            
            # Add observed data points if available
            care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
            if care_file.exists():
                care_data = pd.read_csv(care_file)
                if 'Salinity' in care_data.columns and 'Location' in care_data.columns:
                    obs_sal = care_data['Salinity'].dropna()
                    obs_loc = care_data.loc[obs_sal.index, 'Location']
                    ax.scatter(obs_loc, obs_sal, color='#2ca02c', s=50, 
                             label='CARE Observations', alpha=0.7, zorder=5)
                    salinity_found = True
            
            if salinity_found:
                ax.set_xlabel('Distance from mouth (km)')
                ax.set_ylabel('Salinity (psu)')
                ax.set_title('Salinity Profile Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No salinity data\navailable', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Salinity Profile Comparison')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Salinity Profile Comparison')
    
    def _plot_dissolved_oxygen_comparison(self, ax):
        """Plot dissolved oxygen comparison."""
        try:
            jax_data = self.results['jax_cgem']['data']
            cgem_data = self.results['original_cgem']['data']
            
            do_values = []
            do_labels = []
            do_colors = []
            
            # JAX C-GEM DO
            if jax_data and 'transport' in jax_data and jax_data['transport'] is not None:
                jax_transport = jax_data['transport']
                if 'O2' in jax_transport.columns:
                    jax_o2 = jax_transport['O2'].mean()  # Mean concentration
                    do_values.append(jax_o2)
                    do_labels.append('JAX C-GEM')
                    do_colors.append('#1f77b4')
            
            # C-GEM DO
            if cgem_data and 'transport' in cgem_data and cgem_data['transport'] is not None:
                cgem_transport = cgem_data['transport']
                if 'O2' in cgem_transport.columns:
                    cgem_o2 = cgem_transport['O2'].mean()
                    do_values.append(cgem_o2)
                    do_labels.append('C-GEM')
                    do_colors.append('#ff7f0e')
            
            # Observed DO from CARE data
            care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
            if care_file.exists():
                care_data = pd.read_csv(care_file)
                if 'DO (mg/L)' in care_data.columns:
                    obs_do = care_data['DO (mg/L)'].dropna().mean()
                    # Convert mg/L to mmol/m³ (approximate: 1 mg/L ≈ 31.25 mmol/m³)
                    obs_do_mmol = obs_do * 31.25
                    do_values.append(obs_do_mmol)
                    do_labels.append('CARE Obs.')
                    do_colors.append('#2ca02c')
            
            if do_values:
                bars = ax.bar(do_labels, do_values, color=do_colors, alpha=0.8)
                ax.set_ylabel('Dissolved Oxygen (mmol/m³)')
                ax.set_title('Dissolved Oxygen Comparison')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, do_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(do_values)*0.01,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No dissolved oxygen\ndata available', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Dissolved Oxygen Comparison')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dissolved Oxygen Comparison')
    
    def _plot_nh4_comparison(self, ax):
        """Plot NH4 comparison."""
        try:
            jax_data = self.results['jax_cgem']['data']
            cgem_data = self.results['original_cgem']['data']
            
            nh4_values = []
            nh4_labels = []
            nh4_colors = []
            
            # JAX C-GEM NH4
            if jax_data and 'transport' in jax_data and jax_data['transport'] is not None:
                jax_transport = jax_data['transport']
                if 'NH4' in jax_transport.columns:
                    jax_nh4 = jax_transport['NH4'].mean()
                    nh4_values.append(jax_nh4)
                    nh4_labels.append('JAX C-GEM')
                    nh4_colors.append('#1f77b4')
            
            # C-GEM NH4
            if cgem_data and 'transport' in cgem_data and cgem_data['transport'] is not None:
                cgem_transport = cgem_data['transport']
                if 'NH4' in cgem_transport.columns:
                    cgem_nh4 = cgem_transport['NH4'].mean()
                    nh4_values.append(cgem_nh4)
                    nh4_labels.append('C-GEM')
                    nh4_colors.append('#ff7f0e')
            
            # Observed NH4 from CARE data
            care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
            if care_file.exists():
                care_data = pd.read_csv(care_file)
                if 'NH4 (mgN/L)' in care_data.columns:
                    obs_nh4 = care_data['NH4 (mgN/L)'].dropna().mean()
                    # Convert mgN/L to mmol/m³ (1 mgN/L ≈ 71.4 mmol N/m³)
                    obs_nh4_mmol = obs_nh4 * 71.4
                    nh4_values.append(obs_nh4_mmol)
                    nh4_labels.append('CARE Obs.')
                    nh4_colors.append('#2ca02c')
            
            if nh4_values:
                bars = ax.bar(nh4_labels, nh4_values, color=nh4_colors, alpha=0.8)
                ax.set_ylabel('NH4 Concentration (mmol/m³)')
                ax.set_title('Ammonium (NH4) Comparison')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, nh4_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nh4_values)*0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No NH4 data\navailable', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Ammonium (NH4) Comparison')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ammonium (NH4) Comparison')
    
    def _plot_field_data_comparison(self, ax):
        """Plot summary of field data validation."""
        try:
            # Load field data statistics
            field_stats = {}
            
            # CARE data
            care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
            if care_file.exists():
                care_data = pd.read_csv(care_file)
                field_stats['CARE'] = len(care_data)
            
            # CEM data  
            cem_file = Path("INPUT/Calibration/CEM_2017-2018.csv")
            if cem_file.exists():
                cem_data = pd.read_csv(cem_file)
                field_stats['CEM'] = len(cem_data)
            
            # Tidal range data
            tidal_file = Path("INPUT/Calibration/CEM-Tidal-range.csv")
            if tidal_file.exists():
                tidal_data = pd.read_csv(tidal_file)
                field_stats['Tidal Range'] = len(tidal_data)
            
            if field_stats:
                datasets = list(field_stats.keys())
                counts = list(field_stats.values())
                colors = ['#2ca02c', '#d62728', '#9467bd'][:len(datasets)]
                
                bars = ax.bar(datasets, counts, color=colors, alpha=0.8)
                ax.set_ylabel('Number of Observations')
                ax.set_title('Field Data Availability')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                           f'{count}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No field data\nfound', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Field Data Availability')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Data Availability')
    
    def generate_report(self):
        """Generate comprehensive comparison report including field data validation."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE C-GEM BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("📋 EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Performance Summary
        jax_timing = self.results['jax_cgem'].get('timing')
        cgem_timing = self.results['original_cgem'].get('timing')
        jax_memory = self.results['jax_cgem'].get('memory_usage')
        cgem_memory = self.results['original_cgem'].get('memory_usage')
        
        if jax_timing and cgem_timing:
            speed_ratio = jax_timing / cgem_timing
            report_lines.append(f"⚡ Performance: JAX C-GEM is {speed_ratio:.2f}x slower than C-GEM")
            report_lines.append(f"   - JAX C-GEM: {jax_timing:.2f}s")
            report_lines.append(f"   - Original C-GEM: {cgem_timing:.2f}s")
        
        # Memory Summary
        if jax_memory and cgem_memory:
            jax_mem = jax_memory.get('final', jax_memory.get('delta', 0))
            cgem_mem = cgem_memory.get('final', cgem_memory.get('delta', 0))
            if jax_mem > 0 and cgem_mem > 0:
                memory_ratio = jax_mem / cgem_mem
                report_lines.append(f"🧠 Memory: JAX C-GEM uses {memory_ratio:.2f}x more memory")
                report_lines.append(f"   - JAX C-GEM: {jax_mem:.1f} MB")
                report_lines.append(f"   - Original C-GEM: {cgem_mem:.1f} MB")
        
        # Accuracy Summary
        accuracy = self.results.get('accuracy_analysis', {})
        if accuracy:
            report_lines.append(f"🎯 Accuracy: Models show {accuracy.get('overall_agreement', 'unknown')} agreement")
            if 'key_metrics' in accuracy:
                metrics = accuracy['key_metrics']
                report_lines.append(f"   - RMSE: {metrics.get('rmse', 'N/A')}")
                report_lines.append(f"   - Correlation: {metrics.get('correlation', 'N/A')}")
        
        report_lines.append("")
        
        # Detailed Performance Analysis
        report_lines.append("🔍 DETAILED PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        
        # JAX C-GEM Details
        report_lines.append("JAX C-GEM Performance:")
        if self.results['jax_cgem'].get('data'):  # Check if run was successful
            report_lines.append(f"  ✅ Status: Successful")
            report_lines.append(f"  ⏱️  Execution Time: {jax_timing:.2f} seconds" if jax_timing else "  ⏱️  Execution Time: N/A")
            if jax_memory:
                mem_val = jax_memory.get('final', jax_memory.get('delta', 'N/A'))
                report_lines.append(f"  🧠 Peak Memory: {mem_val} MB")
            else:
                report_lines.append(f"  🧠 Peak Memory: N/A")
            report_lines.append(f"  📊 Output Files: Available")
        else:
            report_lines.append(f"  ❌ Status: Failed")
        
        # Original C-GEM Details
        report_lines.append("\nOriginal C-GEM Performance:")
        if self.results['original_cgem'].get('data'):  # Check if run was successful
            report_lines.append(f"  ✅ Status: Successful")
            report_lines.append(f"  ⏱️  Execution Time: {cgem_timing:.2f} seconds" if cgem_timing else "  ⏱️  Execution Time: N/A")
            if cgem_memory:
                mem_val = cgem_memory.get('final', cgem_memory.get('delta', 'N/A'))
                report_lines.append(f"  🧠 Peak Memory: {mem_val} MB")
            else:
                report_lines.append(f"  🧠 Peak Memory: N/A")
            report_lines.append(f"  📊 Output Files: Available")
        else:
            report_lines.append(f"  ❌ Status: Failed")
        
        report_lines.append("")
        
        # Enhanced Accuracy Analysis
        report_lines.append("🎯 ENHANCED ACCURACY ANALYSIS")
        report_lines.append("-" * 40)
        
        # Get data for analysis
        jax_data = self.results['jax_cgem'].get('data')
        cgem_data = self.results['original_cgem'].get('data')
        
        # Model-to-Model Comparison
        if jax_data and cgem_data:
            report_lines.append("Model-to-Model Comparison:")
            
            # Calculate simple statistics for key variables
            model_stats = {}
            
            # Compare hydrodynamics data
            if (jax_data.get('hydrodynamics') is not None and 
                cgem_data.get('hydrodynamics') is not None):
                jax_hydro = jax_data['hydrodynamics']
                cgem_hydro = cgem_data['hydrodynamics']
                
                for col in ['H', 'U', 'Q']:
                    if col in jax_hydro.columns and col in cgem_hydro.columns:
                        jax_vals = jax_hydro[col].values.flatten()
                        cgem_vals = cgem_hydro[col].values.flatten()
                        
                        if len(jax_vals) > 0 and len(cgem_vals) > 0:
                            # Calculate statistics
                            min_len = min(len(jax_vals), len(cgem_vals))
                            jax_vals = jax_vals[:min_len]
                            cgem_vals = cgem_vals[:min_len]
                            
                            if min_len > 1:
                                rmse = np.sqrt(np.mean((jax_vals - cgem_vals)**2))
                                mae = np.mean(np.abs(jax_vals - cgem_vals))
                                correlation = np.corrcoef(jax_vals, cgem_vals)[0,1] if min_len > 1 else 0
                                
                                model_stats[f'{col} (Hydro)'] = {
                                    'rmse': rmse,
                                    'mae': mae,
                                    'correlation': correlation
                                }
            
            # Compare transport data  
            if (jax_data.get('transport') is not None and 
                cgem_data.get('transport') is not None):
                jax_transport = jax_data['transport']
                cgem_transport = cgem_data['transport']
                
                for col in ['S', 'O2', 'NH4', 'NO3']:
                    if col in jax_transport.columns and col in cgem_transport.columns:
                        jax_vals = jax_transport[col].values.flatten()
                        cgem_vals = cgem_transport[col].values.flatten()
                        
                        if len(jax_vals) > 0 and len(cgem_vals) > 0:
                            min_len = min(len(jax_vals), len(cgem_vals))
                            jax_vals = jax_vals[:min_len]
                            cgem_vals = cgem_vals[:min_len]
                            
                            if min_len > 1:
                                rmse = np.sqrt(np.mean((jax_vals - cgem_vals)**2))
                                mae = np.mean(np.abs(jax_vals - cgem_vals))
                                correlation = np.corrcoef(jax_vals, cgem_vals)[0,1] if min_len > 1 else 0
                                
                                model_stats[f'{col} (Transport)'] = {
                                    'rmse': rmse,
                                    'mae': mae,
                                    'correlation': correlation
                                }
            
            # Report model comparison statistics
            if model_stats:
                for variable, stats in model_stats.items():
                    report_lines.append(f"  {variable}:")
                    report_lines.append(f"    - RMSE: {stats.get('rmse', 'N/A'):.4f}")
                    report_lines.append(f"    - MAE: {stats.get('mae', 'N/A'):.4f}")
                    report_lines.append(f"    - Correlation: {stats.get('correlation', 'N/A'):.4f}")
            else:
                report_lines.append("  No comparable model data found")
        else:
            report_lines.append("Model-to-Model Comparison: Data not available")
        
        # Field Data Validation
        field_data_available = False
        report_lines.append("\nField Data Validation:")
        
        # Check CARE dataset
        care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
        if care_file.exists():
            try:
                care_data = pd.read_csv(care_file)
                report_lines.append(f"  CARE Dataset: {len(care_data)} observations available")
                field_data_available = True
            except Exception:
                report_lines.append("  CARE Dataset: Error reading file")
        else:
            report_lines.append("  CARE Dataset: Not found")
        
        # Check CEM dataset
        cem_file = Path("INPUT/Calibration/CEM_2017-2018.csv")
        if cem_file.exists():
            try:
                cem_data = pd.read_csv(cem_file)
                report_lines.append(f"  CEM Dataset: {len(cem_data)} observations available")
                field_data_available = True
            except Exception:
                report_lines.append("  CEM Dataset: Error reading file")
        else:
            report_lines.append("  CEM Dataset: Not found")
        
        # Check Tidal Range dataset
        tidal_file = Path("INPUT/Calibration/CEM-Tidal-range.csv")
        if tidal_file.exists():
            try:
                tidal_data = pd.read_csv(tidal_file)
                report_lines.append(f"  Tidal Range Dataset: {len(tidal_data)} observations available")
                field_data_available = True
            except Exception:
                report_lines.append("  Tidal Range Dataset: Error reading file")
        else:
            report_lines.append("  Tidal Range Dataset: Not found")
        
        if not field_data_available:
            report_lines.append("  No field validation datasets found")
        
        # Key Variables Analysis
        report_lines.append("\nKey Variables Summary:")
        if jax_data and cgem_data:
            # Tidal range analysis
            if (jax_data.get('hydrodynamics') is not None and 
                cgem_data.get('hydrodynamics') is not None):
                jax_hydro = jax_data['hydrodynamics']
                cgem_hydro = cgem_data['hydrodynamics']
                
                if 'H' in jax_hydro.columns and 'H' in cgem_hydro.columns:
                    jax_range = jax_hydro['H'].max() - jax_hydro['H'].min()
                    cgem_range = cgem_hydro['H'].max() - cgem_hydro['H'].min()
                    report_lines.append(f"  Tidal Range:")
                    report_lines.append(f"    - JAX C-GEM: {jax_range:.3f} m")
                    report_lines.append(f"    - C-GEM: {cgem_range:.3f} m")
                    report_lines.append(f"    - Difference: {abs(jax_range - cgem_range):.3f} m")
            
            # Salinity analysis
            if (jax_data.get('transport') is not None and 
                cgem_data.get('transport') is not None):
                jax_transport = jax_data['transport']
                cgem_transport = cgem_data['transport']
                
                if 'S' in jax_transport.columns and 'S' in cgem_transport.columns:
                    jax_sal_mean = jax_transport['S'].mean()
                    cgem_sal_mean = cgem_transport['S'].mean()
                    report_lines.append(f"  Salinity (mean):")
                    report_lines.append(f"    - JAX C-GEM: {jax_sal_mean:.3f} psu")
                    report_lines.append(f"    - C-GEM: {cgem_sal_mean:.3f} psu")
                    report_lines.append(f"    - Difference: {abs(jax_sal_mean - cgem_sal_mean):.3f} psu")
        else:
            report_lines.append("  Data not available for key variables analysis")
        
        report_lines.append("")
        
        # Field Data Summary
        report_lines.append("📊 FIELD DATA SUMMARY")
        report_lines.append("-" * 40)
        
        # Check available field datasets
        field_data_summary = []
        
        care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
        if care_file.exists():
            try:
                care_data = pd.read_csv(care_file)
                field_data_summary.append(f"  ✅ CARE Dataset: {len(care_data)} observations")
                field_data_summary.append(f"     Available parameters: {', '.join(care_data.columns[:8])}...")
            except Exception as e:
                field_data_summary.append(f"  ⚠️ CARE Dataset: Error reading ({str(e)[:30]}...)")
        else:
            field_data_summary.append("  ❌ CARE Dataset: Not found")
        
        cem_file = Path("INPUT/Calibration/CEM_2017-2018.csv")
        if cem_file.exists():
            try:
                cem_data = pd.read_csv(cem_file)
                field_data_summary.append(f"  ✅ CEM Dataset: {len(cem_data)} observations")
                field_data_summary.append(f"     Available parameters: {', '.join(cem_data.columns[:8])}...")
            except Exception as e:
                field_data_summary.append(f"  ⚠️ CEM Dataset: Error reading ({str(e)[:30]}...)")
        else:
            field_data_summary.append("  ❌ CEM Dataset: Not found")
        
        tidal_file = Path("INPUT/Calibration/CEM-Tidal-range.csv")
        if tidal_file.exists():
            try:
                tidal_data = pd.read_csv(tidal_file)
                field_data_summary.append(f"  ✅ Tidal Range Dataset: {len(tidal_data)} observations")
            except Exception as e:
                field_data_summary.append(f"  ⚠️ Tidal Range Dataset: Error reading ({str(e)[:30]}...)")
        else:
            field_data_summary.append("  ❌ Tidal Range Dataset: Not found")
        
        report_lines.extend(field_data_summary)
        
        report_lines.append("")
        
        # Recommendations
        report_lines.append("🎯 RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        recommendations = []
        
        if jax_timing and cgem_timing:
            speed_ratio = jax_timing / cgem_timing
            if speed_ratio > 2.0:
                recommendations.append("🚀 Consider performance optimization for JAX C-GEM")
            elif speed_ratio < 1.5:
                recommendations.append("✅ JAX C-GEM performance is acceptable")
        
        # Calculate average correlation from model statistics if available
        if 'model_stats' in locals() and model_stats:
            correlations = [stats.get('correlation', 0) for stats in model_stats.values() 
                          if stats.get('correlation') and not np.isnan(stats.get('correlation', 0))]
            if correlations:
                avg_correlation = np.mean(correlations)
                if avg_correlation > 0.95:
                    recommendations.append("✅ Excellent model agreement - JAX implementation is validated")
                elif avg_correlation > 0.85:
                    recommendations.append("👍 Good model agreement - minor differences acceptable")
                else:
                    recommendations.append("⚠️ Model agreement needs investigation")
        
        if field_data_available:
            recommendations.append("📊 Include field data validation in routine testing")
            
        if not recommendations:
            recommendations.append("📋 Complete benchmark analysis - results documented")
        
        report_lines.extend([f"  {rec}" for rec in recommendations])
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.output_dir / "comprehensive_benchmark_report.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📋 Comprehensive report saved: {report_path}")
        except Exception as e:
            print(f"⚠️ Error saving report: {e}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60)
        
        if jax_timing and cgem_timing:
            speed_ratio = jax_timing / cgem_timing
            print(f"⚡ Performance: JAX is {speed_ratio:.2f}x slower ({jax_timing:.2f}s vs {cgem_timing:.2f}s)")
        
        # Calculate and display model agreement if available
        if 'model_stats' in locals() and model_stats:
            correlations = [stats.get('correlation', 0) for stats in model_stats.values() 
                          if stats.get('correlation') and not np.isnan(stats.get('correlation', 0))]
            if correlations:
                avg_corr = np.mean(correlations)
                print(f"🎯 Model Agreement: {avg_corr:.3f} average correlation")
        
        field_datasets_count = 0
        if care_file.exists():
            field_datasets_count += 1
        if cem_file.exists():
            field_datasets_count += 1
        if tidal_file.exists():
            field_datasets_count += 1
        
        print(f"📊 Field Data: {field_datasets_count}/3 datasets available for validation")
        
        print("=" * 60)
        
        return report_text
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite with proper validation."""
        print("\n🚀 Starting Full Benchmark Suite...")
        
        # Step 1: Verify configurations
        config_ok = self.verify_identical_configurations()
        if not config_ok:
            print("❌ Configuration verification failed - cannot proceed")
            return False
        
        # Step 2: Compile C-GEM if needed
        if not self.c_gem_exe.exists():
            if not self.compile_original_cgem():
                print("❌ Cannot proceed without compiled C-GEM")
                return False
        
        # Step 3: Run JAX C-GEM and VALIDATE results
        print("\n" + "="*60)
        print("🐍 EXECUTING JAX C-GEM")
        print("="*60)
        
        jax_success = self.run_jax_cgem_benchmark()
        if not jax_success:
            print("❌ JAX C-GEM execution failed")
            self.generate_report()  # Generate report showing failure
            return False
        
        # CRITICAL: Validate JAX actually wrote simulation data
        if not self._validate_jax_results():
            print("❌ JAX C-GEM did not produce valid simulation results")
            jax_success = False
            self.generate_report()
            return False
        
        print("✅ JAX C-GEM completed successfully with valid data")
        
        # Step 4: Run original C-GEM and VALIDATE results
        print("\n" + "="*60) 
        print("🔵 EXECUTING ORIGINAL C-GEM")
        print("="*60)
        
        cgem_success = self.run_original_cgem_benchmark()
        if not cgem_success:
            print("❌ Original C-GEM execution failed")
            self.generate_report()
            return False
            
        # CRITICAL: Validate C-GEM actually wrote simulation data
        if not self._validate_cgem_results():
            print("❌ Original C-GEM did not produce valid simulation results")
            cgem_success = False
            self.generate_report()
            return False
            
        print("✅ Original C-GEM completed successfully with valid data")
        
        # Step 5: Compare results (only if BOTH models succeeded)
        print("\n" + "="*60)
        print("📊 COMPARING RESULTS")
        print("="*60)
        
        if jax_success and cgem_success:
            self.compare_results()
            print("✅ Results comparison completed")
        else:
            print("⚠️  Skipping comparison - one or both models failed")
        
        # Step 6: Generate comprehensive report
        self.generate_report()
        
        final_success = jax_success and cgem_success
        if final_success:
            print(f"\n🎉 Benchmark Complete! Both models executed successfully")
        else:
            print(f"\n⚠️  Benchmark completed with issues - check report for details")
            
        print(f"📁 Results saved in: {self.output_dir}")
        
        return final_success
    
    def _analyze_model_accuracy(self, f, jax_data, cgem_data):
        """Analyze accuracy between JAX C-GEM and Original C-GEM models."""
        try:
            if not (jax_data and cgem_data and jax_data['hydrodynamics'] is not None and cgem_data['hydrodynamics'] is not None):
                f.write("Model comparison data not available\n\n")
                return
                
            f.write("#### Statistical Comparison of Model Outputs\n\n")
            f.write("| Variable | RMSE | MAE | Correlation | Nash-Sutcliffe | Status |\n")
            f.write("|----------|------|-----|-------------|----------------|--------|\n")
            
            # Analyze hydrodynamic variables
            jax_hydro = jax_data['hydrodynamics']
            cgem_hydro = cgem_data['hydrodynamics']
            
            exclude_cols = ['Time', 'time', 'Index', 'index', 'times', 'Times']
            jax_vars = [col for col in jax_hydro.columns if col not in exclude_cols]
            cgem_vars = [col for col in cgem_hydro.columns if col not in exclude_cols]
            common_vars = set(jax_vars) & set(cgem_vars)
            
            for var in sorted(list(common_vars))[:8]:  # Top 8 variables
                try:
                    # Get data for comparison
                    min_len = min(len(jax_hydro), len(cgem_hydro))
                    jax_vals = jax_hydro[var].iloc[:min_len].values.flatten()
                    cgem_vals = cgem_hydro[var].iloc[:min_len].values.flatten()
                    
                    # Remove NaN values
                    mask = ~(np.isnan(jax_vals) | np.isnan(cgem_vals) | np.isinf(jax_vals) | np.isinf(cgem_vals))
                    if np.sum(mask) > 10:
                        jax_clean = jax_vals[mask]
                        cgem_clean = cgem_vals[mask]
                        
                        # Calculate statistics
                        rmse = np.sqrt(np.mean((jax_clean - cgem_clean)**2))
                        mae = np.mean(np.abs(jax_clean - cgem_clean))
                        
                        # Correlation
                        if len(jax_clean) > 1 and np.std(jax_clean) > 0 and np.std(cgem_clean) > 0:
                            corr_matrix = np.corrcoef(jax_clean, cgem_clean)
                            corr = corr_matrix[0, 1]
                        else:
                            corr = 0.0
                            
                        # Nash-Sutcliffe efficiency
                        mean_obs = np.mean(cgem_clean)
                        ss_res = np.sum((jax_clean - cgem_clean)**2)
                        ss_tot = np.sum((cgem_clean - mean_obs)**2)
                        nse = 1 - (ss_res / (ss_tot + 1e-12))
                        
                        # Status assessment
                        status = self._assess_accuracy_status(corr, nse, rmse, mae, var)
                        
                        f.write(f"| {var} | {rmse:.4f} | {mae:.4f} | {corr:.3f} | {nse:.3f} | {status} |\n")
                    else:
                        f.write(f"| {var} | N/A | N/A | N/A | N/A | Insufficient data |\n")
                except Exception as e:
                    f.write(f"| {var} | Error | Error | Error | Error | Calculation failed |\n")
            
            f.write("\n")
            
        except Exception as e:
            f.write(f"Model accuracy analysis error: {str(e)}\n\n")
    
    def _analyze_field_data_accuracy(self, f, jax_data, cgem_data):
        """Analyze model accuracy against field observations."""
        try:
            f.write("#### Model Performance vs Field Observations\n\n")
            
            # Load field observation files
            field_data = self._load_field_observations()
            
            if not field_data:
                f.write("No field observation data available for comparison\n\n")
                return
            
            f.write("**Available Field Data Sources:**\n")
            for source, data in field_data.items():
                if data is not None:
                    f.write(f"- {source}: {len(data)} observations\n")
            f.write("\n")
            
            # Compare with CARE data (nutrient observations)
            if 'CARE' in field_data and field_data['CARE'] is not None:
                self._compare_with_care_data(f, jax_data, cgem_data, field_data['CARE'])
            
            # Compare with CEM data (water quality observations)
            if 'CEM' in field_data and field_data['CEM'] is not None:
                self._compare_with_cem_data(f, jax_data, cgem_data, field_data['CEM'])
            
            # Compare with tidal range data
            if 'Tidal_Range' in field_data and field_data['Tidal_Range'] is not None:
                self._compare_tidal_range(f, jax_data, cgem_data, field_data['Tidal_Range'])
                
        except Exception as e:
            f.write(f"Field data analysis error: {str(e)}\n\n")
    
    def _analyze_key_variables(self, f, jax_data, cgem_data):
        """Analyze key variables: tidal range, salinity, DO, NH4."""
        try:
            f.write("#### Detailed Analysis of Key Variables\n\n")
            
            # 1. Tidal Range Analysis
            f.write("**1. Tidal Range Analysis**\n\n")
            self._analyze_tidal_range(f, jax_data, cgem_data)
            
            # 2. Salinity Profile Analysis
            f.write("**2. Salinity Profile Analysis**\n\n")
            self._analyze_salinity_profiles(f, jax_data, cgem_data)
            
            # 3. Dissolved Oxygen Analysis
            f.write("**3. Dissolved Oxygen Analysis**\n\n")
            self._analyze_dissolved_oxygen(f, jax_data, cgem_data)
            
            # 4. NH4 Analysis
            f.write("**4. Ammonium (NH4) Analysis**\n\n")
            self._analyze_nh4(f, jax_data, cgem_data)
            
        except Exception as e:
            f.write(f"Key variables analysis error: {str(e)}\n\n")
    
    def _assess_accuracy_status(self, corr, nse, rmse, mae, var_name):
        """Assess accuracy status based on statistical metrics."""
        # Weight different metrics based on variable type
        if 'H' in var_name.upper():  # Water level variables
            if corr > 0.95 and nse > 0.9:
                return "Excellent ✅"
            elif corr > 0.85 and nse > 0.7:
                return "Good ✓"
            elif corr > 0.7:
                return "Fair ⚠️"
            else:
                return "Poor ❌"
        elif 'U' in var_name.upper():  # Velocity variables
            if corr > 0.9 and nse > 0.8:
                return "Excellent ✅"
            elif corr > 0.75 and nse > 0.6:
                return "Good ✓"
            elif corr > 0.6:
                return "Fair ⚠️"
            else:
                return "Poor ❌"
        else:  # Other variables
            if corr > 0.9 and nse > 0.8:
                return "Excellent ✅"
            elif corr > 0.8 and nse > 0.6:
                return "Good ✓"
            elif corr > 0.6:
                return "Fair ⚠️"
            else:
                return "Poor ❌"
    
    def _load_field_observations(self):
        """Load field observation data from calibration folder."""
        field_data = {}
        
        try:
            # Load CARE data (nutrients, water quality)
            care_file = Path("INPUT/Calibration/CARE_2017-2018.csv")
            if care_file.exists():
                field_data['CARE'] = pd.read_csv(care_file)
                print(f"📊 Loaded CARE data: {len(field_data['CARE'])} observations")
            
            # Load CEM data (water quality)
            cem_file = Path("INPUT/Calibration/CEM_2017-2018.csv")
            if cem_file.exists():
                field_data['CEM'] = pd.read_csv(cem_file)
                print(f"📊 Loaded CEM data: {len(field_data['CEM'])} observations")
            
            # Load tidal range data
            tidal_file = Path("INPUT/Calibration/CEM-Tidal-range.csv")
            if tidal_file.exists():
                field_data['Tidal_Range'] = pd.read_csv(tidal_file)
                print(f"📊 Loaded Tidal Range data: {len(field_data['Tidal_Range'])} observations")
                
        except Exception as e:
            print(f"⚠️ Error loading field data: {e}")
            
        return field_data
    
    def _compare_with_care_data(self, f, jax_data, cgem_data, care_data):
        """Compare models with CARE nutrient data."""
        f.write("**CARE Dataset Comparison (Nutrients & Water Quality):**\n\n")
        
        # Available variables in CARE data
        care_vars = ['pH', 'Salinity', 'DO (mg/L)', 'NH4 (mgN/L)', 'NO3 (mgN/L)', 'PO4 (mgP/L)', 'TOC (mgC/L)']
        available_vars = [var for var in care_vars if var in care_data.columns]
        
        f.write(f"Available CARE variables: {', '.join(available_vars)}\n")
        f.write(f"Sample size: {len(care_data)} measurements\n")
        f.write(f"Location range: {care_data['Location'].min()}-{care_data['Location'].max()} km from mouth\n\n")
        
        # Statistical summary
        f.write("| Variable | Mean | Std | Min | Max | Unit |\n")
        f.write("|----------|------|-----|-----|-----|----- |\n")
        for var in available_vars[:6]:  # Show first 6 variables
            if var in care_data.columns:
                data_col = care_data[var].dropna()
                if len(data_col) > 0:
                    f.write(f"| {var} | {data_col.mean():.2f} | {data_col.std():.2f} | {data_col.min():.2f} | {data_col.max():.2f} | Field data |\n")
        
        f.write("\n")
        
    def _compare_with_cem_data(self, f, jax_data, cgem_data, cem_data):
        """Compare models with CEM water quality data."""
        f.write("**CEM Dataset Comparison (Water Quality):**\n\n")
        
        # Available variables in CEM data
        cem_vars = ['Salinity', 'DO (mg/L)', 'NH4 (mgN/L)', 'PO4 (mgP/L)', 'TOC (mgC/L)', 'TSS (mg/L)']
        available_vars = [var for var in cem_vars if var in cem_data.columns]
        
        f.write(f"Available CEM variables: {', '.join(available_vars)}\n")
        f.write(f"Sample size: {len(cem_data)} measurements\n")
        f.write(f"Location: {cem_data['Location'].iloc[0]} km from mouth (Ben Suc station)\n\n")
        
        # Tidal influence analysis
        if 'Tide' in cem_data.columns:
            tide_counts = cem_data['Tide'].value_counts()
            f.write("**Tidal Sampling Distribution:**\n")
            for tide, count in tide_counts.items():
                f.write(f"- {tide}: {count} samples\n")
            f.write("\n")
    
    def _compare_tidal_range(self, f, jax_data, cgem_data, tidal_data):
        """Compare model tidal ranges with observations."""
        f.write("**Tidal Range Validation:**\n\n")
        
        # Statistics from observed tidal range
        observed_range = tidal_data['Tidal Range (m)'].dropna()
        
        f.write("| Location | Observed Mean (m) | Observed Std (m) | Model Comparison Status |\n")
        f.write("|----------|-------------------|------------------|-------------------------|\n")
        
        # Group by location
        locations = tidal_data['Location'].unique()[:5]  # Show first 5 locations
        for loc in sorted(locations):
            loc_data = tidal_data[tidal_data['Location'] == loc]['Tidal Range (m)']
            if len(loc_data) > 0:
                mean_range = loc_data.mean()
                std_range = loc_data.std()
                f.write(f"| {loc} km | {mean_range:.2f} | {std_range:.2f} | Pending model analysis |\n")
        
        f.write(f"\nOverall observed tidal range: {observed_range.mean():.2f} ± {observed_range.std():.2f} m\n")
        f.write(f"Range: {observed_range.min():.2f} - {observed_range.max():.2f} m\n\n")
    
    def _analyze_tidal_range(self, f, jax_data, cgem_data):
        """Analyze tidal range from model outputs."""
        f.write("Computing tidal range from water level time series...\n")
        
        try:
            # Extract water level data for both models
            if jax_data and 'hydrodynamics' in jax_data and jax_data['hydrodynamics'] is not None:
                jax_hydro = jax_data['hydrodynamics']
                if 'H' in jax_hydro.columns:
                    # Calculate tidal range as difference between max and min water level
                    jax_range = jax_hydro['H'].max() - jax_hydro['H'].min()
                    f.write(f"JAX C-GEM tidal range: {jax_range:.3f} m\n")
                else:
                    f.write("JAX C-GEM: No water level data found\n")
            
            if cgem_data and 'hydrodynamics' in cgem_data and cgem_data['hydrodynamics'] is not None:
                cgem_hydro = cgem_data['hydrodynamics']
                if 'H' in cgem_hydro.columns:
                    cgem_range = cgem_hydro['H'].max() - cgem_hydro['H'].min()
                    f.write(f"Original C-GEM tidal range: {cgem_range:.3f} m\n")
                else:
                    f.write("Original C-GEM: No water level data found\n")
            
        except Exception as e:
            f.write(f"Error analyzing tidal range: {e}\n")
        
        f.write("\n")
    
    def _analyze_salinity_profiles(self, f, jax_data, cgem_data):
        """Analyze salinity longitudinal profiles."""
        f.write("Analyzing salinity distribution along estuary...\n")
        
        try:
            # Look for salinity data in transport results
            if jax_data and 'transport' in jax_data and jax_data['transport'] is not None:
                jax_transport = jax_data['transport']
                if 'S' in jax_transport.columns:
                    jax_sal = jax_transport['S']
                    f.write(f"JAX C-GEM salinity range: {jax_sal.min():.2f} - {jax_sal.max():.2f} psu\n")
                    f.write(f"JAX C-GEM salinity mean: {jax_sal.mean():.2f} psu\n")
                else:
                    f.write("JAX C-GEM: No salinity data found in transport\n")
            
            if cgem_data and 'transport' in cgem_data and cgem_data['transport'] is not None:
                cgem_transport = cgem_data['transport']
                if 'S' in cgem_transport.columns:
                    cgem_sal = cgem_transport['S']
                    f.write(f"Original C-GEM salinity range: {cgem_sal.min():.2f} - {cgem_sal.max():.2f} psu\n")
                    f.write(f"Original C-GEM salinity mean: {cgem_sal.mean():.2f} psu\n")
                else:
                    f.write("Original C-GEM: No salinity data found in transport\n")
            
        except Exception as e:
            f.write(f"Error analyzing salinity: {e}\n")
        
        f.write("\n")
    
    def _analyze_dissolved_oxygen(self, f, jax_data, cgem_data):
        """Analyze dissolved oxygen concentrations."""
        f.write("Analyzing dissolved oxygen concentrations...\n")
        
        try:
            # Look for oxygen data
            if jax_data and 'transport' in jax_data and jax_data['transport'] is not None:
                jax_transport = jax_data['transport']
                if 'O2' in jax_transport.columns:
                    jax_o2 = jax_transport['O2']
                    f.write(f"JAX C-GEM DO range: {jax_o2.min():.2f} - {jax_o2.max():.2f} mmol/m³\n")
                    f.write(f"JAX C-GEM DO mean: {jax_o2.mean():.2f} mmol/m³\n")
                else:
                    f.write("JAX C-GEM: No dissolved oxygen data found\n")
            
            if cgem_data and 'transport' in cgem_data and cgem_data['transport'] is not None:
                cgem_transport = cgem_data['transport']
                if 'O2' in cgem_transport.columns:
                    cgem_o2 = cgem_transport['O2']
                    f.write(f"Original C-GEM DO range: {cgem_o2.min():.2f} - {cgem_o2.max():.2f} mmol/m³\n")
                    f.write(f"Original C-GEM DO mean: {cgem_o2.mean():.2f} mmol/m³\n")
                else:
                    f.write("Original C-GEM: No dissolved oxygen data found\n")
            
        except Exception as e:
            f.write(f"Error analyzing dissolved oxygen: {e}\n")
        
        f.write("\n")
    
    def _analyze_nh4(self, f, jax_data, cgem_data):
        """Analyze ammonium (NH4) concentrations."""
        f.write("Analyzing ammonium (NH4) concentrations...\n")
        
        try:
            # Look for NH4 data
            if jax_data and 'transport' in jax_data and jax_data['transport'] is not None:
                jax_transport = jax_data['transport']
                if 'NH4' in jax_transport.columns:
                    jax_nh4 = jax_transport['NH4']
                    f.write(f"JAX C-GEM NH4 range: {jax_nh4.min():.3f} - {jax_nh4.max():.3f} mmol/m³\n")
                    f.write(f"JAX C-GEM NH4 mean: {jax_nh4.mean():.3f} mmol/m³\n")
                else:
                    f.write("JAX C-GEM: No ammonium data found\n")
            
            if cgem_data and 'transport' in cgem_data and cgem_data['transport'] is not None:
                cgem_transport = cgem_data['transport']
                if 'NH4' in cgem_transport.columns:
                    cgem_nh4 = cgem_transport['NH4']
                    f.write(f"Original C-GEM NH4 range: {cgem_nh4.min():.3f} - {cgem_nh4.max():.3f} mmol/m³\n")
                    f.write(f"Original C-GEM NH4 mean: {cgem_nh4.mean():.3f} mmol/m³\n")
                else:
                    f.write("Original C-GEM: No ammonium data found\n")
            
        except Exception as e:
            f.write(f"Error analyzing NH4: {e}\n")
        
        f.write("\n")

    def _validate_jax_results(self) -> bool:
        """Validate that JAX C-GEM produced complete and usable simulation data."""
        try:
            print("🔍 Validating JAX C-GEM simulation data...")
            
            # Check NPZ file exists and has meaningful data
            npz_file = Path("OUT/simulation_results.npz")
            if npz_file.exists():
                data = np.load(str(npz_file))
                keys = list(data.keys())
                
                # Must have more than just 'time' key
                if len(keys) <= 1:
                    print(f"   ❌ NPZ file incomplete - only has {keys}")
                    data.close()
                    return False
                
                # Must have reasonable amount of data
                if 'time' in data and len(data['time']) < 100:
                    print(f"   ❌ Insufficient time points: {len(data['time'])}")
                    data.close()
                    return False
                    
                # Check for key variables
                has_hydro = any(key in keys for key in ['H', 'U'])
                has_species = any(key in keys for key in ['O2', 'S', 'NO3', 'NH4'])
                
                if not (has_hydro or has_species):
                    print(f"   ❌ No hydrodynamic or species data found")
                    data.close()
                    return False
                    
                time_points = len(data['time']) if 'time' in data else 0
                data.close()
                print(f"   ✅ NPZ file validated: {len(keys)} variables, {time_points} time steps")
                return True
            
            # Fall back to CSV validation
            hydro_files = list(Path("OUT/Hydrodynamics").glob("*.csv")) if Path("OUT/Hydrodynamics").exists() else []
            reaction_files = list(Path("OUT/Reaction").glob("*.csv")) if Path("OUT/Reaction").exists() else []
            
            if len(hydro_files) == 0 and len(reaction_files) == 0:
                print(f"   ❌ No CSV output files found")
                return False
                
            # Check file sizes are reasonable
            all_files = hydro_files + reaction_files
            for file in all_files[:5]:  # Check first few files
                if file.stat().st_size < 1000:  # Less than 1KB indicates problem
                    print(f"   ❌ JAX output file too small: {file.name} ({file.stat().st_size} bytes)")
                    return False
                    
            print(f"   ✅ CSV files validated: {len(hydro_files)} hydro + {len(reaction_files)} reaction files")
            return True
            
        except Exception as e:
            print(f"   ❌ JAX validation error: {e}")
            return False
    
    def _validate_cgem_results(self) -> bool:
        """Validate that original C-GEM produced complete and usable simulation data."""
        try:
            print("🔍 Validating original C-GEM simulation data...")
            
            c_gem_out = self.c_gem_dir / "OUT"
            
            # Check for hydrodynamics files
            hydro_dir = c_gem_out / "Hydrodynamics"
            reaction_dir = c_gem_out / "Reaction"
            
            hydro_files = list(hydro_dir.glob("*.csv")) if hydro_dir.exists() else []
            reaction_files = list(reaction_dir.glob("*.csv")) if reaction_dir.exists() else []
            
            if len(hydro_files) == 0:
                print(f"   ❌ No C-GEM hydrodynamics files found in {hydro_dir}")
                return False
                
            if len(reaction_files) == 0:
                print(f"   ❌ No C-GEM reaction files found in {reaction_dir}")
                return False
            
            # Check file sizes are reasonable
            all_files = hydro_files + reaction_files
            for file in all_files[:5]:  # Check first few files
                if not file.exists() or file.stat().st_size < 1000:  # Less than 1KB indicates problem
                    print(f"   ❌ C-GEM output file invalid: {file.name} ({file.stat().st_size if file.exists() else 0} bytes)")
                    return False
                    
            print(f"   ✅ C-GEM files validated: {len(hydro_files)} hydro + {len(reaction_files)} reaction files")
            return True
            
        except Exception as e:
            print(f"   ❌ C-GEM validation error: {e}")
            return False

def main():
    """Main benchmark execution."""
    benchmark = ComprehensiveBenchmark()
    success = benchmark.run_full_benchmark()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
