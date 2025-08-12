#!/usr/bin/env python3
"""
JAX C-GEM Setup Verification Tool

This script checks that your JAX C-GEM installation and configuration
are ready for running simulations.

Usage: python tools/setup_check.py
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'jax': 'JAX computational framework',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'matplotlib': 'Plotting',
        'scipy': 'Scientific computing'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package:12s} - {description}")
        except ImportError:
            print(f"❌ {package:12s} - {description} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_jax_functionality():
    """Check if JAX is working properly."""
    print("\n⚡ Testing JAX functionality...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Test basic operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        
        print(f"✅ JAX basic operations working (test sum: {y})")
        
        # Test JIT compilation
        @jax.jit
        def test_jit(x):
            return x * 2
        
        result = test_jit(5.0)
        print(f"✅ JAX JIT compilation working (test result: {result})")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX functionality test failed: {e}")
        return False

def check_directory_structure():
    """Check if required directories exist."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        'src/',
        'src/core/',
        'config/',
        'INPUT/',
        'INPUT/Boundary/',
        'INPUT/Calibration/',
        'tools/',
        'tools/plotting/',
        'tools/validation/'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (MISSING)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n💡 Create missing directories with:")
        for dir_path in missing_dirs:
            print(f"   mkdir -p {dir_path}")
        return False
    
    return True

def check_configuration_files():
    """Check if configuration files exist and are valid."""
    print("\n⚙️  Checking configuration files...")
    
    config_files = {
        'config/model_config.txt': 'Model configuration',
        'config/input_data_config.txt': 'Input data configuration'
    }
    
    missing_files = []
    
    for file_path, description in config_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path} - {description}")
            
            # Quick validation of content
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if len(content.strip()) > 0:
                        print(f"   📄 File contains {len(content.splitlines())} lines")
                    else:
                        print(f"   ⚠️  File is empty")
            except Exception as e:
                print(f"   ⚠️  Could not read file: {e}")
                
        else:
            print(f"❌ {file_path} - {description} (MISSING)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_input_data():
    """Check if input data directories have required structure."""
    print("\n💾 Checking input data structure...")
    
    # Check boundary conditions
    boundary_dirs = ['INPUT/Boundary/UB/', 'INPUT/Boundary/LB/']
    
    for boundary_dir in boundary_dirs:
        if os.path.exists(boundary_dir):
            files = list(Path(boundary_dir).glob('*.csv'))
            print(f"✅ {boundary_dir} ({len(files)} CSV files)")
        else:
            print(f"⚠️  {boundary_dir} (directory missing - will use defaults)")
    
    # Check calibration data
    calibration_dir = 'INPUT/Calibration/'
    if os.path.exists(calibration_dir):
        files = list(Path(calibration_dir).glob('*.csv'))
        print(f"✅ {calibration_dir} ({len(files)} observation files)")
        if len(files) == 0:
            print("   💡 Add field observation CSV files here for model validation")
    else:
        print(f"⚠️  {calibration_dir} (no field data - validation will be skipped)")
    
    return True

def check_core_modules():
    """Check if core model modules can be imported."""
    print("\n🔧 Testing core model modules...")
    
    core_modules = {
        'src.core.config_parser': 'Configuration parser',
        'src.core.data_loader': 'Data loader',
        'src.core.hydrodynamics': 'Hydrodynamics module',
        'src.core.transport': 'Transport module',
        'src.core.biogeochemistry': 'Biogeochemistry module'
    }
    
    sys.path.insert(0, 'src')
    
    failed_imports = []
    
    for module_name, description in core_modules.items():
        try:
            __import__(module_name)
            print(f"✅ {module_name:25s} - {description}")
        except ImportError as e:
            print(f"❌ {module_name:25s} - {description} (FAILED: {e})")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def run_quick_performance_test():
    """Run a quick performance test."""
    print("\n🚀 Quick performance test...")
    
    try:
        import time
        import jax
        import jax.numpy as jnp
        
        # Test array operations
        n = 10000
        x = jnp.arange(n, dtype=jnp.float32)
        
        @jax.jit
        def test_computation(x):
            return jnp.sum(x * x + jnp.sin(x))
        
        # Warm up JIT
        _ = test_computation(x)
        
        # Time the computation
        start_time = time.time()
        for _ in range(100):
            result = test_computation(x)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000 / 100
        
        print(f"✅ JAX performance test: {elapsed_ms:.2f} ms per operation")
        
        if elapsed_ms < 1.0:
            print("   🚀 Performance is excellent!")
        elif elapsed_ms < 5.0:
            print("   ✅ Performance is good!")
        else:
            print("   ⚠️  Performance is slower than expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all verification checks."""
    
    print("🔧 JAX C-GEM Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("JAX Functionality", check_jax_functionality),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("Input Data", check_input_data),
        ("Core Modules", check_core_modules),
        ("Performance Test", run_quick_performance_test)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        try:
            if check_function():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed unexpectedly: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ JAX C-GEM is ready to run!")
        print("\n💡 Next steps:")
        print("   🚀 Run a simulation: python main_ultra_performance.py")
        print("   📊 View results: python tools/plotting/show_results.py")
        return 0
        
    elif passed_checks >= total_checks - 2:
        print("⚠️  MINOR ISSUES DETECTED")
        print("JAX C-GEM should work but may have limited functionality.")
        print("\n💡 Consider fixing the issues above for full functionality.")
        return 0
        
    else:
        print("❌ MAJOR ISSUES DETECTED")
        print("JAX C-GEM may not work properly.")
        print("\n💡 Please fix the issues above before running simulations.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
