#!/usr/bin/env python3
"""
Run Transport Physics Tests

Simple command-line tool to run the diagnostic tests for finding 
the root cause of the salinity gradient inversion issue.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_test(test_num, verbose=False):
    """Run a specific diagnostic test."""
    
    # Dictionary of test files and their descriptions
    tests = {
        1: ("test1_coordinate_system.py", "Coordinate System Verification"),
        2: ("test2_boundary_application.py", "Boundary Condition Application"),
        3: ("test3_initial_salinity.py", "Initial Salinity Profile"),
        4: ("test4_transport_evolution.py", "Transport Evolution Test"),
        5: ("test5_execution_flow.py", "Execution Flow Analysis"),
        7: ("test7_boundary_mapping.py", "Forcing Data and Boundary Mapping")
    }
    
    if test_num not in tests:
        print(f"Error: Test {test_num} not found")
        return False
    
    test_file, test_name = tests[test_num]
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"Error: Test file {test_path} not found")
        return False
    
    print(f"Running Test {test_num}: {test_name}")
    print("=" * 80)
    
    # Create the output directory if it doesn't exist
    os.makedirs('OUT/diagnostics', exist_ok=True)
    
    # Construct the command
    cmd = [sys.executable, str(test_path)]
    
    try:
        # Run the test
        if verbose:
            # Run with output visible
            subprocess.run(cmd, check=True)
        else:
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        
        print(f"\nTest {test_num} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\nTest {test_num} failed with error code {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False

def run_full_suite(verbose=False):
    """Run the full diagnostic test suite."""
    
    print("Running Full Diagnostic Test Suite")
    print("=" * 80)
    
    # Create the output directory if it doesn't exist
    os.makedirs('OUT/diagnostics', exist_ok=True)
    
    # Run the diagnostic suite runner
    suite_runner = Path(__file__).parent / "run_diagnostics.py"
    
    if not suite_runner.exists():
        print(f"Error: Suite runner {suite_runner} not found")
        return False
    
    try:
        # Run the suite
        cmd = [sys.executable, str(suite_runner)]
        if verbose:
            # Run with output visible
            subprocess.run(cmd, check=True)
        else:
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        
        print("\nFull test suite completed successfully")
        print(f"Report available at: {os.path.abspath('OUT/diagnostics/diagnostic_report.html')}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\nTest suite failed with error code {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Run Transport Physics Diagnostic Tests')
    parser.add_argument('--test', '-t', type=int, choices=[1, 2, 3, 4, 5, 7],
                        help='Run a specific test (1-5, 7)')
    parser.add_argument('--all', '-a', action='store_true', 
                        help='Run the full diagnostic suite')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    if not args.test and not args.all:
        parser.print_help()
        print("\nAvailable tests:")
        print("  1: Coordinate System Verification")
        print("  2: Boundary Condition Application")
        print("  3: Initial Salinity Profile")
        print("  4: Transport Evolution Test")
        print("  5: Execution Flow Analysis")
        print("  7: Forcing Data and Boundary Mapping")
        print("Use --all to run the full suite")
        return 1
    
    if args.test:
        success = run_test(args.test, args.verbose)
    elif args.all:
        success = run_full_suite(args.verbose)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())