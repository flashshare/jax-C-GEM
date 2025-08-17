#!/usr/bin/env python3
"""
JAX C-GEM Transport Physics Diagnostic Suite
===========================================

This script coordinates the execution of all diagnostic tests to help identify
the root cause of the salinity gradient inversion issue.

The suite performs systematic testing of:
1. Coordinate system verification
2. Boundary condition application
3. Initial salinity profile
4. Transport evolution over time
5. Execution flow comparison
6. Boundary persistence

Each test generates detailed output and visualizations to diagnose the issue.
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def run_diagnostic_suite():
    """Run the complete diagnostic test suite."""
    
    start_time = time.time()
    
    print("=" * 80)
    print("JAX C-GEM TRANSPORT PHYSICS DIAGNOSTIC SUITE")
    print("=" * 80)
    print()
    
    # Ensure output directory exists
    os.makedirs('OUT/diagnostics', exist_ok=True)
    
    # Setup HTML report file
    report_file = open('OUT/diagnostics/diagnostic_report.html', 'w')
    write_report_header(report_file)
    
    # Run each test and record results
    test_results = {}
    
    # Test 1: Coordinate System Verification
    print("\n\nRunning Test 1: Coordinate System Verification...")
    try:
        from test1_coordinate_system import analyze_coordinates
        test_results['test1'] = analyze_coordinates()
        write_test_result(report_file, 1, "Coordinate System Verification", True)
    except Exception as e:
        print(f"Error in Test 1: {e}")
        test_results['test1'] = {'error': str(e)}
        write_test_result(report_file, 1, "Coordinate System Verification", False, error=str(e))
    
    # Test 2: Boundary Condition Application
    print("\n\nRunning Test 2: Boundary Condition Application...")
    try:
        from test2_boundary_application import test_boundary_application
        test_results['test2'] = test_boundary_application()
        write_test_result(report_file, 2, "Boundary Condition Application", True)
    except Exception as e:
        print(f"Error in Test 2: {e}")
        test_results['test2'] = {'error': str(e)}
        write_test_result(report_file, 2, "Boundary Condition Application", False, error=str(e))
    
    # Test 3: Initial Salinity Profile
    print("\n\nRunning Test 3: Initial Salinity Profile...")
    try:
        from test3_initial_salinity import test_initial_salinity
        test_results['test3'] = test_initial_salinity()
        write_test_result(report_file, 3, "Initial Salinity Profile", True)
    except Exception as e:
        print(f"Error in Test 3: {e}")
        test_results['test3'] = {'error': str(e)}
        write_test_result(report_file, 3, "Initial Salinity Profile", False, error=str(e))
    
    # Test 4: Transport Evolution Test
    print("\n\nRunning Test 4: Transport Evolution Test...")
    try:
        from test4_transport_evolution import run_short_transport_simulation
        test_results['test4'] = run_short_transport_simulation()
        write_test_result(report_file, 4, "Transport Evolution Test", True)
    except Exception as e:
        print(f"Error in Test 4: {e}")
        test_results['test4'] = {'error': str(e)}
        write_test_result(report_file, 4, "Transport Evolution Test", False, error=str(e))
    
    # Test 5: Execution Flow Analysis
    print("\n\nRunning Test 5: Execution Flow Analysis...")
    try:
        from test5_execution_flow import analyze_execution_flow
        test_results['test5'] = analyze_execution_flow()
        write_test_result(report_file, 5, "Execution Flow Analysis", True)
    except Exception as e:
        print(f"Error in Test 5: {e}")
        test_results['test5'] = {'error': str(e)}
        write_test_result(report_file, 5, "Execution Flow Analysis", False, error=str(e))
    
    # Finish HTML report
    elapsed_time = time.time() - start_time
    write_report_summary(report_file, test_results, elapsed_time)
    write_report_footer(report_file)
    report_file.close()
    
    print("\n\n" + "=" * 80)
    print(f"DIAGNOSTIC SUITE COMPLETED IN {elapsed_time:.2f} SECONDS")
    print("=" * 80)
    print(f"\nDetailed report saved to: OUT/diagnostics/diagnostic_report.html")
    
    # Analyze results and provide recommendations
    analyze_results(test_results)
    
    return test_results

def write_report_header(file):
    """Write the HTML report header."""
    
    header = """<!DOCTYPE html>
<html>
<head>
    <title>JAX C-GEM Transport Diagnostic Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        .test-result { margin: 15px 0; padding: 10px; border-radius: 5px; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeeba; }
        img { max-width: 100%; margin: 20px 0; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        code { background-color: #f8f9fa; padding: 2px 5px; border-radius: 3px; }
        .summary { margin-top: 40px; padding: 15px; background-color: #e9ecef; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>JAX C-GEM Transport Physics Diagnostic Report</h1>
    <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <hr>
    """
    
    file.write(header)

def write_test_result(file, test_num, test_name, success, error=None):
    """Write test result to the HTML report."""
    
    result_class = "success" if success else "error"
    status = "SUCCESS" if success else "ERROR"
    
    content = f"""
    <h2>Test {test_num}: {test_name}</h2>
    <div class="test-result {result_class}">
        <strong>Status: {status}</strong>
        {f"<p>Error: {error}</p>" if error else ""}
    </div>
    """
    
    if success:
        content += f"""
        <p>For detailed results, see the test output above.</p>
        <img src="test{test_num}_{'_'.join(test_name.lower().split())}.png" alt="Test {test_num} Results">
        """
    
    file.write(content)

def write_report_summary(file, results, elapsed_time):
    """Write summary of all test results to the HTML report."""
    
    success_count = sum(1 for r in results.values() if 'error' not in r)
    error_count = len(results) - success_count
    
    summary = f"""
    <div class="summary">
        <h2>Test Suite Summary</h2>
        <p><strong>Total tests:</strong> {len(results)}</p>
        <p><strong>Successful:</strong> {success_count}</p>
        <p><strong>Failed:</strong> {error_count}</p>
        <p><strong>Elapsed time:</strong> {elapsed_time:.2f} seconds</p>
    </div>
    """
    
    file.write(summary)

def write_report_footer(file):
    """Write the HTML report footer."""
    
    footer = """
    <hr>
    <p><em>JAX C-GEM Diagnostic Suite - © Nguyen Truong An</em></p>
</body>
</html>
    """
    
    file.write(footer)

def analyze_results(results):
    """Analyze test results and provide recommendations."""
    
    print("\n\nDIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Check for specific issues in the results
    issues = []
    
    # Check coordinate system issues
    if 'test1' in results and 'error' not in results['test1']:
        test1 = results['test1']
        if test1.get('jax_total_length') != test1.get('c_gem_total_length'):
            issues.append("Coordinate system length mismatch between JAX and C-GEM implementations")
    
    # Check boundary application issues
    if 'test2' in results and 'error' not in results['test2']:
        # Add specific checks based on test2 results
        pass
    
    # Check initial salinity issues
    if 'test3' in results and 'error' not in results['test3']:
        test3 = results['test3']
        if test3.get('actual_initialization') is not None:
            # Check if initial gradient is inverted
            mouth_sal = test3['actual_initialization'][0]
            head_sal = test3['actual_initialization'][-1]
            if mouth_sal < head_sal:
                issues.append("Initial salinity gradient is inverted (low at mouth, high at head)")
    
    # Check transport evolution issues
    if 'test4' in results and 'error' not in results['test4']:
        # Add specific checks based on test4 results
        pass
    
    # Check execution flow issues
    if 'test5' in results and 'error' not in results['test5']:
        # Add specific checks based on test5 results
        pass
    
    # Print identified issues
    if issues:
        print("The following issues were identified:")
        for i, issue in enumerate(issues):
            print(f"{i+1}. {issue}")
    else:
        print("No specific issues were identified from the test results.")
    
    # Provide general recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Check that the coordinate system is consistent between implementations")
    print("2. Verify that boundary conditions are applied correctly at the right indices")
    print("3. Ensure initial salinity gradient is set correctly (high at mouth, low at head)")
    print("4. Compare the order of operations between C-GEM and JAX implementations")
    print("5. Check for index offsets or off-by-one errors in array accesses")
    print("6. Verify that velocity is checked at the correct indices for boundary conditions")

def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='JAX C-GEM Transport Physics Diagnostic Suite')
    parser.add_argument('--test', type=int, help='Run only a specific test (1-5)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.test:
        print(f"Running only test {args.test}...")
        # Import and run specific test
        if args.test == 1:
            from test1_coordinate_system import analyze_coordinates
            analyze_coordinates()
        elif args.test == 2:
            from test2_boundary_application import test_boundary_application
            test_boundary_application()
        elif args.test == 3:
            from test3_initial_salinity import test_initial_salinity
            test_initial_salinity()
        elif args.test == 4:
            from test4_transport_evolution import run_short_transport_simulation
            run_short_transport_simulation()
        elif args.test == 5:
            from test5_execution_flow import analyze_execution_flow
            analyze_execution_flow()
        else:
            print(f"Invalid test number: {args.test}")
    else:
        # Run full suite
        run_diagnostic_suite()