# Transport Physics Diagnostic Tests

This directory contains a comprehensive suite of diagnostic tests designed to identify the root cause of the salinity gradient inversion issue in the JAX C-GEM implementation.

## Background

The JAX C-GEM implementation currently shows an inverted salinity gradient compared to the original C-GEM code. Specifically:

- **Original C-GEM**: Shows the correct gradient with high salinity at the mouth/ocean (28.19 PSU) decreasing to low salinity at the head/river (0.09 PSU).
- **JAX C-GEM**: Shows an inverted gradient with low salinity at the mouth/ocean (0.37 PSU) increasing to high salinity at the head/river (31.43 PSU).

This inversion persists despite implementing the C-GEM transport physics (velocity-dependent boundary conditions and TVD advection scheme).

## Diagnostic Approach

These tests systematically isolate different aspects of the transport implementation to identify the root cause:

1. **Coordinate System Verification**: Checks if the JAX and C-GEM implementations use the same coordinate system (ocean at index 0, river at index M-1).

2. **Boundary Condition Application**: Verifies if boundary conditions are correctly applied based on velocity direction.

3. **Initial Salinity Profile**: Examines how salinity is initialized to check for incorrect gradient direction.

4. **Transport Evolution Test**: Tracks salinity changes over a short simulation to identify where inversion occurs.

5. **Execution Flow Analysis**: Compares the order of operations between JAX and C-GEM implementations.

7. **Forcing Data and Boundary Mapping**: Examines how boundary data is loaded, mapped, and applied to identify potential inversions.

## Running the Tests

### Prerequisites

Ensure that you have:
- Python 3.8+
- JAX and other dependencies installed
- The JAX C-GEM codebase properly configured

### Using the Test Runner

The `run_tests.py` script provides a convenient way to run individual tests or the full suite:

```bash
# Run a specific test (e.g., test 1)
python tools/diagnostics/run_tests.py --test 1

# Run the full suite
python tools/diagnostics/run_tests.py --all

# Run with verbose output
python tools/diagnostics/run_tests.py --all --verbose
```

### Running Individual Tests

You can also run the individual test scripts directly:

```bash
python tools/diagnostics/test1_coordinate_system.py
python tools/diagnostics/test2_boundary_application.py
# ... and so on
```

## Test Outputs

All tests generate:
1. Console output with detailed findings
2. Visualization plots in the `OUT/diagnostics/` directory
3. A comprehensive HTML report (for the full suite)

## Interpreting Results

The diagnostic tests will help identify if the issue is related to:

- Coordinate system mismatches
- Boundary condition application logic
- Initial condition setup
- Order of operations in transport calculations
- Data loading and mapping issues

## Adding New Tests

To add a new test:
1. Create a new test file `testX_descriptive_name.py`
2. Follow the pattern of existing tests
3. Update the `run_tests.py` script to include your new test

## Common Findings

Some common issues that might be identified:

1. **Coordinate System Mismatch**: If JAX C-GEM uses a different coordinate system than original C-GEM (e.g., river at index 0 instead of ocean).

2. **Boundary Application Logic**: If boundary conditions are applied incorrectly or at wrong indices.

3. **Initial Conditions**: If the initial salinity profile is set with an incorrect gradient.

4. **Velocity Direction Handling**: If the velocity direction check for boundary conditions is incorrect.

5. **Data Mapping**: If input data is mapped to incorrect indices.

## Reporting Issues

When reporting the findings, include:
- Which test identified the issue
- Specific visualizations or outputs that show the problem
- Recommended solution approach