# Original C-GEM Compilation and Execution Guide

## Overview

The original C-GEM model is written in C and needs to be compiled before it can generate the CSV output files that are used for benchmark comparison with JAX C-GEM.

## Prerequisites

### Windows (Recommended: MSYS2/MinGW-w64)
1. **Install MSYS2**: Download from https://www.msys2.org/
2. **Install GCC**: In MSYS2 terminal, run:
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   ```
3. **Add to PATH**: Add `C:\msys64\mingw64\bin` to your Windows PATH

### Alternative Windows Options
- **Visual Studio**: Install "Desktop development with C++" workload
- **Dev-C++**: Lightweight IDE with built-in GCC
- **Code::Blocks**: Free C/C++ IDE

### Linux/WSL
```bash
sudo apt-get install gcc make
```

### macOS
```bash
xcode-select --install
```

## Quick Start

### Method 1: Using the Batch Script (Windows)
```cmd
cd "h:\My Drive\Project\TROPECOS\C-GEM\jax-C-GEM\deprecated\original-C-GEM"
compile_and_run.bat
```

### Method 2: Using Makefile (Cross-platform)
```bash
cd "h:\My Drive\Project\TROPECOS\C-GEM\jax-C-GEM\deprecated\original-C-GEM"
make run
```

### Method 3: Manual Compilation
```bash
# Compile
gcc -o c-gem *.c -lm -O2

# Create output directories
mkdir -p OUT/Hydrodynamics OUT/Flux OUT/Reaction

# Run simulation
./c-gem
```

## Expected Output Files

After successful execution, you should see these CSV files:

### Hydrodynamics (`OUT/Hydrodynamics/`)
- `U.csv` - Velocity
- `H.csv` - Free surface height
- `PROF.csv` - Water depth profile
- `tau_b.csv` - Bottom shear stress
- `B.csv` - Channel width
- `Chezy.csv` - Chezy roughness coefficient
- `FRIC.csv` - Friction coefficient
- `disp.csv` - Dispersion coefficient
- `windspeed.csv` - Wind speed
- `slope.csv` - Bottom slope
- `surface.csv` - Water surface elevation
- `elevation.csv` - Water elevation

### Water Quality (`OUT/`)
- `S.csv` - Salinity
- `O2.csv` - Dissolved oxygen
- `NO3.csv` - Nitrate
- `NH4.csv` - Ammonium
- `PO4.csv` - Phosphate
- `PHY1.csv` - Diatoms
- `PHY2.csv` - Non-diatoms
- `SI.csv` - Silica
- `TOC.csv` - Total organic carbon
- `SPM.csv` - Suspended particulate matter
- `DIC.csv` - Dissolved inorganic carbon
- `AT.csv` - Total alkalinity

### Fluxes (`OUT/Flux/`)
- `Advection_*.csv` - Advective fluxes for each species
- `Dispersion_*.csv` - Dispersive fluxes for each species

### Reaction Rates (`OUT/Reaction/`)
- Various biogeochemical reaction rate files

## Automatic File Cleanup

The original C-GEM automatically deletes previous output files at the start of each run (see `init.c` lines 25-28):

```c
system ("rm ./OUT/*.csv >y");
system ("rm ./OUT/Flux/*.csv >y");
system ("rm ./OUT/Hydrodynamics/*.csv >y");
system ("rm ./OUT/Reaction/*.csv >y");
```

This ensures fresh output for each simulation run.

## Integration with JAX C-GEM Benchmark

Once you have generated the original C-GEM CSV files, you can run the benchmark comparison:

```bash
cd "h:\My Drive\Project\TROPECOS\C-GEM\jax-C-GEM"
python tools/validation/benchmark_runner.py --mode full --c-gem-output-dir "deprecated/original-C-GEM/OUT"
```

## Troubleshooting

### Common Issues

1. **GCC not found**
   - Install MinGW-w64 or MSYS2
   - Add compiler to PATH

2. **Permission denied**
   - Run terminal as administrator
   - Check file permissions

3. **Missing input files**
   - Ensure INPUT/ directory structure exists
   - Check that CSV input files are present

4. **Compilation errors**
   - Check for missing header files
   - Ensure all `.c` files are in the same directory

### File Structure Check
```
original-C-GEM/
├── *.c files (source code)
├── *.h files (headers)
├── INPUT/ (input data directory)
└── OUT/ (generated after first run)
    ├── Hydrodynamics/
    ├── Flux/
    └── Reaction/
```

## Simulation Parameters

The simulation parameters are hardcoded in `define.h`:
- **Duration**: 10 days simulation + 5 days warmup
- **Time step**: 180 seconds
- **Spatial step**: 2000 meters
- **Domain length**: 202 km
- **Tidal amplitude**: 4.43 m

These match the equivalent JAX C-GEM configuration for direct comparison.

## Performance Notes

- **Execution time**: ~5-30 seconds depending on system
- **Output size**: ~50-100 MB of CSV files
- **Memory usage**: ~10-50 MB during execution

## Next Steps

After successful compilation and execution:
1. ✅ CSV files generated in `OUT/` directory
2. ✅ Ready for benchmark comparison with JAX C-GEM
3. ✅ Can run visualization scripts (`plotFigure2.py`, `plotWaterQuality2.py`)
4. ✅ Can proceed with comprehensive model validation
