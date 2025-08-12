# JAX C-GEM: Comprehensive User Guide

**How to Set Up and Run the JAX C-GEM Tidal Estuary Model for Your Own System**

---

## 🎯 Quick Start (5 Minutes)

### Fastest Model Run
```bash
# 1. Navigate to project directory
cd /path/to/jax-C-GEM

# 2. Run ultra-performance simulation (30,000+ steps/min)
python main_ultra_performance.py

# 3. View results (automatically created, or manually)
python tools/plotting/show_results.py
```

### Using VS Code Tasks (Recommended)
1. Open VS Code in the project directory
2. Press `Ctrl+Shift+P` → "Tasks: Run Task"
3. Select `⚡ Maximum Performance Mode` for fastest execution
4. **Results automatically displayed** - or use `📊 Comprehensive Results Viewer` for detailed analysis

---

## 📚 Table of Contents

1. [Model Overview](#model-overview)
2. [Installation & Setup](#installation--setup)
3. [Understanding the Model](#understanding-the-model)
4. [Setting Up Your Own Estuary](#setting-up-your-own-estuary)
5. [Input Data Requirements](#input-data-requirements)
6. [Configuration Files](#configuration-files)
7. [Running Simulations](#running-simulations)
8. [Validation & Analysis](#validation--analysis)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## 🌊 Model Overview

JAX C-GEM is a high-performance 1D tidal estuary model that simulates:

- **🌊 Hydrodynamics**: Water levels, velocities, tidal propagation
- **🧪 Transport**: Advection-dispersion of 17 chemical species
- **🦠 Biogeochemistry**: Complete reactive network including:
  - Nutrients: NH4, NO3, PO4, Si
  - Biology: Phytoplankton (2 species), organic matter
  - Chemistry: pH, carbonate system, dissolved oxygen
  - Physics: Salinity, suspended matter

### 🚀 Performance Highlights
- **Ultra-fast**: 30,000+ simulation steps/minute
- **Memory efficient**: Handles long-term simulations (years)
- **Scientific accuracy**: 100% identical to original C-GEM results
- **Modern tools**: Built with JAX for gradient-based optimization

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python**: 3.8+ (3.11 recommended)
- **OS**: Windows 10/11 or Linux
- **Memory**: 8GB minimum (16GB for multi-year simulations)
- **CPU**: Any modern processor (no GPU required)

### Quick Installation
```bash
# Clone the repository
git clone [repository-url]
cd jax-C-GEM

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import jax; print('✅ JAX C-GEM ready!')"
```

### VS Code Setup (Recommended)
1. Install VS Code with Python extension
2. Open project folder: `File → Open Folder → jax-C-GEM`
3. Install recommended extensions (will be prompted)
4. Use integrated tasks: `Ctrl+Shift+P → Tasks: Run Task`

---

## 🎓 Understanding the Model

### What Does JAX C-GEM Simulate?

**JAX C-GEM simulates how tides, rivers, and chemistry interact in an estuary.**

Think of an estuary as a mixing zone where:
- 🌊 **Tides** push salt water upstream twice daily
- 🏞️ **Rivers** push fresh water downstream continuously  
- 🦠 **Biology** consumes and produces chemicals
- ⚖️ **Chemistry** maintains pH, oxygen, nutrient balance

### Model Domain Setup

```
🏞️ UPSTREAM               🌊 DOWNSTREAM
River → → → → → → → → → → → → Ocean
|─────── 200km estuary ─────────|
0km                          200km
(Fresh water)              (Salt water)
```

**Grid Structure:**
- **Length**: Customizable (default: 200km)
- **Resolution**: 2km grid spacing (101 points)
- **Time steps**: 3-minute intervals for accurate tidal dynamics

---

## 🏗️ Setting Up Your Own Estuary

### Step 1: Define Your Estuary Geometry

Edit `config/model_config.txt`:

```plaintext
# =====================
# ESTUARY GEOMETRY
# =====================
EL = 150000              # Estuary length [m] (150km example)
DELXI = 1500            # Grid spacing [m] (1.5km resolution)
AMPL = 3.2              # Tidal amplitude [m] (your location's tide)
Q_AVAIL = 180           # River discharge [m³/s] (your river flow)

# =====================  
# SIMULATION PERIOD
# =====================
MAXT = 365              # Simulation days (1 year)
WARMUP = 30             # Warmup period [days]
simulation_start_date = "2020-01-01"  # Your data period
```

**💡 Key Parameters to Customize:**
- **EL**: Your estuary length from mouth to tidal limit
- **AMPL**: Mean tidal range at your estuary mouth
- **Q_AVAIL**: Mean river discharge during your study period

### Step 2: Prepare Boundary Condition Data

Create files in `INPUT/Boundary/`:

#### **A) Tidal Boundary (Required)**
`INPUT/Boundary/UB/tidal_elevation.csv`:
```csv
time_seconds,elevation_m
0.0,0.0
21600.0,2.1
43200.0,0.0  
64800.0,-2.1
86400.0,0.0
```
*Time in seconds from simulation start, elevation relative to mean sea level*

#### **B) River Boundary (Required)**  
`INPUT/Boundary/LB/river_discharge.csv`:
```csv
time_seconds,discharge_m3s,temperature_C,salinity_psu
0.0,180.0,25.0,0.1
86400.0,190.0,25.2,0.1
172800.0,175.0,24.8,0.1
```

#### **C) Wind Forcing (Optional)**
`INPUT/Boundary/wind.csv`:
```csv
time_seconds,wind_speed_ms,wind_direction_deg
0.0,3.2,45.0
3600.0,3.5,50.0
7200.0,3.1,40.0
```

### Step 3: Configure Input Data Sources

Edit `config/input_data_config.txt`:

```plaintext
# =====================
# BOUNDARY CONDITIONS
# =====================
name=upstream_boundary
type=boundary
location=upstream
file_path=INPUT/Boundary/LB/river_discharge.csv
variables=discharge,temperature,salinity
data_type=time_series

name=downstream_boundary  
type=boundary
location=downstream
file_path=INPUT/Boundary/UB/tidal_elevation.csv
variables=elevation
data_type=time_series

# =====================
# TRIBUTARIES (Optional)
# =====================
name=major_tributary
type=tributary
location_km=75.0
file_path=INPUT/Tributaries/tributary_flow.csv
variables=discharge,temperature,nutrients
```

### Step 4: Prepare Validation Data (Optional)

Create field observation files in `INPUT/Calibration/`:

#### **Field Data Format**
`INPUT/Calibration/field_observations.csv`:
```csv
Date,Site,Location,Salinity,DO (mg/L),NO3 (mgN/L),NH4 (mgN/L)
1/15/2020,Station1,10km,12.5,8.2,0.15,0.08
2/20/2020,Station1,10km,11.8,7.9,0.18,0.12
1/15/2020,Station2,50km,5.2,6.8,0.25,0.15
```

**Required columns:**
- **Date**: MM/DD/YYYY format
- **Site**: Station identifier  
- **Location**: Distance from mouth (for mapping to model grid)
- **Variables**: Any combination of modeled species

---

## 📋 Configuration Files Reference

### Model Configuration (`config/model_config.txt`)

```plaintext
# =====================
# ESSENTIAL PARAMETERS
# =====================
EL = 200000              # Estuary length [m]
DELXI = 2000            # Grid spacing [m] 
MAXT = 365              # Simulation days
WARMUP = 100            # Warmup period [days]
DELTI = 180             # Time step [seconds]
AMPL = 4.43             # Tidal amplitude [m]
Q_AVAIL = 250           # Default discharge [m³/s]

# =====================
# ADVANCED PHYSICS
# =====================
# Transport segments (for varying dispersion)
num_segments = 2        # Number of estuary zones

# Segment 1: Marine zone (mouth to mid-estuary)
index_1 = 0             # Starting grid point
B1 = 3887.00           # Width [m]
LC1 = 65500            # Convergence length [m]
Chezy1 = 65            # Roughness coefficient
Rs1 = 1.0              # Storage ratio

# Segment 2: River zone (mid-estuary to head)
index_2 = 50            # Starting grid point  
B2 = 1887.00           # Width [m]
LC2 = 122500           # Convergence length [m]
Chezy2 = 45            # Roughness coefficient
Rs2 = 1.0              # Storage ratio

# =====================
# BIOGEOCHEMISTRY
# =====================
# All biogeochemical parameters are in the config file
# Modify these for calibration to your estuary
```

### Input Data Configuration (`config/input_data_config.txt`)

This file defines all external data sources:

```plaintext
# Boundary conditions
name=upstream_discharge
type=boundary
location=upstream
file_path=INPUT/Boundary/LB/discharge.csv

name=downstream_elevation
type=boundary  
location=downstream
file_path=INPUT/Boundary/UB/tidal.csv

# Tributaries
name=tributary1
type=tributary
location_km=45.0
file_path=INPUT/Tributaries/trib1_flow.csv

# Forcing data
name=wind
type=forcing
file_path=INPUT/Boundary/wind.csv
```

---

## 🚀 Running Simulations

### Basic Simulation Run

```bash
# Standard run (good for debugging)
python src/main.py --mode run --output-format npz

# Ultra-performance run (fastest)
python main_ultra_performance.py

# With validation against field data
python src/main.py --mode run --output-format npz
python tools/validation/validate_against_field_data.py
```

### VS Code Tasks (Recommended)

Press `Ctrl+Shift+P` → `Tasks: Run Task`:

- **🚀 Run Model**: Standard simulation
- **⚡ Maximum Performance Mode**: Ultra-fast simulation
- **🔬 Run with Physics Validation**: With detailed checking
- **🌊 Field Data Validation**: Compare against observations

### Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  --mode {run}              # Execution mode
  --output-format {csv,npz,auto}  # Output format
  --no-physics-check        # Skip validation for speed
  --debug                   # Detailed debugging output
  --config CONFIG_FILE      # Custom configuration file
```

### Performance Tips

**For Maximum Speed:**
1. Use `main_ultra_performance.py` script
2. Set `--output-format npz` (faster than CSV)
3. Use `--no-physics-check` flag
4. Run on SSD storage for faster I/O

**Expected Performance:**
- **Standard mode**: ~12,000-15,000 steps/minute
- **Ultra-performance**: ~25,000-35,000 steps/minute
- **1-year simulation**: ~5-15 minutes depending on mode

---

## 🔬 Validation & Analysis

### Automatic Validation

```bash
# Validate against field data
python tools/validation/validate_against_field_data.py

# Generate publication plots  
python tools/plotting/publication_plots.py

# Create comprehensive summary plots (auto-runs after simulation)
python tools/plotting/show_results.py
```

### Validation Reports

The model generates comprehensive validation including:

- **Statistical Metrics**: RMSE, correlation, bias
- **Visual Comparisons**: Observed vs predicted time series
- **Longitudinal Profiles**: Spatial distribution patterns
- **Physics Checks**: Mass balance, stability, realistic ranges

### Understanding Outputs

**Simulation creates these files:**
- `OUT/complete_simulation_results.npz`: Complete model output
- `OUT/validation_results/`: Validation plots and statistics
- `OUT/Summary_Plots/`: Quick visualization summaries

**Key output variables:**
- **H**: Water surface elevation [m]
- **U**: Flow velocity [m/s]  
- **S**: Salinity [PSU]
- **O2**: Dissolved oxygen [mmol/m³]
- **NO3, NH4, PO4**: Nutrients [mmol/m³]
- **PHY1, PHY2**: Phytoplankton [mmol/m³]

---

## ⚙️ Advanced Features

### Gradient-Based Calibration

JAX C-GEM supports automatic parameter optimization:

```bash
# Run parameter calibration
python tools/calibration/run_calibration.py

# Sensitivity analysis
python tools/analysis/sensitivity_analysis.py
```

### Performance Profiling

```bash
# Detailed performance analysis
python tools/validation/comprehensive_cgem_benchmark.py
```

### Multi-Year Simulations

For long-term runs, use NPZ format and adjust memory settings:

```plaintext
# In model_config.txt
MAXT = 1095             # 3 years
memory_fraction = 0.9   # Use more RAM
```

---

## ❓ Troubleshooting

### Common Issues

**1. "JAX import failed"**
```bash
# Install JAX CPU version
pip install jax[cpu] --upgrade
```

**2. "Memory error during simulation"**
```bash
# Reduce memory usage
# In model_config.txt:
memory_fraction = 0.6
# Or use shorter simulation period
```

**3. "No field data found for validation"**
- Check field data file format (CSV with Date column)
- Verify file paths in `config/input_data_config.txt`
- Ensure date formats match (MM/DD/YYYY)

**4. "Simulation unstable/unrealistic results"**
- Check boundary condition data for gaps or extreme values
- Verify estuary geometry parameters (width, depth)
- Reduce time step: `DELTI = 120` (2 minutes)

### Getting Help

1. **Check logs**: Look for error messages in console output
2. **Validate setup**: Run `python tools/validation/verify_setup.py`
3. **Physics check**: Use `--debug` flag for detailed diagnostics
4. **Compare with defaults**: Test with provided Saigon River example

### Performance Diagnostics

```bash
# Check model performance
python src/performance_diagnostic.py

# Memory usage analysis
python tools/analysis/memory_profiling.py
```

---

## 📖 Example: Complete Setup Workflow

### Setting Up a New Estuary (Thames Example)

```bash
# 1. Configure estuary geometry
# Edit config/model_config.txt:
EL = 180000              # Thames estuary ~180km
AMPL = 3.8              # Thames tidal range ~3.8m
Q_AVAIL = 65            # Thames mean flow ~65 m³/s

# 2. Prepare tidal data
# Create INPUT/Boundary/UB/thames_tide.csv with hourly tidal data

# 3. Prepare river data  
# Create INPUT/Boundary/LB/thames_river.csv with daily flow data

# 4. Run simulation
python main_ultra_performance.py

# 5. Validate (if field data available)
# Add observations to INPUT/Calibration/thames_observations.csv
python tools/validation/validate_against_field_data.py

# 6. Generate comprehensive analysis plots (auto-generated after simulation)
python tools/plotting/show_results.py --detailed
```

### Results Analysis

After simulation, you'll have:
- **Hydrodynamic patterns**: Tidal propagation, flow velocities
- **Salt intrusion**: How far salt penetrates upstream
- **Oxygen dynamics**: Dissolved oxygen patterns and potential hypoxia
- **Nutrient cycling**: Nitrogen and phosphorus transformations
- **Biological productivity**: Phytoplankton growth patterns

---

## 🎯 Next Steps

Once you have JAX C-GEM running for your estuary:

1. **Calibrate parameters** using field observations
2. **Run scenarios** (climate change, management options)
3. **Analyze results** with built-in statistical tools
4. **Generate publications** using high-quality plotting tools
5. **Extend model** with custom biogeochemical processes

**Happy modeling! 🌊🧪📊**

### Method 1: Direct Python Execution

#### Basic Run (Fastest)
```bash
python src/main.py --mode run --output-format csv --no-physics-check
```

#### Run with Physics Validation
```bash
python src/main.py --mode run --output-format csv --physics-check
```

#### Debug Mode
```bash
python src/main.py --mode run --output-format csv --debug --physics-check
```

#### High-Performance Mode
```bash
python src/main_optimized.py --mode run --no-physics-check
```

### Method 2: VS Code Tasks (Recommended)

Open Command Palette (`Ctrl+Shift+P`) and select "Tasks: Run Task":

- **🚀 Run Model** - Maximum speed execution
- **🔬 Run with Physics Validation** - With physics checks
- **🐛 Debug Mode** - Full debugging and validation
- **🎨 Generate Publication Figures** - Create plots after run

### Method 3: Tools CLI
```bash
# Unified command interface
python tools/tools_cli.py validation --benchmark
python tools/tools_cli.py visualization --plot-profiles
python tools/tools_cli.py calibration --optimize
```

---

## ⚙️ Command Line Options

### Main Script Options (`src/main.py`)

| Option | Values | Default | Description |
|--------|---------|---------|-------------|
| `--mode` | `run`, `calibrate`, `validate` | `run` | Execution mode |
| `--output-format` | `csv`, `npz` | `csv` | Output file format |
| `--physics-check` | flag | disabled | Enable physics validation |
| `--no-physics-check` | flag | enabled | Disable physics validation (faster) |
| `--debug` | flag | disabled | Enable debug output |
| `--config` | path | `config/model_config.txt` | Model configuration file |
| `--data-config` | path | `config/input_data_config.txt` | Data configuration file |

### High-Performance Script Options (`src/main_optimized.py`)

| Option | Values | Default | Description |
|--------|---------|---------|-------------|
| `--mode` | `run`, `optimize`, `validate` | `run` | Execution mode |
| `--output-format` | `csv`, `npz` | `csv` | Output file format |
| `--no-physics-check` | flag | enabled | Maximum performance |

### Usage Examples
```bash
# Standard run with CSV output
python src/main.py --mode run --output-format csv --physics-check

# Fast run without validation
python src/main.py --mode run --no-physics-check

# Debug run with full validation
python src/main.py --mode run --debug --physics-check

# High-performance mode
python src/main_optimized.py --mode run --no-physics-check

# Custom configuration
python src/main.py --mode run --config my_config.txt --output-format npz
```

---

## 🎮 VS Code Tasks

The project includes pre-configured VS Code tasks for common operations:

### Essential Execution Tasks
- **🚀 Run Model** - Fast execution without physics validation
- **🔬 Run with Physics Validation** - Complete validation enabled
- **🐛 Debug Mode** - Full debugging and physics validation

### Visualization Tasks
- **🎨 Generate Publication Figures** - Create publication-quality plots

### Validation Tasks
- **🏆 Comprehensive C-GEM vs JAX Benchmark** - Complete performance comparison
- **🌊 Field Data Validation** - Validate against CARE, CEM observations
- **🔧 Verify C-GEM Setup** - Check system configuration

### System Utilities
- **🧹 Clean Output Directory** - Clear previous results
- **📦 Install Requirements** - Update Python dependencies

### Running Tasks
1. Open Command Palette: `Ctrl+Shift+P`
2. Type: "Tasks: Run Task"
3. Select desired task from the list

---

## 📄 Configuration Files

### Model Configuration (`config/model_config.txt`)

Key parameters you can modify:

```plaintext
# Simulation Time
MAXT = 864000         # Total simulation time [seconds] (10 days)
WARMUP = 432000       # Warm-up period [seconds] (5 days)
DELTI = 180           # Time step [seconds]
TS = 10               # Output save frequency

# Spatial Domain
DELXI = 2000          # Spatial step [meters]
EL = 202000           # Estuarine length [meters]

# Boundary Conditions
AMPL = 4.43           # Tidal amplitude [meters]
Q_AVAIL = 250         # River discharge [m³/s]

# Performance Settings
platform_mode = "cpu"        # Force CPU execution
jax_threads = 4               # Number of CPU threads
memory_fraction = 0.8         # Memory usage fraction
```

### Data Configuration (`config/input_data_config.txt`)

Specifies input data file paths:
- Boundary conditions (tidal elevation, river discharge)
- Forcing data (wind, temperature)
- Tributary inputs
- Initial conditions

### Key Configuration Tips
- **Faster runs**: Increase `DELTI` (time step)
- **Higher resolution**: Decrease `DELXI` (spatial step)
- **Shorter simulation**: Decrease `MAXT`
- **Memory issues**: Reduce `memory_fraction`

---

## 📊 Output Files

### Standard Output (`OUT/` directory)

#### Hydrodynamics (12 files)
- `H.csv` - Water surface elevation
- `U.csv` - Velocity
- `Q.csv` - Discharge
- `PROF.csv` - River bed profile
- `tau_b.csv` - Bottom shear stress
- `B.csv` - River width
- `Chezy.csv` - Chezy coefficient
- `FRIC.csv` - Friction coefficient
- `disp.csv` - Dispersion coefficient
- `windspeed.csv` - Wind speed
- `slope.csv` - Water surface slope
- `elevation.csv` - Water elevation

#### Transport/Chemistry (17 files)
- `S.csv` - Salinity
- `O2.csv` - Dissolved oxygen
- `NH4.csv`, `NO3.csv`, `PO4.csv` - Nutrients
- `PHY1.csv`, `PHY2.csv` - Phytoplankton
- `SI.csv` - Silica
- `TOC.csv` - Total organic carbon
- `SPM.csv` - Suspended particulate matter
- `DIC.csv` - Dissolved inorganic carbon
- `AT.csv` - Total alkalinity
- `HS.csv` - Hydrogen sulfide
- `PH.csv` - pH
- `ALKC.csv` - Alkalinity from carbonate
- `CO2.csv` - Carbon dioxide
- `PIP.csv` - Particulate inorganic phosphorus

#### Biogeochemical Rates (14 files)
- `NPP.csv` - Net primary production
- `Si_consumption.csv` - Silica consumption
- `NPP_NO3.csv`, `NPP_NH4.csv` - Nutrient-specific production
- `phydeath.csv` - Phytoplankton mortality
- `adegrad.csv` - Aerobic degradation
- `denit.csv` - Denitrification
- `nitrif.csv` - Nitrification
- `o2air.csv` - Oxygen air-sea exchange
- `sorption.csv` - Phosphate sorption
- `eross.csv` - Erosion
- `deps.csv` - Deposition
- `integral.csv` - Integrated rates
- `nlim.csv` - Nutrient limitation

### Output Format Options

#### CSV Format (`--output-format csv`)
- Human-readable text files
- Easy to import into Excel, MATLAB, R
- Compatible with original C-GEM analysis tools

#### NPZ Format (`--output-format npz`)
- Compressed NumPy binary format
- Faster I/O and smaller file sizes
- Python-specific format

---

## 🔬 Advanced Features

### Model Validation and Benchmarking

#### Comprehensive Benchmark
```bash
# Compare JAX C-GEM vs Original C-GEM
python tools/validation/comprehensive_cgem_benchmark.py
```
**Generates**: Performance comparison, accuracy analysis, field data validation

#### Field Data Validation
```bash
# Validate against observations
python tools/validation/validate_against_field_data.py
```
**Uses**: CARE, CEM, and tidal range datasets from `INPUT/Calibration/`

### Performance Optimization

#### High-Performance Mode
```bash
# Use optimized simulation engine
python src/main_optimized.py --mode run --no-physics-check
```
**Benefits**: 
- Pre-computed forcing data
- Eliminated interpolation overhead
- Pure JAX operations
- Target: 15-20 seconds (vs C-GEM ~12 seconds)

#### Memory Optimization
```bash
# For large simulations
python src/main.py --mode run --config config_large.txt
```
**Features**:
- Efficient memory management
- Reduced storage overhead
- Optimized for long simulations

### Gradient-Based Optimization

#### Parameter Calibration
```bash
# Calibrate against sparse data
python src/main.py --mode calibrate
```
**Capabilities**:
- JAX gradient computation
- Advanced optimizers (Optimistix, JAXopt)
- Statistical aggregate comparison
- Sparse field data calibration

#### Sensitivity Analysis
```bash
# Analyze parameter sensitivity
python tools/tools_cli.py analysis --sensitivity
```

### Advanced Plotting
```bash
# Publication-quality figures
python tools/plotting/publication_plots.py

# Interactive plot viewer
python tools/plotting/show_plots.py
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'jax'
pip install -r requirements.txt

# Error: Cannot import core modules
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
# or ensure you're running from project root
```

#### 2. Performance Issues
```bash
# Model running too slowly?
python src/main_optimized.py --no-physics-check

# Memory issues?
# Edit config/model_config.txt:
memory_fraction = 0.6
jax_threads = 2
```

#### 3. Physics Validation Errors
```bash
# Hypoxia detected or extreme values?
python src/main.py --debug --physics-check
# Check debug output for specific issues

# Disable physics validation for testing:
python src/main.py --no-physics-check
```

#### 4. Configuration Problems
```bash
# Check configuration validity
python -c "
import sys; sys.path.append('src')
from core.config_parser import parse_model_config
config = parse_model_config('config/model_config.txt')
print(f'✅ Config loaded: {len(config)} parameters')
"
```

#### 5. Missing Input Data
```bash
# Verify data files exist
ls INPUT/Boundary/
ls INPUT/Calibration/
ls INPUT/Geometry/
ls INPUT/Tributaries/

# Check data configuration
cat config/input_data_config.txt
```

### Performance Benchmarks

**Typical execution times** (Windows Core i7 laptop):
- **JAX C-GEM**: 35-45 seconds (standard mode)
- **JAX C-GEM Optimized**: 15-20 seconds (performance mode)
- **Original C-GEM**: 12-15 seconds (reference)

**Memory usage**:
- **JAX C-GEM**: ~150 MB
- **Original C-GEM**: ~150 MB

### Getting Help

1. **Check the logs**: Debug mode provides detailed information
2. **Verify setup**: Run the verification tools
3. **Compare with C-GEM**: Use the benchmark tools
4. **Check field data**: Validate against observations

### Error Codes

- **Exit Code 0**: Successful execution
- **Exit Code 1**: General error (check debug output)
- **Exit Code 2**: Configuration error
- **Exit Code 3**: Input data error
- **Exit Code 13**: C-GEM compatibility mode (expected)

---

## 🎓 Usage Recommendations

### For Model Development
```bash
# Use standard mode with validation
python src/main.py --mode run --physics-check --debug
```

### For Production Runs
```bash
# Use optimized mode
python src/main_optimized.py --mode run --no-physics-check
```

### For Research and Calibration
```bash
# Use full-featured mode
python src/main.py --mode calibrate --output-format csv
```

### For Performance Testing
```bash
# Use benchmark tools
python tools/validation/comprehensive_cgem_benchmark.py
```

---

## 📈 Next Steps

After running your first simulation:

1. **Visualize Results**: Use `python tools/plotting/show_plots.py`
2. **Validate Output**: Compare with field data using validation tools
3. **Optimize Parameters**: Use gradient-based calibration
4. **Generate Reports**: Create publication-quality figures

## 🔗 Related Files

- [`README.md`](README.md) - Project overview
- [`ENHANCED_BENCHMARK_SUMMARY.md`](ENHANCED_BENCHMARK_SUMMARY.md) - Benchmark details
- [`config/model_config.txt`](config/model_config.txt) - Model parameters
- [`config/input_data_config.txt`](config/input_data_config.txt) - Data configuration

---

**Last Updated**: August 2025  
**JAX C-GEM Version**: 1.0  
**Compatible with**: Original C-GEM output format
