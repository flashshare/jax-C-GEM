# Installation & Setup

Complete installation guide for JAX C-GEM.

## System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM  
- 4GB free disk space
- Windows 10/11, macOS, or Linux

**Recommended:**
- Python 3.10 or 3.11
- 16GB RAM
- SSD storage

## Installation

### Method 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/flashshare/jax-C-GEM.git
cd jax-C-GEM

# Install all dependencies
pip install -r requirements.txt

# Test installation
python -c "import jax; print('JAX version:', jax.__version__)"
```

### Method 2: Virtual Environment (Recommended)

```bash
# Create isolated environment
python -m venv jax-cgem-env

# Activate (Windows)
jax-cgem-env\Scripts\activate

# Activate (Mac/Linux)
source jax-cgem-env/bin/activate

# Install
git clone https://github.com/flashshare/jax-C-GEM.git
cd jax-C-GEM
pip install -r requirements.txt
```

### Method 3: Conda

```bash
# Create conda environment
conda create -n jax-cgem python=3.10
conda activate jax-cgem

# Install
git clone https://github.com/flashshare/jax-C-GEM.git
cd jax-C-GEM
pip install -r requirements.txt
```

## Verify Installation

Run this verification script:

```bash
python -c "
import sys
print('Python version:', sys.version)

import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ All core dependencies working!')

# Test JAX functionality
x = jax.numpy.array([1, 2, 3, 4])
y = jax.numpy.sum(x)
print('JAX test:', x, '→', y)
print('🎉 JAX C-GEM ready to run!')
"
```

Expected output:
```
Python version: 3.10.x
JAX version: 0.4.38
JAX devices: [CpuDevice(id=0)]
✅ All core dependencies working!
JAX test: [1 2 3 4] → 10
🎉 JAX C-GEM ready to run!
```

## Project Structure

After installation, your directory should look like:

```
jax-C-GEM/
├── src/                    # JAX C-GEM implementation
│   ├── main.py            # Standard mode
│   ├── main_ultra_performance.py  # Fast mode
│   └── core/              # Core model components
├── config/                # Configuration files
│   ├── model_config.txt   # Model parameters
│   └── input_data_config.txt  # Data sources
├── INPUT/                 # Input data (boundary conditions, geometry)
├── deprecated/original-C-GEM/  # Original C implementation
├── tools/                 # Analysis and plotting tools
└── docs/                  # This documentation
```

## Configuration Files

JAX C-GEM uses two main configuration files:

### 1. Model Configuration (`config/model_config.txt`)

Core simulation parameters:
```ini
# Simulation time
MAXT = 365                 # Total days to simulate
WARMUP = 100              # Warmup period (days)
DELTI = 180               # Time step (seconds)

# Spatial domain  
EL = 200000               # Estuary length (m)
DELXI = 2000              # Grid spacing (m)

# Tidal forcing
AMPL = 4.43               # Tidal amplitude (m)
```

### 2. Data Configuration (`config/input_data_config.txt`)

Input data file paths and settings:
```ini
# Boundary conditions
boundary_data_dir = "INPUT/Boundary"
upstream_salinity = 0.0
downstream_salinity = 35.0

# Geometry
geometry_file = "INPUT/Geometry/Geometry.csv"
```

## Basic Usage

### Run Simulation

```bash
# Maximum performance (recommended)
python src/main_ultra_performance.py

# Standard mode with full validation
python src/main.py

# Debug mode
python src/main.py --debug
```

### View Results

```bash
# Interactive plots
python tools/plotting/show_results.py

# Generate publication figures
python tools/plotting/publication_output.py
```

### Compare with Original C-GEM

```bash
# Compile and run original (Windows)
cd deprecated/original-C-GEM
compile_and_run.bat

# Run comprehensive benchmark
python tools/validation/comprehensive_cgem_benchmark.py
```

## Troubleshooting

### JAX Installation Issues

**Windows users:**
```bash
# If you get JAX installation errors
pip install --upgrade pip
pip install jax==0.4.38 jaxlib==0.4.38
```

**Mac/Linux users:**
```bash
# For CPU-only installation
pip install -U "jax[cpu]"
```

### Memory Issues

If you run out of memory:

1. **Reduce simulation length:**
   ```ini
   # In config/model_config.txt
   MAXT = 90  # Reduce from 365 to 90 days
   ```

2. **Increase grid spacing:**
   ```ini
   DELXI = 4000  # Increase from 2000m to 4000m
   ```

3. **Use CSV output instead of NPZ:**
   ```bash
   python src/main.py --output-format csv
   ```

### Performance Issues

For maximum speed:
- Use `main_ultra_performance.py`
- Close other applications
- Ensure sufficient RAM (16GB recommended)
- Use SSD storage for faster I/O

### Import Errors

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# If specific package fails
pip install --upgrade numpy scipy pandas matplotlib jax jaxlib
```

## Hardware Recommendations

| Use Case | RAM | CPU | Storage | Expected Speed |
|----------|-----|-----|---------|----------------|
| Testing | 8GB | Any | HDD | 15,000 steps/min |
| Research | 16GB | Modern CPU | SSD | 25,000 steps/min |
| Production | 32GB | High-end CPU | NVMe SSD | 30,000+ steps/min |

## Next Steps

✅ Installation complete!  

**👉 [Quick Start](quick-start.md)** - Run your first simulation  
**📊 [Understanding Results](results.md)** - Analyze simulation outputs  
**⚖️ [C vs JAX Comparison](comparison.md)** - Performance and validation
