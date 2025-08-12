# Quick Start Guide

Get JAX C-GEM running in under 5 minutes!

## Prerequisites

- Python 3.8+ (3.10-3.11 recommended)
- Git
- 4GB free disk space

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/flashshare/jax-C-GEM.git
cd jax-C-GEM

# Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import jax; print('✅ JAX C-GEM ready!')"
```

## Step 2: Run Your First Simulation

### Ultra-Fast Mode (Recommended)
```bash
python src/main_ultra_performance.py
```

This runs a **365-day tidal estuary simulation** in about 15 minutes!

### Standard Mode (with validation)
```bash
python src/main.py
```

This runs the same simulation in about 20 minutes with comprehensive validation and analysis.

Both modes produce identical scientific results - ultra-fast mode just skips some validation checks.

## Step 3: View Results (1 minute)

### Interactive Results Viewer
```bash
# Quick interactive visualization
python tools/plotting/show_results.py
```

*Note: This creates an interactive dashboard that opens in a separate window*

This creates a comprehensive dashboard with:
- Longitudinal profiles of key variables
- Time series at multiple stations
- Statistical summary of model performance
- Physics validation checks

*Figures last generated on 2025-08-12*

```bash
# Generate publication-quality figures
python tools/plotting/publication_output.py
```

![Publication Quality Figure - Hydrodynamics](figures/figure_1_hydrodynamics_transport_comprehensive.png)

These multi-panel figures include:
- Tidal range and salinity validation against field data
- Estuary geometry and velocity patterns
- Time series at three key stations

Your simulation results are also saved as:
- `OUT/simulation_results.npz` - Main simulation data
- `OUT/performance_report.txt` - Timing and memory usage
- `OUT/Publication/` - Publication-ready figures with documentation

## What Just Happened?

You just ran a complete **1D tidal estuary simulation** that includes:

- ✅ **Hydrodynamics**: Tidal waves, water levels, velocities
- ✅ **Transport**: Salt and temperature mixing  
- ✅ **Biogeochemistry**: 17-species reactive network (nutrients, oxygen, plankton)
- ✅ **Validation**: Comprehensive model validation against field data
- ✅ **Visualization**: Publication-quality figures and interactive dashboards

The model simulated **365 days** of estuary dynamics across **101 spatial points** with **180-second time steps** and created multiple visualization outputs for analysis.

## Key Output Files

```
OUT/
├── simulation_results.npz         # Main results (time series + spatial data)
├── Publication/                   # Auto-generated publication figures
│   ├── figures/
│   │   ├── figure_1_hydrodynamics_transport_comprehensive.png
│   │   ├── figure_1_hydrodynamics_transport_comprehensive.pdf
│   │   ├── figure_2_water_quality_comprehensive.png
│   │   └── figure_2_water_quality_comprehensive.pdf
│   ├── README.md                  # Description of generated figures
│   └── figure_captions.txt        # Ready-to-use figure captions
├── Validation/                    # Field data validation 
├── Statistical_Validation/        # Comprehensive statistical analysis
│   ├── validation_report.md       # Detailed statistical metrics
│   ├── salinity_comprehensive_validation.png
│   ├── oxygen_comprehensive_validation.png
│   └── tidal_range_comprehensive_validation.png
└── Advanced_Benchmarks/           # Comparison with other models
```

## Using VS Code (Optional)

If you use VS Code:

1. Open the `jax-C-GEM` folder in VS Code
2. Press `Ctrl+Shift+P` → "Tasks: Run Task"  
3. Select from these tasks:
   - **⚡ Ultra-Performance Mode** - Maximum speed simulation
   - **🔬 Run with Physics Validation** - Development mode
   - **🎨 Generate Publication Figures** - Create multi-panel publication figures
   - **🔍 Quick Results View** - Interactive visualization dashboard
   - **🔬 Statistical Model Validation** - Run comprehensive statistical validation
   - **🌊 Field Data Validation** - Compare against field observations

## Performance Comparison

Want to compare with the original C implementation?

```bash
# Run original C-GEM (Windows)
cd deprecated/original-C-GEM
compile_and_run.bat

# Compare performance
python tools/validation/comprehensive_cgem_benchmark.py
```

## Next Steps

- **📖 [Usage Guide](installation.md)** - Detailed installation and configuration
- **📊 [Results Guide](results.md)** - Understanding and analyzing outputs  
- **⚖️ [C vs JAX Comparison](comparison.md)** - Detailed benchmarks and validation

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade jax jaxlib numpy scipy pandas matplotlib
```

**Slow performance?**
- Use `main_ultra_performance.py` instead of `main.py`
- Check you have at least 8GB RAM available
- Close other applications during simulation

**Need help?** Check the [Usage Guide](installation.md) or create an issue on GitHub.

---

**🎉 Congratulations!** You've successfully run a high-performance tidal estuary simulation using JAX C-GEM!
