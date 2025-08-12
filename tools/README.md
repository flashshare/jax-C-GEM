# JAX C-GEM Analysis and Validation Tools

This directory contains powerful tools for analyzing, validating, and visualizing JAX C-GEM model results.

## 🚀 Quick Start

**Essential tools for every user:**

```bash
# 1. Check your setup before first run
python tools/setup_check.py

# 2. Run simulation (ultra-fast)
python main_ultra_performance.py

# 3. View results immediately
python tools/plotting/show_results.py

# 4. Validate against field data (if available)
python tools/validation/validate_against_field_data.py
```

## 📊 Visualization Tools (`plotting/`)

### `show_results.py` - Comprehensive Results Viewer ⭐
**Most important and comprehensive plotting tool**
- **Merged functionality** from auto_plot.py for enhanced capabilities
- **Automatic format detection**: NPZ (high-performance) or CSV (compatible)
- **Comprehensive 4x4 analysis layout** with 12 detailed panels
- **Enhanced physics validation** and stability checks
- **Salt intrusion analysis** and biogeochemical summaries
- **Automatic execution**: Runs automatically at end of all simulations
- **Flexible usage modes**: Interactive, automated, or detailed analysis

```bash
# Basic interactive use (most common)
python tools/plotting/show_results.py

# Automated mode with saved figures
python tools/plotting/show_results.py --auto --save-figures

# Comprehensive analysis with detailed plots
python tools/plotting/show_results.py --detailed --save-figures

# Force specific format
python tools/plotting/show_results.py --format npz

# Quiet mode for scripts
python tools/plotting/show_results.py --quiet --auto
```

**New Comprehensive Plot Layout (4x4 Grid):**
- Final longitudinal profiles (water level + salinity)  
- Velocity profiles with flow direction indicators
- Time series at mouth, mid-estuary, and head
- Tidal range and salt intrusion analysis
- Dissolved oxygen and nutrient dynamics
- Model stability and physics validation
- Comprehensive simulation statistics
- Available variables overview

### `auto_plot.py` - ⚠️ DEPRECATED - MERGED INTO show_results.py
- **Functionality merged** into enhanced `show_results.py`
- **File archived** to `archive/merged_plotting/`
- **Use `show_results.py`** for all plotting needs

### `publication_plots.py` - Publication Quality
- High-resolution figures for papers
- Professional formatting
- Multiple plot types and layouts

## 🔬 Validation Tools (`validation/`)

### `validate_against_field_data.py` - Field Data Comparison ⭐
**Essential for model validation**
- Compares model outputs with field observations
- Statistical metrics (RMSE, R², bias)
- Automatic station mapping
- Supports multiple observation datasets

### `comprehensive_cgem_benchmark.py` - Performance Comparison
- Compare JAX C-GEM vs original C-GEM
- Performance benchmarking
- Accuracy validation
- Cross-platform compatibility

## ⚙️ Setup and Diagnostics

### `setup_check.py` - Installation Verification ⭐
**Run this first to verify your setup**
- Check Python version and dependencies
- Verify directory structure
- Test JAX functionality
- Performance diagnostics

## 🔧 Analysis Tools (`analysis/`)

### `model_validation.py` - Advanced Validation Framework
- Statistical analysis
- Cross-validation
- Uncertainty quantification
- Comprehensive reporting

### `sensitivity_analysis.py` - Parameter Sensitivity
- Gradient-based sensitivity analysis
- Parameter importance ranking
- Uncertainty propagation

## 📋 How to Use These Tools

### For New Users:
1. **Setup Check**: `python tools/setup_check.py`
2. **Run Model**: `python main_ultra_performance.py`
3. **View Results**: `python tools/plotting/show_results.py`

### For Validation:
1. Add field data to `INPUT/Calibration/`
2. Run: `python tools/validation/validate_against_field_data.py`
3. Check: `OUT/validation_results/`

### For Publication:
1. Run: `python tools/plotting/publication_plots.py`
2. Customize plots in the script as needed
3. High-res figures saved automatically

## 📁 Output Directories

Tools create organized outputs:
- `OUT/plots/` - Quick visualization plots
- `OUT/validation_results/` - Validation analysis
- `OUT/Summary_Plots/` - Auto-generated summaries
- `OUT/publication_figures/` - Publication-ready plots

## 🎯 VS Code Integration

Use VS Code tasks for easy access:
- `Ctrl+Shift+P` → "Tasks: Run Task"
- Select from available tools:
  - 🔧 Setup Check
  - 🚀 Run Model  
  - 🖼️ Show Results
  - 🌊 Field Data Validation
  - 🎨 Generate Publication Figures

## 💡 Tips and Best Practices

### Output Format Selection:
- **NPZ format**: Faster, smaller files, better for large simulations
- **CSV format**: Human-readable, compatible with Excel/other tools

### Performance:
- Use NPZ format for multi-year simulations
- CSV format is fine for short runs (< 1 month)

### Validation:
- Field data files should have Date column in MM/DD/YYYY format
- Station locations are automatically mapped to nearest grid points
- Check validation reports for data quality issues

### Troubleshooting:
- Run setup_check.py if you encounter issues
- Check console output for detailed error messages
- Verify input data file formats and paths

## 🔗 Dependencies

Most tools automatically handle missing dependencies:
- **Required**: numpy, matplotlib, pandas
- **Optional**: scipy (advanced statistics), seaborn (enhanced plots)
- **JAX**: Required for gradient-based analysis tools

## 📖 Example Workflow

Complete analysis workflow for a new estuary:

```bash
# 1. Verify setup
python tools/setup_check.py

# 2. Configure your estuary in config/model_config.txt
# 3. Add boundary conditions to INPUT/Boundary/
# 4. Add field observations to INPUT/Calibration/ (optional)

# 5. Run simulation
python main_ultra_performance.py

# 6. Quick results check
python tools/plotting/show_results.py

# 7. Detailed validation (if field data available)
python tools/validation/validate_against_field_data.py

# 8. Generate publication figures
python tools/plotting/publication_plots.py

# 9. Check all outputs in OUT/ directory
```

---

**Need help?** Check the main [Quickstart.md](../Quickstart.md) for comprehensive setup instructions.
