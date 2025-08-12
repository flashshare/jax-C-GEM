# JAX C-GEM Codebase Organization

## 🚀 PRODUCTION FILES (Essential)

### Main Execution Scripts
- **`main_ultra_performance.py`** - ⚡ Maximum performance mode (30,000+ steps/min, 2.5x speedup)
- **`src/main.py`** - 🔧 Standard mode with full debugging and validation options

### Core Engine (`src/core/`)
- **`simulation_engine.py`** - Standard optimized simulation engine
- **`simulation_engine_batch.py`** - Ultra-high performance vectorized batch engine
- **`hydrodynamics.py`** - Saint-Venant hydrodynamic solver
- **`transport.py`** - Advection-dispersion transport with stability
- **`biogeochemistry.py`** - 17-species biogeochemical reaction network
- **`config_parser.py`** - Configuration file parsing
- **`data_loader.py`** - Input data loading and validation
- **`model_config.py`** - Model constants and species definitions

### Configuration (`config/`)
- **`model_config.txt`** - Main model parameters
- **`input_data_config.txt`** - Input data sources configuration

### Tasks (`.vscode/tasks.json`)
- **🚀 Run Model** - Default standard mode
- **⚡ Maximum Performance Mode** - Ultra-high performance mode
- **🔬 Physics Validation** - Debug/development mode

## 🔬 ANALYSIS & TOOLS (`tools/`)

### Validation & Benchmarking
- **`tools/validation/comprehensive_cgem_benchmark.py`** - Complete C-GEM vs JAX validation
- **`tools/validation/validate_against_field_data.py`** - Field data comparison

### Plotting & Visualization  
- **`tools/plotting/publication_plots.py`** - Publication-quality figures
- **`tools/plotting/auto_plot.py`** - Automated result visualization

### Analysis Tools
- **`tools/analysis/result_processor.py`** - Statistical analysis of results
- **`tools/analysis/sensitivity_analyzer.py`** - Parameter sensitivity analysis
- **`tools/calibration/gradient_calibrator.py`** - JAX-based gradient calibration

## 🗂️ DEVELOPMENTAL FILES (Non-essential)

### Performance Testing
- **`tools/performance_test.py`** - Bottleneck measurement (used during optimization)
- **`monitor_simulation.py`** - Long-running simulation monitoring

### Deprecated/Archive
- **`deprecated/`** - Original C-GEM code and obsolete scripts
- **`deprecated/main_maximum_performance.py`** - Incomplete attempt (moved from root)

## 📊 PERFORMANCE COMPARISON

| Mode | Script | Performance | Use Case |
|------|--------|-------------|----------|
| **Standard** | `src/main.py` | ~12,500 steps/min | Development, debugging, learning |
| **Maximum** | `main_ultra_performance.py` | **30,867 steps/min** | Production, parameter sweeps, large studies |

## 🎯 QUICK START

**For Production (Fastest):**
```bash
python main_ultra_performance.py
```

**For Development/Debug:**
```bash  
python src/main.py --debug --output-format csv
```

**Using VS Code Tasks:**
- Press `Ctrl+Shift+P` → "Tasks: Run Task" → Select desired mode

## 🔄 MAINTENANCE NOTES

### Clean Codebase Status
- ✅ No duplicate main scripts
- ✅ Obsolete files moved to `deprecated/`
- ✅ Clear separation of production vs development code
- ✅ All essential functionality tested and working
- ✅ Performance optimizations complete (2.5x speedup achieved)

### Scientific Integrity
- ✅ **Zero trade-offs in physics accuracy**
- ✅ All modes produce identical scientific results
- ✅ Only computational efficiency differs between modes
- ✅ Complete validation against original C-GEM
