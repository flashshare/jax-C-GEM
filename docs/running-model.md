# Running the Model

How to execute JAX C-GEM simulations and control model behavior.

## Execution Modes

JAX C-GEM offers two main execution modes:

### 1. Ultra-Performance Mode (Recommended)

**Maximum speed - optimized for production runs:**
```bash
python src/main_ultra_performance.py
```

**Features:**
- ⚡ **30,000+ steps/minute** (2.5x faster than C)
- 🚀 Vectorized batch processing
- 📦 Optimized memory usage
- ⚠️ Minimal validation (for speed)

**Use for:** Production runs, long simulations, benchmarking

### 2. Standard Mode

**Full validation - optimized for development:**
```bash
python src/main.py
```

**Features:**
- 🔬 **Physics validation enabled**
- 🐛 **Detailed error checking**  
- 📋 **Comprehensive logging**
- 🔍 **Debug information**

**Use for:** Development, testing, validation

## Command Line Options

### Standard Mode Options

```bash
# Basic run
python src/main.py

# Debug mode with detailed output
python src/main.py --debug

# Force CSV output format
python src/main.py --output-format csv

# Skip physics validation for speed
python src/main.py --no-physics-check

# Use custom configuration
python src/main.py --config config/my_custom_config.txt
```

### Ultra-Performance Mode

Ultra-performance mode uses optimized defaults and has no command-line options for maximum speed.

## Using VS Code Tasks

If you use VS Code, the project includes pre-configured tasks:

**Press `Ctrl+Shift+P` → "Tasks: Run Task" → Select:**

- **🚀 Run Model** - Standard mode with auto format
- **⚡ Ultra-Performance Mode** - Maximum speed
- **🔬 Run with Physics Validation** - Development mode
- **🐛 Debug Mode** - Full debugging output

## Model Configuration

### Key Parameters (`config/model_config.txt`)

**Simulation Time:**
```ini
MAXT = 365                # Total simulation days
WARMUP = 100             # Warmup period (days)
DELTI = 180              # Time step (seconds)
TS = 10                  # Output frequency (every N steps)
```

**Spatial Domain:**
```ini
EL = 200000              # Estuary length (meters)
DELXI = 2000             # Grid spacing (meters)  
M = 101                  # Number of grid points (calculated)
```

**Tidal Forcing:**
```ini
AMPL = 4.43              # Tidal amplitude (meters)
OMEGA = 1.41e-4          # M2 tidal frequency (rad/s)
```

**For different scenarios:**

| Scenario | MAXT | DELXI | Purpose |
|----------|------|-------|---------|
| Quick test | 30 | 4000 | Fast validation |
| Development | 90 | 2000 | Testing changes |
| Production | 365 | 2000 | Full simulation |
| High-res | 365 | 1000 | Detailed analysis |

## Understanding the Simulation

### What the Model Does

1. **Initializes** estuary geometry and boundary conditions
2. **Time-steps** through the simulation period:
   - Solves hydrodynamics (water levels, velocities)
   - Calculates transport (salt, heat, chemicals)
   - Computes biogeochemistry (17-species reactions)
3. **Saves results** for analysis

### Simulation Progress

During execution, you'll see:
```
🌊 JAX C-GEM Ultra-Performance Mode
==================================================
📋 Configuration loaded: 365 days, 101 grid points
⚡ JIT compiling simulation functions...
🚀 Starting simulation: 2017-01-01 to 2017-12-31

Progress: [████████████████████] 100% | 365/365 days | 28,450 steps/min
✅ Simulation completed in 72.3 seconds
💾 Results saved to: OUT/simulation_results.npz
📊 Performance report: OUT/performance_report.txt
```

### Output Files

**Automatic outputs:**
```
OUT/
├── simulation_results.npz      # Main simulation data
├── performance_report.txt      # Timing and memory stats
└── Publication/                # Auto-generated figures (if enabled)
```

**NPZ format** (recommended for large simulations):
- Binary format, fast loading
- Contains all variables as NumPy arrays
- Smaller file size

**CSV format** (good for analysis):
- Human-readable text format
- Easy to import into Excel, R, etc.
- Larger file size

## Common Simulation Scenarios

### Quick Test Run (2 minutes)

```bash
# Edit config/model_config.txt:
# MAXT = 30
# DELXI = 4000

python src/main_ultra_performance.py
```

### Development Run (10 minutes)

```bash
# Edit config/model_config.txt:
# MAXT = 90
# DELXI = 2000

python src/main.py --debug
```

### Production Run (30 minutes)

```bash
# Use default config (365 days, 2km resolution)
python src/main_ultra_performance.py
```

### High-Resolution Run (60 minutes)

```bash
# Edit config/model_config.txt:
# MAXT = 365
# DELXI = 1000  # Higher resolution

python src/main_ultra_performance.py
```

## Performance Tips

### Maximize Speed
1. Use `main_ultra_performance.py`
2. Increase `DELXI` (lower spatial resolution)
3. Close other applications
4. Use NPZ output format
5. Ensure sufficient RAM (16GB+)

### Memory Optimization
1. Reduce `MAXT` (shorter simulation)
2. Increase `TS` (less frequent output)
3. Use CSV format for very long runs
4. Monitor memory usage during simulation

### Development Workflow
1. Start with quick test run (`MAXT = 30`)
2. Validate results with standard mode
3. Scale up to production run
4. Use ultra-performance mode for final runs

## Monitoring Performance

### Built-in Performance Reports

Every run generates `OUT/performance_report.txt`:
```
JAX C-GEM Performance Report
============================
Simulation time: 72.3 seconds
Total steps: 2,106,400
Steps per minute: 28,450
Memory usage: 2.4 GB peak
JIT compilation: 12.1 seconds
Actual simulation: 60.2 seconds
```

### Real-time Monitoring

During long runs, monitor:
- **CPU usage**: Should be high (80-100%)
- **Memory usage**: Should be stable (not growing)
- **Disk I/O**: Periodic spikes during output saves

## Next Steps

**🏃 Ready to run!** Try the [Quick Start](quick-start.md) workflow.

**📊 Analyze results:** See [Understanding Results](results.md)  
**⚖️ Compare performance:** Check [C vs JAX Comparison](comparison.md)

## Troubleshooting

**Simulation runs but very slow?**
- Use `main_ultra_performance.py` instead
- Check available RAM (need 8GB+)
- Increase `DELXI` to reduce grid points

**Memory errors?**
- Reduce `MAXT` (simulation length)
- Increase `DELXI` (grid spacing)  
- Use CSV output format

**Physics errors or warnings?**
- Use standard mode: `python src/main.py`
- Check configuration parameters
- Verify input data files exist
