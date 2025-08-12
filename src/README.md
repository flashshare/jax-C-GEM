# JAX C-GEM Source Directory

**Core Simulation Engine - Essential Scientific Computing Modules**

This directory contains the **clean core** of the JAX C-GEM model: only essential scientific computing modules needed for simulation. The architecture follows the **configuration-driven** design principle where all case-specific parameters are externalized to config files.

## 🏗️ Architecture Overview

The `src/` directory implements a **pure JAX-native paradigm** with:
- **Functional Programming**: All core functions are pure (no side effects)
- **Vectorized Operations**: No explicit loops, fully vectorized with `jax.numpy` 
- **JIT Compilation**: Main simulation loop is JIT-compiled for performance
- **Gradient-Ready**: Built to support `jax.grad` for optimization

## 📁 Core Module Structure

```
src/
├── main.py                # Entry point with graceful tool imports
├── simulation_engine.py   # Core simulation loop coordination
├── hydrodynamics.py       # 1D shallow water equations (de Saint-Venant) 
├── transport.py           # Advection-dispersion with TVD schemes
├── biogeochemistry.py     # 17-species reactive transport network
├── model_config.py        # Physical constants and fundamental definitions
├── config_parser.py       # Configuration file parsing utilities
└── data_loader.py         # Input data management and validation
```

## 🧬 Scientific Components

### `main.py` - Simulation Orchestrator
- **Role**: Lean coordination script with no scientific logic
- **Function**: Parse configs → Load data → Initialize state → Run simulation → Save results
- **Features**: Graceful degradation when auxiliary tools are missing
- **Design**: Pure orchestration following architectural mandate

### `simulation_engine.py` - Core Simulation Loop  
- **Physics**: Main time-stepping loop with JIT compilation
- **Integration**: Coordinates hydrodynamics, transport, and biogeochemistry
- **Performance**: `@jax.jit` decorated for maximum speed
- **Architecture**: Pure functional design with immutable state updates

### `hydrodynamics.py` - Water Flow Dynamics
- **Equations**: 1D de Saint-Venant shallow water equations
- **Method**: Semi-implicit iterative scheme on staggered Arakawa C-grid
- **Solver**: Tridiagonal system with `lineax` for JAX compatibility
- **Boundary Conditions**: Tidal elevation (downstream) + discharge (upstream)
- **Features**: Proper flood/ebb asymmetry and tidal amplification/damping

### `transport.py` - Species Transport
- **Advection**: Total Variation Diminishing (TVD) schemes with Superbee limiter
- **Dispersion**: Crank-Nicolson implicit scheme with Van der Burgh formula
- **Numerical**: Operator splitting (advection → dispersion → reactions)  
- **Conservation**: Strict mass conservation for all 17 species
- **Performance**: Fully vectorized operations with `jax.vmap`

### `biogeochemistry.py` - Reactive Network
- **Species**: Complete 17-species network (PHY1, PHY2, Si, NO3, NH4, PO4, PIP, O2, TOC, S, SPM, DIC, AT, HS, pH, ALKC, CO2)
- **Reactions**: Full C-GEM biogeochemical framework (Volta et al., 2016)
- **Kinetics**: Temperature-dependent rates with Q₁₀ functions
- **Limitations**: Michaelis-Menten kinetics with nutrient limitation/inhibition
- **Stoichiometry**: Scientifically accurate Redfield ratios (C:N:P:Si = 106:16:1:15)

### `model_config.py` - Core Definitions
- **Constants**: Fundamental physical constants (gravity, gas constant, etc.)
- **Species List**: Canonical ordering of 17 biogeochemical species
- **Units**: Consistent unit system definitions
- **Design**: Static definitions only (no case-specific values)

### `config_parser.py` - Configuration Gateway
- **Role**: Exclusive entry point for external configuration
- **Function**: Parse and validate `model_config.txt` and `input_data_config.txt`
- **Validation**: Type checking and parameter bounds enforcement  
- **Error Handling**: Clear error messages for configuration issues

### `data_loader.py` - Input Data Management
- **Role**: Exclusive entry point for external data files
- **Formats**: CSV time series for boundary conditions and forcing data
- **Validation**: Data range checking and temporal consistency
- **Interpolation**: Temporal interpolation for simulation time stepping
- **Error Handling**: Graceful handling of missing or invalid data files

## ⚙️ Architectural Principles Implemented

### 1. Total Configuration-Driven Design ✅
- **Zero Hardcoding**: All parameters read from external config files
- **Portable Framework**: Change estuary by updating configs only
- **`main.py` Orchestration**: Lean coordination with no scientific logic

### 2. JAX-Native Paradigm ✅  
- **Pure Functions**: All core functions are side-effect free
- **Vectorization**: No explicit `for` loops in numerical code
- **JIT Compilation**: `@jax.jit` on main simulation step
- **Gradient-Ready**: Structured for `jax.grad` optimization

### 3. Scientific Rigor ✅
- **Correct Physics**: Proper implementation of governing equations
- **State-of-the-Art Methods**: TVD transport, semi-implicit hydrodynamics
- **Mass Conservation**: Strict conservation for all species
- **Numerical Stability**: CFL condition monitoring and adaptive time stepping

### 4. Maintainability ✅
- **Modularity**: Single-responsibility functions and clear interfaces
- **Documentation**: Comprehensive docstrings explaining scientific purpose
- **Clean Interfaces**: Well-defined function signatures with type hints

## 🔬 Integration with Auxiliary Tools

The core modules use **graceful degradation** to integrate with auxiliary tools in `tools/`:

```python
# Example: Graceful tool import in main.py
try:
    from tools.diagnostics.performance_profiler import ProfilerManager
    HAS_PROFILER = True
except ImportError:
    class NullProfiler:
        def start(self): pass
        def end(self): pass
    ProfilerManager = NullProfiler
    HAS_PROFILER = False
```

**Benefits:**
- Core simulation runs independently of auxiliary tools
- Enhanced features activate when tools are available
- Development flexibility without breaking core functionality

## 🚀 Usage Patterns

### Standard Execution
```python
# Standard simulation with full features
python src/main.py --mode run --output-format csv --physics-check
```

### Core-Only Execution  
```python
# Core simulation without auxiliary tools (still fully functional)
python src/main.py --mode run --output-format csv --no-physics-check
```

### Import as Module
```python
# Use core modules in custom scripts
import sys
sys.path.append('src')
from simulation_engine import run_simulation
from config_parser import parse_model_config

config = parse_model_config('config/model_config.txt')
results = run_simulation(config)
```

## 🎯 Performance Characteristics

### Computational Efficiency
- **JIT Compilation**: Near-C performance for numerical kernels
- **Memory Usage**: ~150 MB (comparable to original C-GEM)  
- **Execution Time**: ~35-45 seconds (standard mode), ~15-20 seconds (optimized)
- **Scalability**: Handles 1000+ grid points efficiently

### Numerical Properties
- **Stability**: CFL-compliant time stepping with convergence monitoring
- **Accuracy**: Mass-conservative transport with minimal numerical diffusion
- **Robustness**: Automatic parameter adjustment for numerical stability

## 📚 Scientific Foundation

The core modules implement established numerical methods:

### Hydrodynamics
- **de Saint-Venant Equations**: Classical shallow water theory
- **Semi-Implicit Scheme**: Unconditionally stable for tidal applications
- **Staggered Grid**: Arakawa C-grid for proper wave propagation

### Transport  
- **TVD Schemes**: High-resolution methods preventing spurious oscillations
- **Van der Burgh Dispersion**: Realistic estuarine mixing parameterization
- **Operator Splitting**: Mathematically rigorous decoupling of processes

### Biogeochemistry
- **C-GEM Framework**: Validated against Scheldt estuary observations  
- **Redfield Stoichiometry**: Consistent nutrient ratios in marine systems
- **Temperature Dependence**: Q₁₀ kinetics for realistic seasonal variation

## 🔧 Development Guidelines

When modifying core modules:

1. **Maintain Purity**: Keep functions pure and side-effect free
2. **Preserve Vectorization**: No explicit loops in numerical code  
3. **Configuration-First**: Add parameters to config files, not code
4. **Document Science**: Explain the physical/chemical meaning
5. **Test Conservation**: Verify mass balance is maintained
6. **Validate Physics**: Ensure results are physically reasonable

---

**The `src/` directory represents the scientific heart of JAX C-GEM**: a clean, efficient, and scientifically rigorous implementation of tidal estuary dynamics that maintains the accuracy of the original C-GEM while providing the computational power needed for modern scientific applications.
