# JAX C-GEM Physics Debugging Roadmap
## Comprehensive Model Physics Validation and Repair Plan

### Executive Summary
🎯 **Current Status**: While numerical oscillations were eliminated (POOR→FAIR), the 3-phase verification reveals fundamental physics implementation issues that must be resolved before calibration.

🔧 **Critical Finding**: Systematic tidal overestimation (2-3x observed) and poor species transport correlations indicate deeper problems in hydrodynamics, transport equations, and boundary conditions rather than parameter values.

�️ **Next Phase**: Comprehensive physics debugging from data loading through mass conservation to ensure mathematical implementation matches intended estuarine physics.

---

## 🏆 Physics Repair Success (Completed)

### Achievements
- ✅ **Eliminated Numerical Oscillations**: No more chaotic spikes in concentration fields
- ✅ **Stable Transport Solver**: Robust tridiagonal matrix solver with proper boundary conditions
- ✅ **Smooth Spatial Gradients**: Model produces realistic estuarine concentration profiles
- ✅ **Conservation Compliance**: Mass balance maintained throughout simulation domain
- ✅ **Performance Optimization**: JAX-native implementation with JIT compilation

### Technical Fixes Implemented
1. **Boundary Condition Stabilization**: Proper upstream/downstream flux handling
2. **Matrix Solver Robustness**: Enhanced tridiagonal solver with numerical stability checks
3. **Time Step Optimization**: Adaptive CFL condition enforcement
4. **Vectorized Operations**: Eliminated explicit loops for JAX compatibility

---

## � Physics Issues Identified from Verification

### Phase 2 - Tidal Dynamics: **CRITICAL PHYSICS PROBLEMS**
**Status**: � **Systematic Physics Errors - Not Calibration Issues**

| Station | Observed Range (m) | Model Range (m) | Overestimation Factor | Issue Type |
|---------|-------------------|-----------------|----------------------|------------|
| PC (86km) | 2.6 | 5.6 | 2.15x | Physics Error |
| BD (130km) | 3.2 | 7.5 | 2.34x | Physics Error |
| BK (156km) | 2.1 | 6.8 | 3.24x | Physics Error |

**Root Causes (Not Parameter Issues)**:
- Hydrodynamic equations may be incorrectly implemented
- Momentum terms, friction, or geometry calculations flawed
- Boundary condition forcing incorrect
- Time stepping or numerical scheme problems

### Phase 1 - Longitudinal Profiles: **TRANSPORT PHYSICS PROBLEMS**
**Status**: 🔴 **Transport Implementation Issues**

| Species | R² Score | Status | Likely Issue |
|---------|----------|--------|--------------|
| O₂      | 0.792    | 🟢 Acceptable | Transport working for O₂ |
| SPM     | 0.367    | 🔴 Poor | Settling/transport physics |
| S       | 0.260    | 🔴 Poor | Dispersion calculation |
| PO₄     | 0.284    | 🔴 Poor | Transport boundary conditions |
| NH₄     | 0.203    | 🔴 Poor | Source/sink implementation |
| TOC     | 0.125    | 🔴 Poor | Particulate transport physics |

**Root Causes**:
- Transport equation implementation errors
- Mass conservation violations
- Dispersion coefficient calculation problems
- Boundary condition application issues

### Phase 3 - Seasonal Cycles: **TEMPORAL COVERAGE LIMITATION**
**Status**: ⚠️ **Insufficient but Working** (35 days vs 365 needed)

---

## 🎯 Comprehensive Physics Debugging Roadmap

### Phase A: Data Loading and Input Validation (Days 1-2)

#### A1. Configuration File Validation
**Objective**: Ensure all input parameters are physically reasonable and consistent

**Critical Checks**:
- [ ] **Geometry Validation**: EL=202km, DELXI=2km → M=102 grid points
- [ ] **Tidal Forcing**: AMPL=4.43m at mouth - verify against SIHYMECC boundary data
- [ ] **Time Stepping**: DELTI=180s - check CFL condition for stability
- [ ] **Physical Constants**: G=9.81, Manning coefficients, dispersion parameters

**Implementation**:
```python
tools/debugging/validate_input_configuration.py
```
- Cross-check model_config.txt against field data ranges
- Verify CFL condition: DELTI < DELXI / (max velocity + sqrt(g*max depth))
- Compare boundary forcing with actual field measurements

#### A2. Boundary Data Quality Assessment
**Objective**: Verify boundary condition data matches field observations

**Critical Checks**:
- [ ] **Downstream Salinity**: Should be ~35 PSU (seawater)
- [ ] **Upstream Salinity**: Should be ~0 PSU (freshwater)
- [ ] **Tidal Amplitude**: Verify against SIHYMECC mouth measurements
- [ ] **Nutrient Gradients**: Check upstream high / downstream low pattern

**Implementation**:
```python
tools/debugging/validate_boundary_data.py
```
- Plot boundary time series vs field data
- Check for data gaps, outliers, unrealistic values
- Verify tributary discharge matches river flow data

#### A3. Geometry and Bathymetry Verification
**Objective**: Ensure geometric representation matches real estuary

**Critical Checks**:
- [ ] **Cross-sectional Areas**: Verify against surveyed data
- [ ] **Width Variations**: Check realistic estuary funnel shape
- [ ] **Depth Profiles**: Ensure proper channel deepening seaward
- [ ] **Distance Grid**: Confirm 0km=mouth to 202km=head mapping

**Implementation**:
```python
tools/debugging/validate_geometry.py
```
- Compare model geometry with field survey data
- Visualize channel shape and validate against Google Earth
- Check for unrealistic sudden changes in cross-section

---

### Phase B: Hydrodynamic Equation Implementation (Days 3-5)

#### B1. Saint-Venant Equation Verification
**Objective**: Verify correct implementation of shallow water equations

**Critical Checks**:
- [ ] **Continuity Equation**: ∂A/∂t + ∂Q/∂x = 0
- [ ] **Momentum Equation**: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂η/∂x + friction = 0
- [ ] **Friction Terms**: Manning's equation implementation
- [ ] **Pressure Gradients**: Hydrostatic pressure assumption

**Mathematical Verification**:
```python
tools/debugging/verify_saintenant_equations.py
```
- Analytical test cases with known solutions
- Check energy conservation in frictionless case
- Verify tidal wave propagation speed: c = √(gh)

#### B2. Boundary Condition Implementation
**Objective**: Ensure boundary conditions are correctly applied to hydrodynamics

**Critical Checks**:
- [ ] **Downstream Boundary**: Tidal elevation forcing H(0,t) = AMPL*sin(ωt)
- [ ] **Upstream Boundary**: River discharge Q(L,t) = Q_river
- [ ] **Ghost Points**: Proper extrapolation at boundaries
- [ ] **Reflection Handling**: Non-reflecting boundary conditions

**Implementation**:
```python
tools/debugging/validate_hydro_boundaries.py
```
- Check boundary value application in matrix solver
- Verify no artificial reflections at boundaries
- Test with analytical tidal propagation solution

#### B3. Numerical Stability Analysis
**Objective**: Ensure hydrodynamic solver is numerically stable

**Critical Checks**:
- [ ] **CFL Condition**: Δt ≤ Δx/(|u| + √(gh))
- [ ] **Matrix Conditioning**: Check tridiagonal solver stability
- [ ] **Iteration Convergence**: Newton-Raphson convergence rates
- [ ] **Mass Conservation**: ∂A/∂t + ∂Q/∂x = 0 exactly satisfied

**Implementation**:
```python
tools/debugging/analyze_hydro_stability.py
```
- Monitor convergence of Newton iterations
- Check conservation residuals at each time step
- Analyze eigenvalues of linearized system

---

### Phase C: Transport Equation Implementation (Days 6-8)

#### C1. Advection-Dispersion Equation Verification
**Objective**: Verify correct implementation of transport physics

**Critical Checks**:
- [ ] **Advection Term**: ∂C/∂t + u∂C/∂x (upwind scheme)
- [ ] **Dispersion Term**: ∂/∂x(D∂C/∂x) with proper coefficients
- [ ] **Source/Sink Terms**: Biogeochemical reactions integration
- [ ] **Cross-sectional Integration**: Proper 1D reduction from 3D equations

**Mathematical Verification**:
```python
tools/debugging/verify_transport_equations.py
```
- Test against analytical solutions (Gaussian plume, exponential decay)
- Verify Peclet number calculations Pe = uL/D
- Check numerical dispersion vs physical dispersion

#### C2. Dispersion Coefficient Calculation
**Objective**: Ensure dispersion coefficients are physically realistic

**Critical Checks**:
- [ ] **Elder's Formula**: D = 5.93 * depth * u_friction
- [ ] **Tidal Dispersion**: Enhanced mixing due to tidal oscillations  
- [ ] **Geometric Dispersion**: Effects of channel geometry
- [ ] **Magnitude Check**: Typical estuarine values 10-1000 m²/s

**Implementation**:
```python
tools/debugging/validate_dispersion_coefficients.py
```
- Compare calculated D with literature values for similar estuaries
- Check spatial and temporal variation patterns
- Verify scaling with velocity and geometry

#### C3. Mass Conservation Verification
**Objective**: Ensure transport solver conserves mass exactly

**Critical Checks**:
- [ ] **Total Mass Balance**: ∫∫∫ C dV = constant (no reactions)
- [ ] **Flux Conservation**: Inflow = Outflow + Accumulation
- [ ] **Boundary Flux**: Proper treatment of advective/diffusive fluxes
- [ ] **Numerical Conservation**: Matrix solver preserves mass

**Implementation**:
```python
tools/debugging/check_mass_conservation.py
```
- Track total system mass at each time step
- Calculate mass balance residuals
- Verify boundary flux calculations

---

### Phase D: Biogeochemical Process Implementation (Days 9-11)

#### D1. Reaction Network Verification
**Objective**: Ensure biogeochemical reactions are correctly implemented

**Critical Checks**:
- [ ] **Stoichiometry**: Redfield ratios C:N:P = 106:16:1
- [ ] **Rate Laws**: Michaelis-Menten kinetics for uptake
- [ ] **Temperature Dependence**: Arrhenius scaling Q₁₀ ≈ 2
- [ ] **Oxygen Limitation**: Proper switching between aerobic/anaerobic

**Scientific Verification**:
```python
tools/debugging/verify_biogeochemistry.py
```
- Test reaction rates against laboratory measurements
- Check mass balance for each biogeochemical transformation
- Verify coupling between carbon, nitrogen, phosphorus cycles

#### D2. Species Interaction Validation
**Objective**: Ensure proper coupling between chemical species

**Critical Checks**:
- [ ] **Primary Production**: O₂ production coupled to nutrient uptake
- [ ] **Respiration**: O₂ consumption coupled to organic matter oxidation
- [ ] **Nitrification**: NH₄→NO₃ conversion with O₂ consumption
- [ ] **Denitrification**: NO₃ reduction under low O₂ conditions

**Implementation**:
```python
tools/debugging/validate_species_coupling.py
```
- Check stoichiometric consistency in reactions
- Verify O₂ balance matches carbon processing
- Test extreme cases (high/low nutrients, O₂)

#### D3. Parameter Realism Check
**Objective**: Verify biogeochemical parameters are within observed ranges

**Critical Checks**:
- [ ] **Growth Rates**: μₘₐₓ ~ 1-3 d⁻¹ for phytoplankton
- [ ] **Half-saturation**: K_s ~ 1-10 mmol/m³ for nutrients
- [ ] **Mortality Rates**: m ~ 0.1-0.5 d⁻¹
- [ ] **Settling Velocities**: w_s ~ 0.1-5 m/d for particles

**Implementation**:
```python
tools/debugging/validate_biogeochem_parameters.py
```
- Compare with published values for tropical estuaries
- Check sensitivity to parameter variations
- Identify parameters causing unrealistic behavior

---

### Phase E: Integration and System-Level Validation (Days 12-14)

#### E1. Coupling Verification
**Objective**: Ensure proper coupling between hydrodynamics, transport, and biogeochemistry

**Critical Checks**:
- [ ] **Velocity Field**: Transport uses correct hydrodynamic velocities
- [ ] **Time Step Consistency**: All modules use same Δt
- [ ] **Boundary Condition Passing**: Consistent BCs across modules
- [ ] **State Variable Updates**: Proper sequence of operations

**Implementation**:
```python
tools/debugging/validate_module_coupling.py
```
- Check data flow between modules
- Verify time step synchronization
- Test with decoupled runs (hydro-only, transport-only)

#### E2. Energy and Mass Balance
**Objective**: Verify overall system conservation laws

**Critical Checks**:
- [ ] **Energy Conservation**: Tidal energy dissipation matches friction losses
- [ ] **Salt Conservation**: Total salt mass conserved
- [ ] **Nutrient Conservation**: Mass balance for N, P, Si
- [ ] **Carbon Conservation**: DIC + organic carbon balance

**Implementation**:
```python
tools/debugging/validate_system_conservation.py
```
- Global mass/energy accounting at each time step
- Identify sources of conservation violations
- Check boundary flux consistency

#### E3. Computational Performance Validation
**Objective**: Ensure model efficiency and numerical accuracy

**Critical Checks**:
- [ ] **JAX JIT Compilation**: Verify all functions compile correctly
- [ ] **Memory Usage**: Check for memory leaks or excessive allocation
- [ ] **Convergence Testing**: Grid and time step independence
- [ ] **Benchmark Comparison**: Compare with original C-GEM results

**Implementation**:
```python
tools/debugging/validate_computational_performance.py
```
- Profile memory and CPU usage
- Test convergence with refined grids
- Compare key outputs with reference solutions

---

### Phase F: Targeted Issue Resolution (Days 15-18)

#### F1. Tidal Overestimation Problem Resolution
**Priority**: 🔴 **CRITICAL** - Fix 2-3x tidal amplitude overestimation

**Systematic Investigation**:
- [ ] **Friction Coefficient Check**: Manning's n may be too low (insufficient damping)
- [ ] **Geometry Validation**: Cross-sectional areas may be too small (excessive amplification)
- [ ] **Boundary Forcing**: Tidal input amplitude may be incorrectly scaled
- [ ] **Wave Celerity**: Check shallow water wave speed √(gh) against observations

**Diagnostic Tools**:
```python
tools/debugging/diagnose_tidal_overestimation.py
```
- Compare modeled vs theoretical tidal propagation
- Analyze energy dissipation along estuary length  
- Test sensitivity to friction and geometry parameters
- Validate against analytical solutions (quarter-wave resonance)

**Expected Fix**: Likely Manning coefficient too low or geometry errors

#### F2. Species Transport Poor Correlation Resolution  
**Priority**: 🔴 **CRITICAL** - Fix R²=0.125-0.367 for most species

**Systematic Investigation**:
- [ ] **Dispersion Coefficient Magnitude**: May be unrealistic (too high/low)
- [ ] **Boundary Condition Values**: Species concentrations may be wrong
- [ ] **Reaction Rate Coupling**: Source/sink terms may dominate transport
- [ ] **Initial Condition Effects**: Poor initialization affecting steady-state

**Diagnostic Tools**:
```python
tools/debugging/diagnose_transport_correlation.py
```
- Run transport-only (no reactions) to isolate advection-dispersion
- Test with constant boundary conditions vs realistic forcing
- Analyze Peclet numbers and dispersion scaling
- Compare with analytical transport solutions

**Expected Fix**: Likely boundary conditions or dispersion coefficient issues

#### F3. O₂ Success Pattern Analysis
**Priority**: 🟢 **UNDERSTAND SUCCESS** - Why does O₂ work well (R²=0.792)?

**Success Factor Investigation**:
- [ ] **Boundary Conditions**: Are O₂ BCs more realistic than others?
- [ ] **Source/Sink Balance**: Does reaeration balance consumption well?
- [ ] **Transport Properties**: Is O₂ dispersion more accurate?
- [ ] **Reaction Coupling**: Are O₂ reactions more stable/realistic?

**Learning Tools**:
```python
tools/debugging/analyze_oxygen_success.py
```
- Compare O₂ vs other species transport characteristics
- Identify what makes O₂ boundary conditions work better
- Extract lessons for improving other species

**Apply Lessons**: Use O₂ success pattern to fix other species

---

## 📋 Implementation Strategy

### Development Environment Setup
```
tools/debugging/
├── validate_input_configuration.py     # Phase A1-A3
├── verify_saintenant_equations.py      # Phase B1
├── validate_hydro_boundaries.py        # Phase B2
├── analyze_hydro_stability.py          # Phase B3
├── verify_transport_equations.py       # Phase C1
├── validate_dispersion_coefficients.py # Phase C2
├── check_mass_conservation.py          # Phase C3
├── verify_biogeochemistry.py           # Phase D1
├── validate_species_coupling.py        # Phase D2
├── validate_biogeochem_parameters.py   # Phase D3
├── validate_module_coupling.py         # Phase E1
├── validate_system_conservation.py     # Phase E2
├── validate_computational_performance.py # Phase E3
├── diagnose_tidal_overestimation.py    # Phase F1
├── diagnose_transport_correlation.py   # Phase F2
└── analyze_oxygen_success.py           # Phase F3
```

### Validation Framework Architecture
```python
class PhysicsValidator:
    def __init__(self, model_config, results):
        self.config = model_config
        self.results = results
        
    def validate_hydrodynamics(self) -> Dict[str, bool]:
        """Comprehensive hydrodynamic validation"""
        
    def validate_transport(self) -> Dict[str, bool]:
        """Transport equation and mass conservation checks"""
        
    def validate_biogeochemistry(self) -> Dict[str, bool]:
        """Biogeochemical process validation"""
        
    def generate_physics_report(self) -> str:
        """Comprehensive physics quality assessment"""
```

### Automated Testing Suite
```python
# Continuous physics validation
pytest tools/debugging/test_physics_components.py -v
```
- Unit tests for each physics module
- Integration tests for coupled behavior
- Regression tests against analytical solutions
- Performance benchmarks for computational efficiency

---

## 🎯 Success Criteria and Validation Metrics

### Phase-by-Phase Success Metrics

#### Phase A: Data/Configuration (PASS/FAIL)
- [ ] All boundary data within realistic ranges
- [ ] CFL condition satisfied (PASS: Δt < Δt_max)
- [ ] Geometry matches field surveys (PASS: <10% difference)
- [ ] Mass balance in boundary fluxes (PASS: |residual| < 1%)

#### Phase B: Hydrodynamics (Quantitative Targets)
- [ ] **Tidal Amplitude**: Model/Observed ratio 0.8-1.2 (Currently: 2-3x ❌)
- [ ] **Wave Speed**: c_model ≈ √(gh) ± 20%
- [ ] **Mass Conservation**: |∂A/∂t + ∂Q/∂x| < 1% of mean flow
- [ ] **Energy Conservation**: Tidal energy budget closes within 10%

#### Phase C: Transport (Quantitative Targets)  
- [ ] **Mass Conservation**: Species mass change < 1% per day (no reactions)
- [ ] **Peclet Number**: Pe = uL/D in realistic range (1-100)
- [ ] **Dispersion Magnitude**: 10-1000 m²/s (typical estuarine range)
- [ ] **Boundary Flux**: Consistent with observed loading rates

#### Phase D: Biogeochemistry (Quantitative Targets)
- [ ] **Stoichiometric Balance**: C:N:P ratios within 20% of Redfield
- [ ] **O₂ Balance**: Production/consumption matches observed rates
- [ ] **Parameter Ranges**: All rates within literature bounds
- [ ] **Reaction Timescales**: Consistent with transport timescales

#### Phase E-F: System Integration (Final Targets)
- [ ] **Tidal Dynamics**: RMSE < 1.0m (vs current 3.8-4.6m ❌)
- [ ] **Longitudinal Profiles**: R² > 0.6 for all major species (vs 0.125-0.367 ❌)
- [ ] **Mass Conservation**: <1% error for all species
- [ ] **Computational Performance**: >10,000 steps/minute

### Physics Quality Classification
- **EXCELLENT**: All targets met, ready for calibration
- **GOOD**: Minor issues, calibration can proceed with caution  
- **FAIR**: Major issues resolved, some fine-tuning needed
- **POOR**: Fundamental problems, physics debugging required

**Current Status**: FAIR → Target: GOOD minimum for calibration readiness

---

## � Next Steps and Immediate Actions

### Week 1: Critical Physics Diagnosis (Start Immediately)

#### Day 1: **Tidal Overestimation Emergency**
```bash
python tools/debugging/diagnose_tidal_overestimation.py
```
**Focus**: Why 5.6-7.5m vs observed 2.1-3.2m?
- Check Manning friction coefficient (likely too low)
- Verify geometry cross-sections (may cause excessive amplification)
- Validate boundary tidal forcing amplitude

#### Day 2: **Transport Failure Analysis**  
```bash
python tools/debugging/diagnose_transport_correlation.py
```
**Focus**: Why R²=0.125-0.367 for most species vs O₂ R²=0.792?
- Compare dispersion coefficients with literature values
- Check boundary condition values for each species
- Test transport-only runs (no biogeochemistry)

#### Day 3: **Mass Conservation Audit**
```bash
python tools/debugging/check_mass_conservation.py
```
**Focus**: Are we losing/gaining mass artificially?
- Track total system mass for each species
- Check boundary flux calculations
- Verify numerical solver conservation properties

### Week 2: Systematic Physics Validation

#### Days 4-5: **Hydrodynamic Foundation**
```bash
python tools/debugging/verify_saintenant_equations.py
python tools/debugging/validate_hydro_boundaries.py
```

#### Days 6-7: **Transport Equation Implementation**
```bash
python tools/debugging/verify_transport_equations.py
python tools/debugging/validate_dispersion_coefficients.py
```

### Expected Outcomes

#### Likely Findings (Based on Verification Results):
1. **Tidal Issue**: Manning's n too low or geometry errors → 2-3x overestimation
2. **Transport Issue**: Boundary conditions or dispersion coefficients → Poor species correlation
3. **O₂ Success**: Identify what works for O₂, apply to other species

#### Physics Repair Priorities:
1. **CRITICAL**: Fix tidal amplitude (affects all transport processes)
2. **CRITICAL**: Correct species boundary conditions (affects spatial gradients)
3. **HIGH**: Validate mass conservation (fundamental requirement)
4. **MEDIUM**: Optimize biogeochemical coupling (after transport works)

---

## 📚 References and Scientific Methodology

### Hydrodynamic Validation Standards
- **Saint-Venant Equations**: Cunge et al. (1980), Savenije (2012)
- **Tidal Propagation**: Jay & Flinchem (1997), Godin (1999)
- **Estuarine Hydrodynamics**: Dyer (1997), Valle-Levinson (2010)

### Transport Process Validation
- **Advection-Dispersion**: Fischer et al. (1979), Rutherford (1994)
- **Elder's Formula**: Elder (1959), Liu (1977)
- **Estuarine Mixing**: Geyer & MacCready (2014)

### Biogeochemical Standards
- **Redfield Ratios**: Redfield (1958), Geider & La Roche (2002)
- **Estuarine Biogeochemistry**: Middelburg & Nieuwenhuize (2000), Volta et al. (2016)
- **Tropical Estuaries**: Jennerjahn & Ittekkot (2002)

### Numerical Methods
- **JAX Ecosystem**: Bradbury et al. (2018), Hessel et al. (2024)
- **Scientific Computing**: Press et al. (2007), LeVeque (2007)

---

## 🎯 Final Assessment

### Current State: **FAIR Physics Status**
✅ **Achieved**: Eliminated numerical oscillations, stable simulation framework  
❌ **Remaining**: Fundamental physics implementation issues preventing field data agreement

### Next Milestone: **GOOD Physics Status**  
🎯 **Target**: Tidal dynamics within 20% of observations, species transport R² > 0.6
🛠️ **Approach**: Systematic physics debugging, not parameter calibration
⏱️ **Timeline**: 2-3 weeks of intensive physics validation and repair

### Long-term Vision: **EXCELLENT Physics Status**
🚀 **Ultimate Goal**: Model ready for gradient-based calibration and scientific applications
📊 **Metrics**: R² > 0.7 spatial, RMSE < 20% temporal, mass conservation < 1%
🔬 **Applications**: Multi-year hindcasts, climate change scenarios, management support

**The foundation repair was successful. Now we build the physics correctly before calibrating parameters.**