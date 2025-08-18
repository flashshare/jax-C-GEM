# JAX C-GEM Publication Roadmap
**Target: Scientific Publication Ready Model**

---

## Executive Summary

Based on the comprehensive audit conducted on August 17, 2025, the JAX C-GEM model exhibits excellent software architecture but suffers from **critical biogeochemical failures** that prevent scientific application. This roadmap provides a systematic path to publication readiness through targeted fixes and validation.

**Current Status:** 🔴 **NOT READY** - Critical dissolved oxygen system failure  
**Estimated Timeline:** 6-8 weeks to publication readiness  
**Priority:** High-impact fixes first, systematic validation, comprehensive documentation

---

## Phase I: CRITICAL FIXES (Weeks 1-2)
**Status: 🚨 EMERGENCY PRIORITY**

### Task 1.1: Dissolved Oxygen Crisis Resolution
**Timeline: 3-4 days**  
**Priority: CRITICAL**  
**Responsible: Lead Developer**

#### Problem Statement
Model predicts 0.003 mg/L dissolved oxygen throughout estuary vs. expected 5-6 mg/L, indicating complete biogeochemical collapse.

#### Specific Actions
1. **Debug oxygen mass balance** in `src/core/biogeochemistry.py`:
   ```bash
   # Create debugging script
   python tools/debugging/oxygen_mass_balance_debugger.py
   ```
   - [ ] Add oxygen production/consumption tracking
   - [ ] Verify primary production O2 generation
   - [ ] Check respiration rates vs. literature values
   - [ ] Validate temperature factors (Q10 coefficients)

2. **Implement atmospheric reaeration** if missing:
   - [ ] Add wind-driven oxygen exchange: `k_O2 = f(wind_speed, temperature)`
   - [ ] Implement O'Connor-Dobbins reaeration model
   - [ ] Validate against published reaeration coefficients

3. **Validate biogeochemical parameters**:
   - [ ] Compare all rates with Volta et al. (2016) and Savenije (2012)
   - [ ] Check Redfield ratios: C:N:P = 106:16:1
   - [ ] Verify species coupling (O2 ↔ NH4, O2 ↔ NO3)

#### Success Criteria
- [ ] Dissolved oxygen levels: 4-8 mg/L (realistic estuarine range)
- [ ] Spatial gradient: higher upstream, lower downstream
- [ ] Mass balance closure: <5% error over simulation period

#### Deliverables
- [ ] `oxygen_crisis_diagnosis_report.md`
- [ ] Fixed `biogeochemistry.py` with validated parameters
- [ ] `oxygen_validation_plots.png` showing before/after

### Task 1.2: Biogeochemical Parameter Calibration
**Timeline: 2-3 days**  
**Priority: HIGH**  
**Responsible: Biogeochemist**

#### Actions
1. **Literature parameter review**:
   - [ ] Extract parameters from Volta et al. (2016) Table 2
   - [ ] Compare with current `DEFAULT_BIO_PARAMS` in `model_config.py`
   - [ ] Document parameter sources and ranges

2. **Implement parameter bounds**:
   ```python
   # Add to model_config.py
   BIOGEO_PARAM_BOUNDS = {
       'mumax_phy1': (0.5, 3.0),  # day⁻¹
       'mumax_phy2': (0.5, 3.0),  # day⁻¹
       'resp': (0.01, 0.1),       # day⁻¹
       'mort': (0.01, 0.1),       # day⁻¹
   }
   ```

3. **Create sensitivity analysis**:
   - [ ] Use JAX autodiff for parameter sensitivity
   - [ ] Generate `biogeochemical_sensitivity_analysis.py`
   - [ ] Identify most influential parameters

#### Success Criteria
- [ ] All parameters within literature ranges
- [ ] Sensitivity analysis complete
- [ ] Parameter uncertainty quantified

### Task 1.3: Mass Balance Validation
**Timeline: 1-2 days**  
**Priority: HIGH**  
**Responsible: Numerical Analyst**

#### Actions
1. **Implement mass balance checking**:
   ```python
   # Create tools/debugging/mass_balance_checker.py
   def check_species_mass_balance(concentrations, fluxes, reactions, dt):
       # Track mass in/out for each species
       # Report violations >1%
   ```

2. **Add conservation diagnostics**:
   - [ ] Track total nitrogen (NH4 + NO3 + Organic N)
   - [ ] Track total phosphorus (PO4 + Organic P + PIP)
   - [ ] Track total carbon (DIC + TOC + Biomass C)

#### Success Criteria
- [ ] Mass balance errors <1% per time step
- [ ] No long-term accumulation/loss
- [ ] Conservation diagnostics pass

---

## Phase II: HYDRODYNAMIC CALIBRATION (Week 3)
**Status: ⚠️ MODERATE PRIORITY**

### Task 2.1: Tidal Range Correction
**Timeline: 3-4 days**  
**Priority: MODERATE**  
**Responsible: Hydrodynamicist**

#### Problem Statement
Model predicts 5.6-5.8 m tidal range vs. observed 2.1-3.3 m (systematic 2x overestimate).

#### Actions
1. **Friction coefficient adjustment**:
   - [ ] Reduce Chezy coefficients in `config/model_config.txt`:
     ```
     Chezy1 = 15.0  # Reduced from 25.0
     Chezy2 = 10.0  # Reduced from 20.0
     ```

2. **Channel geometry validation**:
   - [ ] Compare width/depth profiles with field bathymetry
   - [ ] Validate convergence lengths LC1, LC2 against observations
   - [ ] Check if B1, B2 values match measured channel widths

3. **Tidal propagation analysis**:
   - [ ] Calculate theoretical tidal wave speed: c = √(gH)
   - [ ] Verify phase lag between stations matches observations
   - [ ] Check for unrealistic resonance amplification

#### Success Criteria
- [ ] Tidal ranges within ±30% of observations
- [ ] Spatial phase relationships correct
- [ ] No unrealistic amplification patterns

### Task 2.2: Flow Reversal Validation
**Timeline: 1-2 days**  
**Priority: MODERATE**

#### Actions
1. **Velocity amplitude analysis**:
   - [ ] Extract velocity time series at key stations
   - [ ] Verify flow reversal occurs at all locations
   - [ ] Check velocity-water level phase relationships

2. **Momentum balance verification**:
   - [ ] Validate pressure gradient vs. friction terms
   - [ ] Check if Coriolis effects needed
   - [ ] Verify boundary condition implementation

#### Success Criteria
- [ ] Flow reversal present at all stations
- [ ] Realistic velocity amplitudes (0.5-2.0 m/s)
- [ ] Proper phase relationships

---

## Phase III: FIELD DATA VALIDATION (Weeks 4-5)
**Status: 📊 VALIDATION FOCUS**

### Task 3.1: Three-Phase Validation Campaign
**Timeline: 1 week**  
**Priority: HIGH**  
**Responsible: Validation Team**

#### Phase 1: Longitudinal Profiles
- [ ] Achieve R² > 0.7 for salinity profiles
- [ ] Achieve R² > 0.5 for major nutrients (NH4, NO3, PO4)
- [ ] RMSE within 50% of observed ranges
- [ ] Proper salt intrusion length (~40 km)

#### Phase 2: Tidal Dynamics  
- [ ] Tidal range R² > 0.6 at all three stations
- [ ] Systematic bias <50%
- [ ] Proper spatial tidal amplification pattern

#### Phase 3: Seasonal Cycles
- [ ] Clear seasonal patterns (R² > 0.4)
- [ ] Dry/wet season differentiation
- [ ] No unrealistic extremes or crashes

### Task 3.2: Statistical Validation Framework
**Timeline: 2-3 days**  
**Priority: MODERATE**

#### Actions
1. **Implement sparse data methodology**:
   - [ ] Focus on statistical aggregates vs. point comparisons
   - [ ] Weight errors by measurement uncertainty
   - [ ] Compare monthly means, standard deviations

2. **Advanced metrics implementation**:
   - [ ] Nash-Sutcliffe Efficiency (NSE)
   - [ ] Kling-Gupta Efficiency (KGE)
   - [ ] Index of Agreement (IOA)
   - [ ] Bias ratio and variability ratio

#### Success Criteria
- [ ] NSE > 0.5 for major variables
- [ ] KGE > 0.6 for key species
- [ ] Bias ratios within 0.7-1.3 range

### Task 3.3: Phase III Validation Results Analysis
**Timeline: COMPLETED**  
**Status: 🚨 CRITICAL ISSUES IDENTIFIED**  

#### Validation Results Summary
**COMPLETED: August 17, 2025**

**Overall Performance:**
- NSE > 0.5: **0/15 validations (0.0%)** ❌
- KGE > 0.6: **0/15 validations (0.0%)** ❌  
- R² > 0.5: **2/15 validations (13.3%)** ❌

#### Critical Findings

**1. Biogeochemical System Collapse:**
- TOC: NSE = -25.595 (28x magnitude error)
- O2: NSE = -90.571 (oxygen system failure)  
- PO4: NSE = -17.357 (phosphorus dynamics broken)

**2. Hydrodynamic Over-Amplification:**
- PC: 208.2% tidal error (6.55m vs 2.12m)
- BD: 148.5% tidal error (7.31m vs 2.94m)
- BK: 87.6% tidal error (6.14m vs 3.27m)

**3. Partial Transport Success:**
- Salinity: R² = 0.937 ✅ (excellent spatial structure)
- SPM: R² = 0.773 ✅ (good sediment transport)

### Task 3.4: Root Cause Diagnosis & Action Plan
**Timeline: COMPLETED**
**Status: 📋 SYSTEMATIC FIXES IDENTIFIED**

#### Priority 1: Biogeochemical Emergency Fixes
**Root Causes Identified:**
- Missing atmospheric oxygen reaeration
- Incorrect Q10 temperature factors
- Wrong TOC mineralization rates
- Broken O2-organic matter coupling

**Specific Actions Required:**
```python
# src/core/biogeochemistry.py fixes needed:
1. Implement k_reaeration = f(wind_speed, temperature)
2. Fix Q10 coefficients for all reactions
3. Correct TOC → CO2 + O2 stoichiometry  
4. Validate all rate constants against literature
```

#### Priority 2: Hydrodynamic Amplitude Correction
**Root Causes Identified:**
- Inadequate friction despite Phase II improvements
- Possible boundary condition errors (AMPL=4.43m)
- Numerical amplification in shallow regions

**Specific Actions Required:**
```python
# src/core/hydrodynamics.py fixes needed:
1. Increase friction: Chezy1=150→250, Chezy2=200→300
2. Implement depth-dependent friction formulation
3. Add wave energy dissipation terms
4. Validate boundary forcing amplitude
```

#### Priority 3: Transport Calibration Refinement
**Root Causes Identified:**
- Species-specific dispersion not implemented
- Tributary input magnitudes questionable
- Boundary concentrations need validation

---

## Phase IV: SYSTEMATIC MODEL IMPROVEMENTS (Week 6)
**Status: 🔧 IMPLEMENTATION READY**

### Task 4.1: Biogeochemical System Reconstruction
**Timeline: 3-4 days**  
**Priority: 🚨 EMERGENCY**  

#### Oxygen System Emergency Fix
- [ ] Implement atmospheric reaeration: `k_O2 = f(wind, temp, salinity)`
- [ ] Fix temperature dependencies: `rate = rate_20C * Q10^((T-20)/10)`
- [ ] Correct O2 consumption stoichiometry for all processes
- [ ] Add surface oxygen exchange boundary condition

#### Organic Carbon Cycling Repair
- [ ] Fix TOC mineralization: `TOC + O2 → CO2 + biomass`
- [ ] Implement proper C:N:P ratios (Redfield: 106:16:1)
- [ ] Add temperature-dependent decay rates
- [ ] Validate against Volta et al. (2016) parameters

#### Success Criteria
- [ ] O2 levels: 3-8 mg/L (realistic estuarine range)
- [ ] TOC magnitude within 3x of observations
- [ ] NSE > 0.3 for all biogeochemical variables
- [ ] No negative concentrations or crashes

### Task 4.2: Hydrodynamic Amplitude Calibration  
**Timeline: 2-3 days**
**Priority: 🚨 EMERGENCY**

#### Enhanced Friction Implementation
- [ ] Increase base friction: Chezy1=250, Chezy2=300
- [ ] Implement depth-dependent formulation: `f = f0 * (h/h_ref)^n`
- [ ] Add vegetation/roughness effects in shallow areas
- [ ] Test spatial friction variation

#### Boundary Condition Validation
- [ ] Validate AMPL=4.43m against field measurements
- [ ] Implement reflection coefficient: `R = (A_up - A_down)/(A_up + A_down)`
- [ ] Check phase relationships in tidal forcing
- [ ] Add energy dissipation at boundaries

#### Success Criteria  
- [ ] Tidal errors < 100% at all stations
- [ ] Proper spatial amplification pattern
- [ ] No unrealistic velocity spikes
- [ ] Mass conservation maintained

### Task 4.3: Transport Physics Enhancement
**Timeline: 1-2 days**
**Priority: 🔶 HIGH**

#### Species-Specific Transport  
- [ ] Implement variable dispersion: `D_species = D_base * factor_species`
- [ ] Add molecular diffusion for nutrients
- [ ] Check tributary input concentrations vs. field data
- [ ] Validate boundary condition magnitudes

#### Numerical Accuracy Improvements
- [ ] Check CFL stability conditions
- [ ] Implement higher-order advection schemes if needed
- [ ] Add mass balance verification
- [ ] Test grid resolution sensitivity

### Task 4.4: Integrated System Validation
**Timeline: 1-2 days** 
**Priority: 🔶 HIGH**

#### Comprehensive Re-Testing
- [ ] Run full validation suite after each fix
- [ ] Monitor system stability and mass balance
- [ ] Check for new coupling issues
- [ ] Validate against all three field datasets

#### Performance Verification
- [ ] Achieve NSE > 0.5 for primary variables
- [ ] Achieve KGE > 0.6 for key species
- [ ] Maintain R² > 0.7 for salinity
- [ ] Reduce tidal errors to < 50%

---

## Phase V: PUBLICATION-READY VALIDATION (Week 7)
**Status: 🎯 AWAITING PHASE IV COMPLETION**

### Task 5.1: Final Validation Campaign
**Timeline: 2-3 days**
**Prerequisites: Phase IV fixes complete**

#### Complete Statistical Assessment  
- [ ] Re-run advanced statistical validation
- [ ] Generate publication-quality figures
- [ ] Calculate confidence intervals and uncertainty bounds
- [ ] Perform sensitivity analysis on key parameters

#### Multi-Dataset Cross-Validation
- [ ] Validate against all three datasets simultaneously
- [ ] Check temporal consistency (2017-2018 coverage)
- [ ] Verify spatial patterns across estuary length
- [ ] Test seasonal cycle reproduction

#### Success Criteria (Publication Ready)
- [ ] NSE > 0.5 for all major variables
- [ ] KGE > 0.7 for primary species (NH4, NO3, PO4, O2)
- [ ] R² > 0.7 for spatial profiles
- [ ] R² > 0.4 for seasonal cycles
- [ ] Tidal dynamics within 25% error

### Task 5.2: Scientific Documentation
**Timeline: 2 days**
**Priority: 📝 PUBLICATION PREP**

#### Comprehensive Method Documentation
- [ ] Document all calibration procedures
- [ ] Explain sparse data validation methodology  
- [ ] Detail model improvements and their scientific basis
- [ ] Create reproducibility documentation

#### Results Presentation
- [ ] Generate multi-panel validation figures
- [ ] Create performance summary tables
- [ ] Develop error analysis and uncertainty quantification
- [ ] Prepare supplementary materials

---

## EXPECTED OUTCOMES

**Week 6 End State:**
- Functional biogeochemical system with realistic O2/TOC dynamics
- Properly calibrated hydrodynamics with <100% tidal errors
- Enhanced transport physics with species-specific parameters

**Week 7 End State:**  
- Publication-ready model with NSE > 0.5 for major variables
- Comprehensive validation against three independent datasets
- Full documentation and reproducibility package
- Ready for scientific journal submission

**Publication Impact:**
- First JAX-based estuarine biogeochemical model
- Advanced sparse data validation methodology  
- High-performance computing for environmental science
- Transferable framework for global estuary applications

The systematic approach outlined above transforms the current research prototype into a scientifically rigorous, publication-ready model through targeted fixes based on comprehensive validation results.

## Phase IV: CALIBRATION & OPTIMIZATION (Week 6)
**Status: 🎯 SYSTEMATIC CALIBRATION**

### Task 4.1: JAX-Native Gradient-Based Calibration
**Timeline: 4-5 days**  
**Priority: HIGH**  
**Responsible: Optimization Specialist**

#### Actions
1. **Implement calibration framework**:
   ```python
   # Extend tools/calibration/gradient_calibrator.py
   @jax.jit
   def objective_function(params, model_state, field_data):
       # Multi-faceted error: profiles + seasonal + variability
       return weighted_rmse_sum
   
   def calibrate_model():
       grad_fn = jax.grad(objective_function)
       # Use Optimistix or JAXopt
   ```

2. **Parameter selection**:
   - [ ] Select 8-12 most sensitive parameters
   - [ ] Focus on biogeochemical rates and transport coefficients
   - [ ] Avoid over-parameterization

3. **Multi-objective optimization**:
   - [ ] Weight spatial vs. temporal errors
   - [ ] Balance different species importance
   - [ ] Include uncertainty in objective function

#### Success Criteria
- [ ] Automated parameter optimization working
- [ ] Convergence achieved within bounds
- [ ] Cross-validation performance good

### Task 4.2: Uncertainty Quantification
**Timeline: 2-3 days**  
**Priority: MODERATE**

#### Actions
1. **Parameter uncertainty**:
   - [ ] Monte Carlo parameter sampling
   - [ ] Confidence intervals for predictions
   - [ ] Sensitivity indices

2. **Model structural uncertainty**:
   - [ ] Compare against simpler models
   - [ ] Assess prediction intervals
   - [ ] Document model limitations

---

## Phase V: DOCUMENTATION & PUBLICATION (Weeks 7-8)
**Status: 📝 PUBLICATION PREPARATION**

### Task 5.1: Scientific Documentation
**Timeline: 1 week**  
**Priority: HIGH**  
**Responsible: Lead Author**

#### Manuscript Structure
1. **Abstract** (250 words)
   - [ ] Model innovation (JAX-native, gradient-optimized)
   - [ ] Validation against 3 field datasets
   - [ ] Key results and performance metrics

2. **Introduction** (800 words)
   - [ ] Literature review: estuarine modeling challenges
   - [ ] Sparse data calibration methodology
   - [ ] JAX computational advantages

3. **Methods** (1500 words)
   - [ ] Governing equations (de Saint-Venant, advection-dispersion, biogeochemistry)
   - [ ] Numerical methods and JAX implementation
   - [ ] Calibration methodology and objective function
   - [ ] Validation datasets and metrics

4. **Results** (1200 words)
   - [ ] Three-phase validation results
   - [ ] Performance benchmarks vs. original C-GEM
   - [ ] Parameter sensitivity analysis
   - [ ] Uncertainty quantification

5. **Discussion** (800 words)
   - [ ] Model capabilities and limitations
   - [ ] Comparison with other estuarine models
   - [ ] Future applications and extensions

#### Figures (6-8 publication-quality)
- [ ] Figure 1: Model domain and validation stations
- [ ] Figure 2: Longitudinal profiles (Phase 1 validation)
- [ ] Figure 3: Tidal dynamics (Phase 2 validation)  
- [ ] Figure 4: Seasonal cycles (Phase 3 validation)
- [ ] Figure 5: Parameter sensitivity analysis
- [ ] Figure 6: Performance comparison (JAX vs. original)
- [ ] Figure 7: Uncertainty quantification
- [ ] Figure 8: Model applications example

### Task 5.2: Code and Data Publication
**Timeline: 2-3 days**  
**Priority: HIGH**

#### Actions
1. **Code repository preparation**:
   - [ ] Complete documentation in `docs/`
   - [ ] API reference with examples
   - [ ] Installation and quickstart guides
   - [ ] Comprehensive test suite

2. **Data package**:
   - [ ] Validation datasets properly formatted
   - [ ] Model configuration files documented
   - [ ] Example applications included

3. **Reproducibility package**:
   - [ ] Docker container with environment
   - [ ] Jupyter notebooks for key analyses
   - [ ] Benchmark scripts and expected outputs

#### Deliverables
- [ ] GitHub repository: `flashshare/jax-cgem-publication`
- [ ] Zenodo data package with DOI
- [ ] Comprehensive `README.md` with citations
- [ ] `CONTRIBUTING.md` for community engagement

---

## Phase VI: PEER REVIEW & REVISION (Weeks 9-12)
**Status: 🔄 ITERATIVE IMPROVEMENT**

### Task 6.1: Journal Submission
**Target Journals** (in priority order):
1. **Water Resources Research** (AGU) - High impact, computational methods focus
2. **Journal of Computational Physics** - JAX methodology emphasis  
3. **Estuarine, Coastal and Shelf Science** - Application domain focus
4. **Environmental Modelling & Software** - Software contribution angle

### Task 6.2: Review Response Protocol
- [ ] Rapid response framework (<30 days)
- [ ] Additional validation if requested
- [ ] Code improvements based on feedback
- [ ] Extended applications if needed

---

## Success Metrics & KPIs

### Technical Performance
- [ ] **Dissolved Oxygen**: 4-8 mg/L realistic range achieved
- [ ] **Tidal Validation**: R² > 0.6, bias < 50%
- [ ] **Nutrient Cycles**: R² > 0.5 for NH4, NO3, PO4
- [ ] **Mass Balance**: Errors < 1% per timestep
- [ ] **Performance**: >20,000 steps/minute maintained

### Scientific Quality
- [ ] **Validation Coverage**: 3 independent datasets
- [ ] **Statistical Rigor**: NSE > 0.5, KGE > 0.6
- [ ] **Uncertainty**: Confidence intervals provided
- [ ] **Reproducibility**: Complete code/data package
- [ ] **Innovation**: JAX-native implementation demonstrated

### Publication Impact
- [ ] **Target Journal**: Water Resources Research submission
- [ ] **Citations Ready**: >50 relevant references
- [ ] **Code Sharing**: Open source GitHub repository
- [ ] **Data Sharing**: FAIR principles compliance
- [ ] **Community**: User documentation and examples

---

## Risk Management

### High-Risk Items
1. **Oxygen Crisis Resolution** - May require fundamental biogeochemical restructuring
   - *Mitigation*: Parallel development of simplified O2-only model
   
2. **Field Data Alignment** - Sparse observations may limit validation quality
   - *Mitigation*: Focus on statistical aggregates, not point comparisons
   
3. **Computational Performance** - Calibration may be computationally intensive
   - *Mitigation*: Use efficient JAX optimizers, parameter subset selection

### Contingency Plans
- **Plan B**: Submit to Environmental Modelling & Software if technical issues persist
- **Plan C**: Focus on methodology paper if validation remains challenging
- **Plan D**: Conference presentation route if journal timeline extends

---

## Resource Requirements

### Personnel (FTE weeks)
- **Lead Developer**: 4 weeks (biogeochemistry, calibration)
- **Validation Specialist**: 2 weeks (field data analysis)
- **Technical Writer**: 1 week (documentation)
- **Total**: 7 person-weeks

### Computational Resources
- **Development**: Standard laptop sufficient (JAX CPU mode)
- **Calibration**: Multi-core workstation recommended
- **Large Simulations**: Consider cloud computing for extensive validation

### Data Requirements
- **Field Data**: Already available (CARE, CEM, SIHYMECC)
- **Additional Data**: None required
- **Storage**: <10 GB total for all simulations and results

---

## Conclusion

This roadmap provides a systematic path from the current critical failure state to a publication-ready JAX C-GEM model. The key insight from the audit is that the excellent software architecture provides a solid foundation for rapid debugging and improvement once the biogeochemical crisis is resolved.

**Success is achievable within 6-8 weeks** with focused effort on the prioritized tasks. The JAX-native framework will enable efficient gradient-based calibration and optimization, leading to a novel contribution to the estuarine modeling literature.

**Timeline Summary:**
- **Weeks 1-2**: Fix critical dissolved oxygen crisis
- **Week 3**: Calibrate hydrodynamics  
- **Weeks 4-5**: Complete field validation
- **Week 6**: Systematic optimization
- **Weeks 7-8**: Publication preparation
- **Weeks 9-12**: Peer review process

The model's potential for high-impact publication is strong, given its innovative JAX implementation and comprehensive validation against three independent field datasets.