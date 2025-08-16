# **COMPREHENSIVE ROADMAP: JAX C-GEM PHYSICS REPAIR**

## **CURRENT STATUS ANALYSIS (August 2025)**

**LATEST RUNTIME ANALYSIS REVEALS CRITICAL ISSUE:** 
- **Performance**: Excellent - 20,205 steps/minute (target achieved ✅)
- **Architecture**: JAX implementation functioning properly ✅
- **PHYSICS CATASTROPHE**: Complete hydrodynamic failure with NaN propagation 🔴

| Component | Expected Result | Actual Runtime | Status |
|-----------|-----------------|----------------|---------|
| Water Level (H) | Tidal range 0-8.86m | `Range: nan to nan` | 🔴 **COMPLETE FAILURE** |
| Velocity (U) | ±0.5-2.0 m/s with reversals | `Range: nan to nan` | 🔴 **NUMERICAL INSTABILITY** |
| Flow Reversal | `(U < 0).any() = True` | `False` (no valid data) | 🔴 **IMPOSSIBLE** |
| Transport | Species gradients | Working (despite NaN hydro) | 🟡 **TRANSPORT ISOLATED** |

**ROOT CAUSE**: Numerical instability in hydrodynamic solver causing NaN propagation. This is more severe than the originally reported "zero velocities" - the solver is completely unstable.

---

## **External Review vs. Runtime Reality Assessment**

**CRITICAL DISCREPANCY IDENTIFIED:** An external reviewer concluded the model is "scientifically sound and stable" based on static code analysis. However, **actual execution reveals catastrophic physics failures**:

| External Review Assessment | Actual Runtime Results | Reality Check |
|---------------------------|------------------------|---------------|
| "Hydrodynamic solver functioning correctly" | **Velocity = 0.000000 everywhere** | 🔴 **COMPLETELY BROKEN** |
| "Produces physically plausible results" | **No flow reversal, no tidal propagation** | 🔴 **PHYSICALLY IMPOSSIBLE** |
| "Ready for scientific calibration" | **Zero velocities prevent any transport** | 🔴 **NOT READY - PHYSICS BROKEN** |

**Conclusion**: The external review provides excellent architectural insights but missed the fundamental runtime failure. **Priority must remain on emergency physics repair before any architectural improvements.**

---

## **INTEGRATED ROADMAP: Physics Repair + Architectural Improvements**

Based on actual model execution and validated external review insights: # **DETAILED ROADMAP: STEP-BY-STEP PHYSICS REPAIR**

## **PHASE I: EMERGENCY PHYSICS REPAIRS** 🚨 (HIGHEST PRIORITY)

### **Task 1: Fix Critical Hydrodynamic Solver** (CONFIRMED CATASTROPHICALLY BROKEN)
**External Review**: "Hydrodynamic solver functioning correctly"  
**ACTUAL RUNTIME**: `U Range: nan to nan` and `H Range: nan to nan` ⚠️  
**Root Cause**: Numerical instability causing NaN propagation throughout hydrodynamic arrays - more severe than zero velocities

**Specific Actions**:
1. **Examine current `hydrodynamic_step` function** (line 482) - CONFIRMED: Contains artificial velocity clamping
2. **Replace momentum equation** with correct de Saint-Venant formulation
3. **Add convective acceleration terms**: `∂U/∂t + U∂U/∂x` (currently missing)
4. **Fix iterative solver** convergence criteria (TOL = 1e-6 too loose)

**Test**: Run model + verify `U` and `H` contain no NaN values, `U.max() > 0.5 m/s` and `(U < 0).any() = True`

---

### **Task 2: Fix Boundary Condition Implementation** (CONFIRMED BROKEN)
**External Review**: Did not test runtime boundary propagation  
**ACTUAL RUNTIME**: Tidal range 3.71m → 0.000m at 50km (should be exponential decay)  
**Root Cause**: `apply_boundary_conditions()` fails to propagate tidal energy

**Specific Actions**:
1. **Fix `apply_boundary_conditions()` function** - CONFIRMED: Blocks wave propagation
2. **Implement proper tidal forcing** at downstream boundary
3. **Remove artificial damping** that prevents inland propagation
4. **Test upstream discharge boundary** under working tidal conditions

**Test**: Verify tidal range decreases exponentially: 4m → 3m → 2m → 1m (not abrupt cutoff)

---

### **Task 3: Fix Friction Coefficient Calculation** (CONFIRMED BROKEN)
**External Review**: Did not analyze friction formula impact  
**ACTUAL RUNTIME**: Friction locks velocities to zero (infinite resistance)  
**Root Cause**: Incorrect friction implementation in momentum balance

**Specific Actions**:
1. **Correct friction formula**: Use `g/(Chezy²)` not current formula causing lock-up
2. **Validate Chezy coefficients** from model_config.txt (currently 40.0, 35.0)
3. **Implement proper bottom stress** calculation
4. **Test friction sensitivity** - should dampen but not eliminate flow

**Test**: Ensure friction allows ±1-2 m/s velocities with proper energy dissipation

---

## **PHASE II: VALIDATION OF PHYSICS FIXES** ✅

### **Task 4-5: Validate Hydrodynamic Repairs**
**Success Criteria** (Based on Actual Current Failures):
- **Phase 1**: Tidal amplitude > 1m at stations (currently 0.000m)
- **Phase 1**: Smooth salinity transition (currently sharp boundary at 27.5→0.2 psu)
- **Phase 2**: Flow reversal `(U < 0).any() = True` (currently False)
- **Phase 2**: Hovmöller diagonal propagation (currently vertical lines)

---

## **PHASE III: ARCHITECTURAL IMPROVEMENTS** 🏗️ (EXTERNAL REVIEW VALIDATED)

### **Task 6: Fix Critical Code Duplication** (EXTERNAL REVIEW CONFIRMED)
**External Finding**: "Severe redundancy between main.py and main_ultra_performance.py"  
**Impact**: Maintenance liability, 90% identical code

**Specific Actions**:
1. **Create shared utilities module** for argument parsing, config loading
2. **Consolidate result saving logic** - currently duplicated
3. **Refactor main scripts** to use shared functions
4. **Remove redundant code** (estimated 200+ lines of duplication)

### **Task 7: Consolidate Result Writing** (EXTERNAL REVIEW CONFIRMED)  
**External Finding**: "result_writer.py exists but is unused"  
**Impact**: Architectural inconsistency

**Specific Actions**:
1. **Move save_results_npz/csv** from main scripts to result_writer.py
2. **Update imports** in main scripts
3. **Remove duplicated saving functions**

---

## **PHASE IV: CONFIGURATION IMPROVEMENTS** ⚙️ (EXTERNAL REVIEW VALIDATED)

### **Task 8: Externalize Hardcoded Parameters** (EXTERNAL REVIEW CONFIRMED)
**External Finding**: Hardcoded boundary conditions and species bounds  
**Impact**: Hidden scientific assumptions

**Specific Actions**:
1. **Move default boundary conditions** from transport.py to model_config.txt
2. **Move species physical bounds** to model_config.py
3. **Update config parser** to handle new parameters

---

## **PHASE V: TRANSPORT & BIOGEOCHEMISTRY** 🌊 (BLOCKED UNTIL HYDRO WORKS)

### **Task 9-11: Transport System Repair**
**Current Issue**: Cannot test transport with zero velocities  
**Prerequisite**: Working hydrodynamic solver producing realistic flows

### **Task 12-14: Long-term Validation**
**Current Issue**: Model appears stable but with wrong physics  
**Prerequisite**: Working physics before biogeochemical tuning

---

## **SUCCESS METRICS & VALIDATION STRATEGY**

### **Phase I Success Criteria** (Emergency Physics):
- **NO NaN VALUES**: `H` and `U` arrays must contain only finite numbers (currently all NaN)
- `U.max() > 0.5 m/s` (currently NaN, impossible to test)
- `(U < 0).any() = True` (flow reversal, currently False due to NaN)
- Tidal range propagates exponentially, not abruptly cut off

### **Phase II Success Criteria** (Validation):
- Phase 1: Tidal amplitude > 1m at stations (currently 0.000m)
- Phase 2: Hovmöller diagonal patterns (currently vertical lines)
- Flow reversal in time series plots

### **Phase III Success Criteria** (Architecture):
- Code duplication eliminated (<50% overlap between main scripts)
- All hardcoded parameters externalized to config files
- Consolidated result writing system

### **Phase IV Success Criteria** (Transport):
- Smooth salinity profiles with realistic mixing
- Mass conservation errors < 1%
- Working advection-dispersion transport

### **Phase V Success Criteria** (Long-term):
- 2-year simulation stability without crashes
- Realistic biogeochemical cycles
- Ready for scientific calibration

---

## **IMPLEMENTATION APPROACH - VALIDATED BY EXTERNAL REVIEW**

### **Why Physics Must Come First**:
The external reviewer correctly identified excellent architectural foundations:
- ✅ "Full vectorization" - confirmed, no Python loops in numerical kernels
- ✅ "JIT compilation correctly applied" - confirmed, optimal performance design
- ✅ "Functional purity" - confirmed, stateless core functions
- ✅ "High-fidelity scientific implementation" - structure is correct

**However**: Even perfect architecture cannot overcome zero velocities everywhere. The physics failure is a **runtime catastrophe** that static analysis cannot detect.

### **Post-Physics Integration Plan**:
Once the hydrodynamic solver produces realistic velocities:
1. **Immediate**: Apply external review's architectural improvements
2. **Validate**: Confirm transport works with realistic flow field  
3. **Optimize**: Implement calibration infrastructure leveraging excellent JAX foundation
4. **Scale**: Use vectorization and JIT performance for large-scale studies

### **External Review Value**:
- **Architectural guidance**: Will be implemented in Phase III
- **Code quality insights**: Excellent foundation confirmed
- **Performance validation**: JAX paradigm perfectly executed
- **Limitation**: Static analysis missed critical runtime failure

---

## **READY TO BEGIN: TASK 1 - HYDRODYNAMIC SOLVER EMERGENCY REPAIR**

The external review confirms our JAX implementation is excellent. Now we need to make it **physically realistic**.

**Next Action**: Examine the current `hydrodynamic_step` function to understand why it produces zero velocities everywhere, then implement a correct momentum solver that achieves realistic tidal flow reversal.

---

## **EXECUTION STRATEGY - UPDATED WITH EXTERNAL REVIEW**

### **Critical Priority Order**:
1. **🚨 EMERGENCY**: Fix zero velocity crisis (Tasks 1-3)
2. **✅ VALIDATE**: Confirm physics work (Tasks 4-5)  
3. **🏗️ ARCHITECTURE**: Apply external review improvements (Tasks 6-8)
4. **🌊 TRANSPORT**: Complete system integration (Tasks 9-14)

### **External Review Integration**:
- **Validated Concerns**: Code duplication, unused modules, hardcoded parameters
- **Missed Critical Issue**: Zero velocity runtime failure (static analysis limitation)
- **Architectural Strengths**: JAX paradigm adherence, vectorization, JIT compilation
- **Recommended Approach**: Fix physics first, then apply architectural improvements

### **Next Steps**:
The external reviewer's architectural insights are valuable and will be integrated **after** the emergency physics repairs. The model's excellent JAX implementation and performance optimization provide a solid foundation once the fundamental momentum solver is corrected.

**REALITY CHECK**: No amount of architectural improvement can fix a model that produces zero velocities everywhere. Physics repair remains the absolute priority.

---

## **PHASE IV: FINAL SYSTEM VALIDATION** 🔬

### **Task 8-10: Complete System Testing**
- **Mass conservation validation**
- **Biogeochemical rate optimization**
- **Long-term stability testing**

### **Task 11-12: Performance Enhancements**
- **Adaptive time stepping**
- **Gradient-based calibration**

### **Task 13: Code Cleanup**
- **Remove deprecated files**
- **Fix import errors**
- **Consolidate engines**

---

# **EXECUTION STRATEGY**

## **Immediate Next Steps**:

1. **START WITH TASK 1**: Fix the hydrodynamic solver
2. **Test after each fix**: Run model → check velocities → run validation
3. **Don't proceed** until each phase shows physical realism
4. **Use actual diagnostic values** to measure success

## **Success Metrics Per Phase**:

**Phase I Success**: 
- `U.max() > 0.5 m/s`
- `(U < 0).any() = True` (flow reversal)
- Tidal range propagates beyond 50km

**Phase II Success**:
- Phase 1 validation: Tidal amplitude > 0m at all stations
- Phase 2 validation: Hovmöller shows diagonal patterns

**Phase III Success**:
- Smooth salinity profiles
- No artificial boundaries
- Mass conservation < 1% error

**Phase IV Success**:
- 2-year simulation completes without crashes
- Biogeochemical cycles remain stable
- Model ready for scientific calibration

---