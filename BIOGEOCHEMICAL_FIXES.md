🔬 BIOGEOCHEMICAL FIXES: Enable Seasonal Dynamics & Proper Cycling

## 🚨 Critical Issues Identified & Fixed:

### 1. **TOC Completely Static (CV = 0.00%)**
**Problem**: TOC was clamped to exactly 1.0 mmol/m³ due to tight minimum bound
**Solution**: Relaxed TOC bounds from [1.0, 5000.0] to [0.1, 5000.0]
**Impact**: Allows natural organic carbon cycling and seasonal variation

### 2. **No Spatial Gradients in Biogeochemistry** 
**Problem**: All species except salinity were spatially uniform
**Solution**: Enhanced initial conditions with realistic estuarine gradients
**Impact**: Creates proper nutrient/oxygen gradients correlated with salinity

### 3. **Over-Constrained Species Bounds**
**Problem**: Tight minimum bounds prevented natural variability
**Solutions Applied**:
- PHY1/PHY2: Relaxed to [0.001, 500.0] (allow near-zero phytoplankton)
- NO3/NH4: Reduced minimums to [0.01, 1000.0] and [0.01, 500.0]
- PO4: Reduced to [0.001, 100.0] (allow very low phosphate)
- O2: Reduced to [0.1, 500.0] (allow anoxic conditions)
- SPM: Reduced to [0.1, 1000.0] (allow clear water)

### 4. **Improved Initial Conditions**
**Replaced**: Uniform initial values for all species
**With**: Realistic estuarine gradients:
- **Oxygen**: 180 mmol/m³ (river) → 280 mmol/m³ (marine)
- **TOC**: 200 mmol/m³ (river) → 50 mmol/m³ (marine) 
- **NO3**: 25 mmol/m³ (river) → 5 mmol/m³ (marine)
- **NH4**: 5.0 mmol/m³ (river) → 0.5 mmol/m³ (marine)
- **PO4**: 3.0 mmol/m³ (river) → 0.2 mmol/m³ (marine)
- **SPM**: 100 mg/L (river) → 20 mg/L (marine)

## 📊 Expected Improvements:

1. **TOC Dynamic Range**: From static 1.000 to seasonal cycling
2. **Spatial Gradients**: Proper biogeochemical gradients along estuary
3. **Seasonal Responsiveness**: Increased CV values for all species
4. **Validation Metrics**: Improved R² values against field observations
5. **Biogeochemical Coupling**: Better transport-biogeochemistry interaction

## 🔬 Scientific Rationale:

These changes restore the natural biogeochemical behavior of estuarine systems:
- **Organic Carbon Cycling**: TOC can now respond to production/degradation
- **Nutrient Dynamics**: Proper gradients reflect mixing and biogeochemical processes
- **Redox Conditions**: Oxygen can vary naturally, enabling anaerobic processes
- **Phytoplankton Dynamics**: Can respond to light/nutrient availability

## 🎯 Validation Impact:
Expected to significantly improve Phase 3 seasonal validation and statistical metrics by enabling realistic biogeochemical variability and proper estuarine gradients.