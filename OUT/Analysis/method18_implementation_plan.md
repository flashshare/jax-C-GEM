
# METHOD 18 IMPLEMENTATION PLAN
=============================

## ROOT CAUSE IDENTIFIED:
The transport.py apply_boundary_conditions() function only handles salinity explicitly.
Other species (NH4, TOC, O2) are not getting boundary conditions applied!

## CRITICAL DISCOVERY:
- Line ~58: salinity_idx = 9  # Only salinity handled
- Lines ~63-80: Only salinity boundary logic implemented
- No universal species boundary condition application

## METHOD 18 SOLUTION:
1. Create universal boundary condition application for all species
2. Modify transport.py to call new universal function
3. Ensure all species get boundary values enforced at each timestep
4. Test with NH4, TOC, O2 target values

## IMPLEMENTATION STEPS:
1. ✅ Investigate current boundary condition implementation
2. ✅ Create universal boundary condition application function  
3. ⏳ Modify transport.py to use universal boundary conditions
4. ⏳ Test with METHOD 17 boundary values
5. ⏳ Run simulation and validate improvements
6. ⏳ Check validation metrics for breakthrough

## EXPECTED OUTCOMES:
- NH4: 9.5 → 23.0 mmol/m³ (boundary preservation)
- TOC: 2.7 → 500.0 mmol/m³ (massive improvement)
- O2: 210.1 → 75.0 mmol/m³ (overproduction control)
- Multiple NSE criteria breakthrough (2-3/15 → 5-8/15)

## CRITICAL SUCCESS FACTOR:
This addresses the fundamental issue that has prevented all previous
methods from working - only salinity was getting boundary conditions!
