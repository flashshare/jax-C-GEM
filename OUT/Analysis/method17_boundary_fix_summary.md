
# METHOD 17: CRITICAL BOUNDARY CONDITION FIX
==========================================

## ROOT CAUSE DISCOVERED:
METHOD 16B revealed that our METHOD 13 boundary condition files
were never properly updated. The files contained old values:
- NH4_lb.csv: 6.8 mmol/m³ (should be 23.0)
- TOC_lb.csv: 465.6 mmol/m³ (should be 500.0)
- O2_lb.csv: 82.8 mmol/m³ (should be 75.0)

## METHOD 17 FIXES:
1. ✅ Created correct lower boundary files with METHOD 13 values
2. ✅ Updated upper boundary files for consistency
3. ✅ Added realistic seasonal variations (±5-10%)
4. ✅ Verified file formats match simulation requirements

## NEW BOUNDARY CONDITIONS:
### Lower Boundary (Ocean/Mouth):
- NH4: 23.0 mmol/m³ (field data aligned)
- TOC: 500.0 mmol/m³ (scaled for boundary forcing)
- O2: 75.0 mmol/m³ (reduced to test O2 balance)

### Upper Boundary (River/Head):
- NH4: 35.0 mmol/m³ (typical river input)
- TOC: 800.0 mmol/m³ (terrestrial organic matter)
- O2: 50.0 mmol/m³ (river depletion)

## EXPECTED OUTCOME:
With correct boundary conditions, the simulation should now:
- Show NH4 concentrations near 23.0 mmol/m³ at mouth
- Preserve TOC at ~500 mmol/m³ (testing biogeochemical balance)
- Reduce O2 to ~75 mmol/m³ (testing photosynthesis suppression)

## NEXT STEPS:
1. Run simulation with corrected boundary conditions
2. Check if concentrations now match boundary values
3. Validate improved NSE metrics
4. If successful, proceed with METHOD 18+ for other species
