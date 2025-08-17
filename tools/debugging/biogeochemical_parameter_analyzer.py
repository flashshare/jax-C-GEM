#!/usr/bin/env python
"""
Biogeochemical Parameter Analysis and Calibration Tool

This script analyzes current biogeochemical parameters, compares them with
literature ranges, and identifies parameters that need calibration to improve
model performance for NH4, NO3, PO4, and other biogeochemical species.

Based on Phase I Task 1.2: Biogeochemical Parameter Calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.config_parser import parse_model_config
    from core.model_config import DEFAULT_BIO_PARAMS
    from core.biogeochemistry import create_biogeo_params
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

class BiogeochemicalParameterAnalyzer:
    """Comprehensive biogeochemical parameter analysis and calibration tool."""
    
    def __init__(self):
        self.current_params = {}
        self.literature_ranges = {}
        self.performance_targets = {}
        self.load_current_parameters()
        self.define_literature_ranges()
        self.define_performance_targets()
        
    def load_current_parameters(self):
        """Load current model parameters."""
        try:
            config = parse_model_config("config/model_config.txt")
            self.current_params = create_biogeo_params(config)
            print(f"✅ Loaded {len(self.current_params)} biogeochemical parameters")
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            self.current_params = DEFAULT_BIO_PARAMS.copy()
    
    def define_literature_ranges(self):
        """Define literature ranges for biogeochemical parameters."""
        self.literature_ranges = {
            # Phytoplankton growth parameters (from Volta et al. 2016, Cloern et al. 2014)
            'mumax_dia': (0.8, 3.5),        # day⁻¹ - diatom maximum growth rate
            'mumax_ndia': (0.5, 2.8),       # day⁻¹ - non-diatom maximum growth rate
            'alpha_light': (15.0, 50.0),     # μE⁻¹ m² s day⁻¹ - photosynthetic efficiency
            
            # Respiration and mortality (Vollenweider 1976, Jørgensen 1979)  
            'resp_dia': (0.02, 0.15),       # day⁻¹ - diatom respiration rate
            'resp_ndia': (0.02, 0.15),      # day⁻¹ - non-diatom respiration rate
            'mort_dia': (0.01, 0.10),       # day⁻¹ - diatom mortality rate
            'mort_ndia': (0.01, 0.10),      # day⁻¹ - non-diatom mortality rate
            
            # Nutrient cycling (Seitzinger 1988, Kemp et al. 2005)
            'nitrif_rate': (0.05, 0.30),   # day⁻¹ - nitrification rate
            'denitrif_rate': (0.01, 0.10), # day⁻¹ - denitrification rate  
            'degrad_rate': (0.05, 0.25),   # day⁻¹ - organic matter degradation
            
            # Half-saturation constants (Kmp values) (Cloern 1996)
            'ks_no3': (0.5, 5.0),          # mmol N/m³ - NO3 half-saturation
            'ks_nh4': (0.5, 3.0),          # mmol N/m³ - NH4 half-saturation  
            'ks_po4': (0.05, 0.5),         # mmol P/m³ - PO4 half-saturation
            'ks_si': (1.0, 10.0),          # mmol Si/m³ - Si half-saturation
            
            # Temperature coefficients (Q10 values) (Eppley 1972)
            'q10_phyto': (1.8, 2.8),       # Q10 for phytoplankton growth
            'q10_resp': (1.5, 2.5),        # Q10 for respiration
            'q10_nitrif': (2.0, 3.0),      # Q10 for nitrification
            'q10_degrad': (1.8, 2.5),      # Q10 for degradation
            
            # Stoichiometric ratios (Redfield et al. 1963, updated)
            'n_to_c': (0.12, 0.20),        # N:C ratio in phytoplankton
            'p_to_c': (0.008, 0.015),      # P:C ratio in phytoplankton
            'si_to_c': (0.10, 0.25),       # Si:C ratio in diatoms
            
            # Oxygen stoichiometry (Takahashi et al. 1985)
            'o2_to_c_photo': (1.2, 1.4),   # O2:C ratio in photosynthesis
            'o2_to_c_resp': (1.2, 1.4),    # O2:C ratio in respiration
            'o2_to_n_nitrif': (1.8, 2.2),  # O2:N ratio in nitrification
        }
        
        print(f"✅ Defined literature ranges for {len(self.literature_ranges)} parameters")
    
    def define_performance_targets(self):
        """Define performance targets for different biogeochemical species."""
        self.performance_targets = {
            'NH4': {'target_r2': 0.7, 'current_r2': 0.030, 'priority': 'high'},
            'NO3': {'target_r2': 0.7, 'current_r2': None, 'priority': 'high'},  
            'PO4': {'target_r2': 0.7, 'current_r2': 0.298, 'priority': 'high'},
            'TOC': {'target_r2': 0.5, 'current_r2': 0.000, 'priority': 'medium'},
            'O2': {'target_r2': 0.7, 'current_r2': 0.569, 'priority': 'low'},  # Already good
            'PHY1': {'target_r2': 0.5, 'current_r2': None, 'priority': 'medium'},
            'PHY2': {'target_r2': 0.5, 'current_r2': None, 'priority': 'medium'},
        }
    
    def analyze_parameter_compliance(self) -> pd.DataFrame:
        """Analyze current parameter compliance with literature ranges."""
        
        print("\n" + "="*60)
        print("📚 PARAMETER COMPLIANCE ANALYSIS")
        print("="*60)
        
        compliance_data = []
        
        for param, (min_lit, max_lit) in self.literature_ranges.items():
            if param in self.current_params:
                current_val = self.current_params[param]
                
                # Check compliance
                is_compliant = min_lit <= current_val <= max_lit
                
                # Calculate deviation if non-compliant
                if current_val < min_lit:
                    deviation = ((min_lit - current_val) / min_lit) * 100
                    deviation_type = "too_low"
                elif current_val > max_lit:
                    deviation = ((current_val - max_lit) / max_lit) * 100
                    deviation_type = "too_high"
                else:
                    deviation = 0.0
                    deviation_type = "compliant"
                
                compliance_data.append({
                    'parameter': param,
                    'current_value': current_val,
                    'literature_min': min_lit,
                    'literature_max': max_lit,
                    'compliant': is_compliant,
                    'deviation_percent': deviation,
                    'deviation_type': deviation_type
                })
                
                # Print status
                status = "✅" if is_compliant else "❌"
                print(f"{status} {param:15s}: {current_val:8.3f} (lit: {min_lit:6.2f}-{max_lit:6.2f})")
                
            else:
                compliance_data.append({
                    'parameter': param,
                    'current_value': None,
                    'literature_min': min_lit,
                    'literature_max': max_lit,
                    'compliant': False,
                    'deviation_percent': None,
                    'deviation_type': "missing"
                })
                print(f"⚠️  {param:15s}: MISSING from current parameters")
        
        df = pd.DataFrame(compliance_data)
        
        # Summary statistics
        total_params = len(df)
        compliant_params = df['compliant'].sum()
        missing_params = (df['current_value'].isna()).sum()
        
        print(f"\n📊 COMPLIANCE SUMMARY:")
        print(f"   Total parameters analyzed: {total_params}")
        print(f"   Compliant with literature: {compliant_params} ({compliant_params/total_params*100:.1f}%)")
        print(f"   Missing from model: {missing_params} ({missing_params/total_params*100:.1f}%)")
        print(f"   Non-compliant: {total_params - compliant_params - missing_params}")
        
        return df
    
    def identify_calibration_priorities(self, compliance_df: pd.DataFrame) -> List[Dict]:
        """Identify parameters that need urgent calibration based on performance impact."""
        
        print("\n" + "="*60)
        print("🎯 CALIBRATION PRIORITY ANALYSIS")
        print("="*60)
        
        # Priority scoring system
        priority_list = []
        
        # High priority: Parameters affecting poor-performing species (NH4, PO4, TOC)
        high_impact_params = [
            'mumax_dia', 'mumax_ndia',      # Affects all nutrient uptake
            'nitrif_rate',                   # Critical for NH4 -> NO3 conversion
            'resp_dia', 'resp_ndia',        # Affects nutrient release
            'ks_nh4', 'ks_no3', 'ks_po4',  # Half-saturation constants
            'degrad_rate',                   # Affects TOC decay and nutrient release
        ]
        
        # Medium priority: Parameters affecting system dynamics
        medium_impact_params = [
            'mort_dia', 'mort_ndia',        # Mortality -> TOC production
            'denitrif_rate',                # Alternative NO3 pathway
            'q10_phyto', 'q10_resp', 'q10_nitrif'  # Temperature dependencies
        ]
        
        # Process each parameter
        for _, row in compliance_df.iterrows():
            param = row['parameter']
            
            # Determine impact level
            if param in high_impact_params:
                impact_level = "high"
                base_priority = 100
            elif param in medium_impact_params:
                impact_level = "medium"
                base_priority = 50
            else:
                impact_level = "low"
                base_priority = 20
            
            # Adjust priority based on compliance
            if row['deviation_type'] == "missing":
                priority_score = base_priority + 50  # Missing parameters get high priority
            elif not row['compliant']:
                priority_score = base_priority + min(row['deviation_percent'], 50)
            else:
                priority_score = base_priority * 0.1  # Compliant parameters get low priority
            
            priority_list.append({
                'parameter': param,
                'priority_score': priority_score,
                'impact_level': impact_level,
                'compliance_status': row['deviation_type'],
                'current_value': row['current_value'],
                'suggested_min': row['literature_min'],
                'suggested_max': row['literature_max'],
                'suggested_value': (row['literature_min'] + row['literature_max']) / 2
            })
        
        # Sort by priority score
        priority_list.sort(key=lambda x: x['priority_score'], reverse=True)
        
        print(f"🔥 TOP 10 CALIBRATION PRIORITIES:")
        print(f"{'Rank':<4} {'Parameter':<15} {'Priority':<8} {'Impact':<8} {'Status':<12} {'Suggested':<10}")
        print("-" * 70)
        
        for i, item in enumerate(priority_list[:10], 1):
            print(f"{i:<4} {item['parameter']:<15} {item['priority_score']:<8.1f} "
                  f"{item['impact_level']:<8} {item['compliance_status']:<12} "
                  f"{item['suggested_value']:<10.3f}")
        
        return priority_list
    
    def suggest_calibration_plan(self, priority_list: List[Dict]) -> Dict:
        """Generate a systematic calibration plan."""
        
        print("\n" + "="*60)
        print("📋 SYSTEMATIC CALIBRATION PLAN")
        print("="*60)
        
        # Group by impact level
        high_priority = [p for p in priority_list if p['impact_level'] == 'high']
        medium_priority = [p for p in priority_list if p['impact_level'] == 'medium']
        
        calibration_plan = {
            'phase_1_critical': high_priority[:5],  # Top 5 critical parameters
            'phase_2_important': high_priority[5:] + medium_priority[:5],  # Next 5-10 parameters
            'phase_3_fine_tuning': medium_priority[5:],  # Remaining parameters
        }
        
        print(f"📅 PHASE 1 - CRITICAL PARAMETERS ({len(calibration_plan['phase_1_critical'])} params):")
        print(f"   Focus: NH4, NO3, PO4 system core functionality")
        for param_info in calibration_plan['phase_1_critical']:
            current_str = f"{param_info['current_value']:.3f}" if param_info['current_value'] is not None else "MISSING"
            print(f"   • {param_info['parameter']:<15}: {current_str} → {param_info['suggested_value']:.3f}")
        
        print(f"\n📅 PHASE 2 - IMPORTANT PARAMETERS ({len(calibration_plan['phase_2_important'])} params):")
        print(f"   Focus: TOC dynamics and temperature dependencies")
        for param_info in calibration_plan['phase_2_important']:
            current_str = f"{param_info['current_value']:.3f}" if param_info['current_value'] is not None else "MISSING"
            print(f"   • {param_info['parameter']:<15}: {current_str} → {param_info['suggested_value']:.3f}")
            
        print(f"\n📅 PHASE 3 - FINE-TUNING ({len(calibration_plan['phase_3_fine_tuning'])} params):")
        print(f"   Focus: Model optimization and sensitivity refinement")
        
        return calibration_plan
    
    def generate_calibration_config(self, calibration_plan: Dict) -> str:
        """Generate updated model_config.txt with calibrated parameters."""
        
        print("\n" + "="*60)
        print("⚙️ GENERATING CALIBRATED CONFIGURATION")
        print("="*60)
        
        # Load current config
        try:
            with open("config/model_config.txt", 'r') as f:
                config_lines = f.readlines()
        except FileNotFoundError:
            print("❌ model_config.txt not found")
            return ""
        
        # Create backup
        backup_path = "config/model_config_backup.txt"
        with open(backup_path, 'w') as f:
            f.writelines(config_lines)
        print(f"💾 Backed up original config to {backup_path}")
        
        # Apply Phase 1 calibrations
        updated_params = {}
        for param_info in calibration_plan['phase_1_critical']:
            param_name = param_info['parameter']
            suggested_value = param_info['suggested_value']
            updated_params[param_name] = suggested_value
            print(f"🔧 {param_name:<15}: → {suggested_value:.6f}")
        
        # Update config content
        new_config_lines = []
        section_biogeochemistry = False
        
        for line in config_lines:
            line_stripped = line.strip()
            
            # Track biogeochemistry section
            if line_stripped == "[biogeochemistry]":
                section_biogeochemistry = True
                new_config_lines.append(line)
                continue
            elif line_stripped.startswith("[") and line_stripped != "[biogeochemistry]":
                section_biogeochemistry = False
                new_config_lines.append(line)
                continue
            
            # Update parameters in biogeochemistry section
            if section_biogeochemistry and "=" in line:
                param_name = line.split("=")[0].strip()
                if param_name in updated_params:
                    new_line = f"{param_name} = {updated_params[param_name]:.6f}\n"
                    new_config_lines.append(new_line)
                    print(f"   Updated: {param_name}")
                else:
                    new_config_lines.append(line)
            else:
                new_config_lines.append(line)
        
        # Add missing parameters to biogeochemistry section
        missing_params = []
        for param_info in calibration_plan['phase_1_critical']:
            param_name = param_info['parameter']
            if param_info['current_value'] is None:  # Missing parameter
                missing_params.append((param_name, param_info['suggested_value']))
        
        if missing_params:
            print(f"\n➕ Adding {len(missing_params)} missing parameters:")
            # Find end of biogeochemistry section and insert
            for i, line in enumerate(new_config_lines):
                if line.strip() == "[biogeochemistry]":
                    # Find next section or end of file
                    insert_pos = len(new_config_lines)
                    for j in range(i + 1, len(new_config_lines)):
                        if new_config_lines[j].strip().startswith("["):
                            insert_pos = j
                            break
                    
                    # Insert missing parameters
                    for param_name, param_value in missing_params:
                        new_line = f"{param_name} = {param_value:.6f}\n"
                        new_config_lines.insert(insert_pos, new_line)
                        print(f"   Added: {param_name} = {param_value:.6f}")
                        insert_pos += 1
                    break
        
        # Write calibrated config
        calibrated_config_path = "config/model_config_calibrated.txt"
        with open(calibrated_config_path, 'w') as f:
            f.writelines(new_config_lines)
        
        print(f"✅ Generated calibrated config: {calibrated_config_path}")
        return calibrated_config_path
    
    def create_calibration_report(self, compliance_df: pd.DataFrame, 
                                 priority_list: List[Dict], 
                                 calibration_plan: Dict) -> str:
        """Create comprehensive calibration analysis report."""
        
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_summary': {
                'total_parameters': len(compliance_df),
                'compliant_parameters': int(compliance_df['compliant'].sum()),
                'missing_parameters': int(compliance_df['current_value'].isna().sum()),
                'calibration_priorities': len(calibration_plan['phase_1_critical'])
            },
            'compliance_analysis': compliance_df.to_dict('records'),
            'priority_ranking': priority_list,
            'calibration_plan': calibration_plan,
            'performance_targets': self.performance_targets
        }
        
        # Save detailed report
        report_path = "OUT/biogeochemical_calibration_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📊 Comprehensive calibration report saved: {report_path}")
        return report_path

def main():
    """Main calibration analysis workflow."""
    
    print("🔬 Biogeochemical Parameter Analysis & Calibration")
    print("="*60)
    print("Phase I Task 1.2: Systematic parameter optimization")
    print("Target: Improve NH4, NO3, PO4 model performance")
    
    # Create output directory
    Path("OUT").mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = BiogeochemicalParameterAnalyzer()
    
    # Run analysis workflow
    print("\n🔍 Step 1: Analyzing parameter compliance with literature...")
    compliance_df = analyzer.analyze_parameter_compliance()
    
    print("\n🎯 Step 2: Identifying calibration priorities...")
    priority_list = analyzer.identify_calibration_priorities(compliance_df)
    
    print("\n📋 Step 3: Creating systematic calibration plan...")
    calibration_plan = analyzer.suggest_calibration_plan(priority_list)
    
    print("\n⚙️ Step 4: Generating calibrated configuration...")
    calibrated_config_path = analyzer.generate_calibration_config(calibration_plan)
    
    print("\n📊 Step 5: Creating comprehensive report...")
    report_path = analyzer.create_calibration_report(
        compliance_df, priority_list, calibration_plan
    )
    
    print("\n" + "="*60)
    print("✅ CALIBRATION ANALYSIS COMPLETE")
    print("="*60)
    print(f"📄 Analysis report: {report_path}")
    print(f"⚙️ Calibrated config: {calibrated_config_path}")
    print("\nNext steps:")
    print("1. Review the calibrated parameters")
    print("2. Test with: cp config/model_config_calibrated.txt config/model_config.txt")
    print("3. Run simulation to validate improvements")
    print("4. Iterate based on results")

if __name__ == "__main__":
    main()