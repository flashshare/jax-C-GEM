#!/usr/bin/env python
"""
Conservative Tidal Friction Adjustment

Test a single conservative friction adjustment to avoid numerical instability
while still working toward reducing the 2x tidal range over-prediction.
"""
import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def create_conservative_config():
    """Create configuration with conservative friction adjustments."""
    
    base_config = "config/model_config.txt"
    
    # Read current config
    with open(base_config, 'r') as f:
        config_content = f.read()
    
    # Make conservative adjustments:
    # Current: Chezy1=25, Chezy2=35 (too low friction)
    # Target: Reduce by ~20% to increase friction
    config_content = config_content.replace(
        "Chezy1 = 25.0                # Chezy coefficient segment 1",
        "Chezy1 = 20.0                # Chezy coefficient segment 1 - CONSERVATIVE FRICTION INCREASE"
    )
    config_content = config_content.replace(
        "Chezy2 = 35.0                # Chezy coefficient segment 2",
        "Chezy2 = 28.0                # Chezy coefficient segment 2 - CONSERVATIVE FRICTION INCREASE"  
    )
    
    # Save conservative config
    conservative_config = "config/model_config_conservative_friction.txt"
    
    with open(conservative_config, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Conservative configuration created: {conservative_config}")
    print("   Chezy1: 25.0 → 20.0 (20% reduction)")
    print("   Chezy2: 35.0 → 28.0 (20% reduction)")
    
    return conservative_config

def main():
    """Create and test conservative friction adjustment."""
    
    print("🌊 Conservative Tidal Friction Adjustment")
    print("=" * 45)
    
    # Create conservative config
    config_file = create_conservative_config()
    
    print("\n🎯 Next Steps:")
    print("1. Manually run model with conservative config:")
    print(f"   python src/main.py --mode run --output-format auto")
    print("2. Validate results with Phase 2:")
    print("   python tools/verification/phase2_tidal_dynamics.py")
    print("3. If successful, can adjust further if needed")
    
    print(f"\n📊 Expected Results:")
    print("   - BD: 6.20m → ~5.0m (target: 2.92m)")
    print("   - BK: 5.60m → ~4.5m (target: 3.22m)")
    print("   - PC: 6.63m → ~5.3m (target: 2.07m)")
    print("   - Still over-predicted but moving in right direction")

if __name__ == "__main__":
    main()