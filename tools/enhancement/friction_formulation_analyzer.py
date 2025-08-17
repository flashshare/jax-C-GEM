#!/usr/bin/env python
"""
Phase II Enhancement: Depth-Dependent Friction Implementation

This module implements depth-dependent friction formulation to address the systematic
tidal over-prediction. Current uniform Chezy coefficients don't account for varying
channel depths and may be causing unrealistic tidal amplification.

Physical Basis:
- Chezy coefficient should vary with depth: C = (h/n)^(1/6) where n is Manning's roughness
- Shallow areas should have higher friction (lower Chezy)
- Deep areas should have lower friction (higher Chezy)

Scientific Reference: 
- Henderson (1966), Open Channel Flow
- Savenije (2012), Salinity and Tides in Alluvial Estuaries
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.config_parser import parse_model_config
    from core.hydrodynamics import initialize_geometry
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

class DepthDependentFrictionAnalyzer:
    """Analyze and implement depth-dependent friction formulation."""
    
    def __init__(self):
        # Simple config parsing
        self.model_config = self._parse_config_manual()
        self.current_friction = self._extract_current_friction()
        self.geometry_data = self._load_geometry()
    
    def _parse_config_manual(self):
        """Manually parse configuration file."""
        config = {}
        with open("config/model_config.txt", 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()
                    try:
                        # Try to convert to float
                        config[key] = float(value)
                    except ValueError:
                        # Keep as string
                        config[key] = value.strip('"\'')
        return config
        
    def _extract_current_friction(self):
        """Extract current uniform friction coefficients."""
        return {
            'Chezy1': self.model_config.get('Chezy1', 25.0),
            'Chezy2': self.model_config.get('Chezy2', 35.0),
            'index_2': self.model_config.get('index_2', 31)
        }
    
    def _load_geometry(self):
        """Load channel geometry data with manual depth profile."""
        try:
            # Use manual depth profile similar to original C-GEM
            M = int(self.model_config.get('EL', 202000) / self.model_config.get('DELXI', 2000)) + 1
            
            # Simple depth profile approximation
            depths = np.zeros(M)
            widths = np.zeros(M)
            
            # Linear depth profile (approximate)
            depth_mouth = 12.0  # Mouth depth
            depth_head = 8.0    # Head depth
            depths = np.linspace(depth_mouth, depth_head, M)
            
            # Width profile from config
            B1 = self.model_config.get('B1', 3887.0)
            B2 = self.model_config.get('B2', 850.0)
            LC1 = self.model_config.get('LC1', 50000.0)
            
            # Exponential width convergence
            distances = np.arange(M) * self.model_config.get('DELXI', 2000.0)
            index_2 = int(self.model_config.get('index_2', 31))
            
            widths[:index_2] = B1 * np.exp(-distances[:index_2] / LC1)
            widths[index_2:] = B2
            
            return {'B': widths, 'PROF': depths, 'M': M}
            
        except Exception as e:
            print(f"Error creating geometry: {e}")
            # Return default geometry
            M = 102
            return {
                'B': np.linspace(3887, 850, M),
                'PROF': np.linspace(12, 8, M),
                'M': M
            }
    
    def analyze_current_friction_distribution(self):
        """Analyze current uniform friction distribution."""
        
        print("🔧 Current Friction Distribution Analysis")
        print("=" * 50)
        
        friction = self.current_friction
        geometry = self.geometry_data
        
        # Current uniform distribution
        M = geometry['M']
        index_2 = int(friction['index_2'])
        
        current_chezy = np.full(M, friction['Chezy1'])
        current_chezy[index_2:] = friction['Chezy2']
        
        depths = np.array(geometry['PROF'])
        widths = np.array(geometry['B'])
        
        print(f"Current Friction Configuration:")
        print(f"  Segment 1 (0-{index_2}): Chezy = {friction['Chezy1']:.1f} m^0.5/s")
        print(f"  Segment 2 ({index_2}-{M-1}): Chezy = {friction['Chezy2']:.1f} m^0.5/s")
        
        print(f"\nChannel Characteristics:")
        print(f"  Depth range: {depths.min():.1f} - {depths.max():.1f} m")
        print(f"  Width range: {widths.min():.0f} - {widths.max():.0f} m")
        print(f"  Depth variation: {(depths.max()-depths.min())/depths.mean()*100:.1f}%")
        
        # Identify potential issues
        shallow_indices = depths < 8.0
        deep_indices = depths > 15.0
        
        if np.any(shallow_indices) and np.any(deep_indices):
            print(f"\n⚠️  FRICTION UNIFORMITY ISSUES:")
            print(f"  Shallow areas (<8m): {np.sum(shallow_indices)} cells")
            print(f"  Deep areas (>15m): {np.sum(deep_indices)} cells")
            print(f"  Problem: Uniform friction ignores depth-dependent flow resistance")
        
        return current_chezy, depths, widths
    
    def compute_depth_dependent_friction(self, manning_n_base=0.04):
        """Compute depth-dependent Chezy coefficients."""
        
        print(f"\n🌊 Computing Depth-Dependent Friction")
        print("-" * 40)
        
        geometry = self.geometry_data
        depths = np.array(geometry['PROF'])
        
        # Method 1: Manning-based depth dependency
        # C = (h^(1/6)) / n, where n varies with depth
        # Shallow areas: higher roughness (more vegetation, bed effects)
        # Deep areas: lower roughness (smoother flow)
        
        # Depth-dependent Manning's n
        n_shallow = manning_n_base * 1.5  # Higher roughness in shallow areas
        n_deep = manning_n_base * 0.8     # Lower roughness in deep areas
        
        # Smooth transition between shallow and deep
        depth_transition = 12.0  # Transition depth [m]
        transition_width = 4.0   # Smoothing width [m]
        
        # Sigmoid transition function
        transition_factor = 1 / (1 + np.exp(-(depths - depth_transition) / transition_width))
        manning_n = n_shallow + (n_deep - n_shallow) * transition_factor
        
        # Compute depth-dependent Chezy coefficients
        chezy_depth_dependent = (depths ** (1/6)) / manning_n
        
        # Method 2: Empirical depth correction
        # Based on estuarine friction studies (e.g., Prandle & Rahman, 1980)
        chezy_base_1 = self.current_friction['Chezy1']  # Reference for segment 1
        chezy_base_2 = self.current_friction['Chezy2']  # Reference for segment 2
        
        # Apply base values with depth corrections
        index_2 = int(self.current_friction['index_2'])
        chezy_empirical = np.full(len(depths), chezy_base_1)
        chezy_empirical[index_2:] = chezy_base_2
        
        # Depth correction factor: C_corrected = C_base * (h/h_ref)^alpha
        h_ref = 10.0  # Reference depth [m]
        alpha = 0.15  # Depth correction exponent (calibration parameter)
        
        depth_correction = (depths / h_ref) ** alpha
        chezy_empirical *= depth_correction
        
        # Ensure reasonable bounds
        chezy_depth_dependent = np.clip(chezy_depth_dependent, 15.0, 50.0)
        chezy_empirical = np.clip(chezy_empirical, 15.0, 50.0)
        
        results = {
            'manning_based': chezy_depth_dependent,
            'empirical': chezy_empirical,
            'manning_n': manning_n,
            'depths': depths,
            'depth_correction': depth_correction
        }
        
        print(f"Depth-Dependent Friction Computed:")
        print(f"  Manning-based range: {chezy_depth_dependent.min():.1f} - {chezy_depth_dependent.max():.1f} m^0.5/s")
        print(f"  Empirical range: {chezy_empirical.min():.1f} - {chezy_empirical.max():.1f} m^0.5/s")
        print(f"  Manning's n range: {manning_n.min():.3f} - {manning_n.max():.3f}")
        
        return results
    
    def visualize_friction_comparison(self, depth_dependent_results):
        """Visualize current vs depth-dependent friction."""
        
        print(f"\n📊 Creating Friction Comparison Visualization")
        
        if depth_dependent_results is None:
            return
        
        current_chezy, depths, widths = self.analyze_current_friction_distribution()
        
        distances = np.arange(len(depths)) * 2.0  # 2km grid spacing
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Depth profile
        axes[0].plot(distances, depths, 'b-', linewidth=2, label='Channel Depth')
        axes[0].set_ylabel('Depth (m)')
        axes[0].set_title('Channel Geometry and Friction Distribution')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Friction coefficients comparison
        axes[1].plot(distances, current_chezy, 'r-', linewidth=2, label='Current Uniform')
        axes[1].plot(distances, depth_dependent_results['manning_based'], 'g-', linewidth=2, label='Manning-based')
        axes[1].plot(distances, depth_dependent_results['empirical'], 'b--', linewidth=2, label='Empirical')
        axes[1].set_ylabel('Chezy Coefficient (m^0.5/s)')
        axes[1].set_title('Friction Coefficient Comparison')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Manning's n distribution
        axes[2].plot(distances, depth_dependent_results['manning_n'], 'purple', linewidth=2, label='Depth-dependent n')
        axes[2].axhline(y=0.04, color='orange', linestyle='--', label='Typical estuarine n=0.04')
        axes[2].set_xlabel('Distance from mouth (km)')
        axes[2].set_ylabel("Manning's n")
        axes[2].set_title("Manning's Roughness Distribution")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        output_file = "OUT/depth_dependent_friction_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_file}")
        
        return output_file
    
    def generate_enhanced_config(self, method='empirical'):
        """Generate model configuration with depth-dependent friction."""
        
        print(f"\n🔧 Generating Enhanced Configuration")
        print("-" * 40)
        
        depth_dependent_results = self.compute_depth_dependent_friction()
        if depth_dependent_results is None:
            return None
        
        # Select friction method
        if method == 'manning_based':
            new_friction = depth_dependent_results['manning_based']
        else:
            new_friction = depth_dependent_results['empirical']
        
        # For implementation, we need to modify the model to accept spatial arrays
        # For now, let's create segment-based approximations
        
        index_2_str = self.current_friction.get('index_2', '31')
        try:
            index_2 = int(index_2_str)
        except (ValueError, TypeError):
            index_2 = 31  # Default fallback
        
        # Segment averages
        chezy1_new = np.mean(new_friction[:index_2])
        chezy2_new = np.mean(new_friction[index_2:])
        
        print(f"Enhanced Friction Configuration ({method} method):")
        print(f"  Segment 1 average: {self.current_friction['Chezy1']:.1f} → {chezy1_new:.1f} m^0.5/s ({((chezy1_new-self.current_friction['Chezy1'])/self.current_friction['Chezy1']*100):+.1f}%)")
        print(f"  Segment 2 average: {self.current_friction['Chezy2']:.1f} → {chezy2_new:.1f} m^0.5/s ({((chezy2_new-self.current_friction['Chezy2'])/self.current_friction['Chezy2']*100):+.1f}%)")
        
        # Create enhanced configuration
        enhanced_config = {
            'Chezy1': round(chezy1_new, 1),
            'Chezy2': round(chezy2_new, 1),
            'method': method,
            'spatial_friction': new_friction  # For future spatial implementation
        }
        
        return enhanced_config

def main():
    """Run depth-dependent friction analysis and enhancement."""
    
    print("🔧 JAX C-GEM Phase II Enhancement: Depth-Dependent Friction")
    print("=" * 65)
    
    analyzer = DepthDependentFrictionAnalyzer()
    
    # Step 1: Analyze current friction issues
    current_chezy, depths, widths = analyzer.analyze_current_friction_distribution()
    
    # Step 2: Compute depth-dependent friction
    depth_results = analyzer.compute_depth_dependent_friction()
    
    # Step 3: Visualize comparison
    if depth_results:
        viz_file = analyzer.visualize_friction_comparison(depth_results)
    
    # Step 4: Generate enhanced configurations
    empirical_config = analyzer.generate_enhanced_config('empirical')
    manning_config = analyzer.generate_enhanced_config('manning_based')
    
    print(f"\n🎯 ENHANCEMENT SUMMARY:")
    print("=" * 25)
    
    if empirical_config and manning_config:
        print("Two enhanced friction formulations generated:")
        print(f"  1. Empirical method: Chezy1={empirical_config['Chezy1']}, Chezy2={empirical_config['Chezy2']}")
        print(f"  2. Manning-based method: Chezy1={manning_config['Chezy1']}, Chezy2={manning_config['Chezy2']}")
        
        print(f"\n🔬 EXPECTED IMPACT:")
        print("- More realistic friction representation")
        print("- Reduced tidal over-amplification in shallow areas")
        print("- Improved momentum balance accuracy")
        print("- Better match with field observations")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Test empirical configuration first (more conservative)")
        print("2. Run simulation and validate against tidal observations")
        print("3. Compare results with uniform friction baseline")
        print("4. If successful, implement full spatial friction arrays")
    
    return empirical_config, manning_config

if __name__ == "__main__":
    main()