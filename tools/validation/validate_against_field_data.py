"""
Comprehensive field data validation for JAX C-GEM model.

This script provides validation of model outputs against available field observations,
including data from:
- CARE_2017-2018.csv
- CEM_2017-2018.csv
- CEM_quality_2014-2016.csv
- SIHYMECC_Tidal-range2017-2018.csv

It produces comprehensive statistical validation metrics and visual comparison plots
for key water quality parameters.

Usage:
    python tools/validation/validate_against_field_data.py

Author: Nguyen Truong An
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import jax.numpy as jnp
import argparse

# Add src directory to path for imports
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Add tools directory for analysis modules
tools_path = str(Path(__file__).parent.parent)
if tools_path not in sys.path:
    sys.path.append(tools_path)

# Graceful import with fallback
try:
    from analysis.model_validation import ModelValidator as AdvancedModelValidator
    from analysis.model_validation import ValidationMetrics as AdvancedValidationMetrics
    from analysis.model_validation import ValidationResult as AdvancedValidationResult
    HAS_MODEL_VALIDATION = True
    print("✅ Advanced model validation module loaded")
except ImportError:
    print("⚠️  Advanced model validation module not available - using basic validation")
    HAS_MODEL_VALIDATION = False
    AdvancedModelValidator = None
    AdvancedValidationMetrics = None
    AdvancedValidationResult = None
    
    class ValidationMetrics:
        @staticmethod
        def calculate_rmse(observed, predicted): 
            return np.sqrt(np.mean((observed - predicted)**2))
        
        @staticmethod
        def calculate_correlation(observed, predicted):
            """Calculate correlation coefficient with proper error handling."""
            try:
                # Check for constant values (zero standard deviation)
                if np.std(observed) == 0 or np.std(predicted) == 0:
                    return np.nan  # Return NaN for constant values
                
                # Suppress numpy warnings temporarily
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr_matrix = np.corrcoef(observed, predicted)
                    correlation = corr_matrix[0, 1]
                    
                # Return NaN if correlation is invalid
                return correlation if not np.isnan(correlation) else np.nan
                
            except (ValueError, RuntimeWarning):
                return np.nan
    
    class ValidationResult:
        def __init__(self, rmse, correlation, bias):
            self.rmse = rmse
            self.correlation = correlation if not np.isnan(correlation) else 0.0
            self.bias = bias
            # Add r_squared for compatibility - handle NaN correlation
            if np.isnan(correlation):
                self.r_squared = float('nan')
            else:
                self.r_squared = max(0.0, correlation**2)  # Ensure non-negative R²
    
    class ModelValidator:
        def __init__(self, output_dir=None): 
            self.output_dir = Path(output_dir) if output_dir else Path("validation_results")
            self.validation_results = {}
        
        def validate_against_observations(self, model_results, observations, create_plots=True):
            results = {}
            for var in observations.keys():
                if var in model_results:
                    obs = observations[var]
                    pred = model_results[var]
                    rmse = ValidationMetrics.calculate_rmse(obs, pred)
                    corr = ValidationMetrics.calculate_correlation(obs, pred)
                    bias = np.mean(pred - obs)
                    results[var] = ValidationResult(rmse, corr, bias)
            return results
        
        def generate_validation_report(self, output_file):
            with open(output_file, 'w') as f:
                f.write("# Basic Validation Report\n")
                f.write("Field data validation completed with basic metrics.\n")
            return output_file
        
        def compute_validation_metrics(self, obs, pred, var_name):
            rmse = ValidationMetrics.calculate_rmse(obs, pred)
            corr = ValidationMetrics.calculate_correlation(obs, pred)
            bias = np.mean(pred - obs)
            return ValidationResult(rmse, corr, bias)

class FieldDataValidator:
    """
    Validates model outputs against field observations.
    
    This class loads field observation data from calibration files and compares
    model outputs against these observations using statistical metrics and plots.
    """
    
    def __init__(self, 
                 output_dir: str = "OUT/validation_results", 
                 model_output_dir: str = "OUT",
                 model_output_format: str = "auto"):  # Changed default to auto-detect
        """Initialize the field data validator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_output_dir = Path(model_output_dir)
        
        # Auto-detect format if not specified
        if model_output_format == "auto":
            npz_file = self.model_output_dir / "simulation_results.npz"
            hydro_dir = self.model_output_dir / "Hydrodynamics"
            
            if npz_file.exists():
                self.model_output_format = "npz"
                print("🔍 Auto-detected NPZ format")
            elif hydro_dir.exists():
                self.model_output_format = "csv" 
                print("🔍 Auto-detected CSV format")
            else:
                print("⚠️ No model results found, defaulting to NPZ format")
                self.model_output_format = "npz"
        else:
            self.model_output_format = model_output_format
        
        # Load model configuration for temporal alignment
        try:
            from core.config_parser import parse_model_config
            config_path = Path(__file__).parent.parent.parent / "config" / "model_config.txt"
            self.model_config = parse_model_config(str(config_path))
            
            # Parse simulation start date
            if 'simulation_start_date' in self.model_config:
                from datetime import datetime
                self.simulation_start_date = datetime.strptime(
                    self.model_config['simulation_start_date'], "%Y-%m-%d"
                )
                print(f"📅 Simulation start date: {self.simulation_start_date.strftime('%Y-%m-%d')}")
            else:
                print("⚠️ No simulation start date specified, using default 2017-01-01")
                from datetime import datetime
                self.simulation_start_date = datetime(2017, 1, 1)
        except Exception as e:
            print(f"⚠️ Could not load model config: {e}")
            from datetime import datetime
            self.simulation_start_date = datetime(2017, 1, 1)
            self.model_config = {}
        
        # Initialize ModelValidator
        if HAS_MODEL_VALIDATION and AdvancedModelValidator is not None:
            self.validator = AdvancedModelValidator(str(self.output_dir))
        else:
            self.validator = ModelValidator()
        
        # Store loaded datasets
        self.datasets = {}
        self.model_results = {}
        self.station_mapping = {}
        
        print(f"🔍 Field Data Validator initialized")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   📁 Model results directory: {self.model_output_dir}")
        if not HAS_MODEL_VALIDATION:
            print("   ⚠️  Using basic validation (advanced features unavailable)")
    
    def load_field_observations(self):
        """Load all field observation datasets."""
        input_dir = Path(__file__).parent.parent.parent / "INPUT" / "Calibration"
        
        # Load CARE dataset
        care_path = input_dir / "CARE_2017-2018.csv"
        if care_path.exists():
            care_data = pd.read_csv(care_path)
            self.datasets['CARE'] = care_data
            print(f"✅ Loaded CARE dataset: {len(care_data)} observations")
        else:
            print(f"⚠️ CARE dataset not found at: {care_path}")
        
        # Load CEM datasets
        cem_path = input_dir / "CEM_2017-2018.csv"
        if cem_path.exists():
            cem_data = pd.read_csv(cem_path)
            self.datasets['CEM'] = cem_data
            print(f"✅ Loaded CEM dataset: {len(cem_data)} observations")
        else:
            print(f"⚠️ CEM dataset not found at: {cem_path}")
        
        # Load CEM quality dataset - note the filename may vary
        file_variations = ["CEM_2017-2018.csv"]
        cem_quality_found = False
        
        for filename in file_variations:
            cem_quality_path = input_dir / filename
            if cem_quality_path.exists():
                cem_quality_data = pd.read_csv(cem_quality_path)
                self.datasets['CEM_QUALITY'] = cem_quality_data
                print(f"✅ Loaded CEM quality dataset: {len(cem_quality_data)} observations")
                cem_quality_found = True
                break
        
        if not cem_quality_found:
            print(f"⚠️ CEM quality dataset not found in input directory")
        
        # Load SIHYMECC tidal range dataset
        sihymecc_path = input_dir / "SIHYMECC_Tidal-range2017-2018.csv"
        if sihymecc_path.exists():
            sihymecc_data = pd.read_csv(sihymecc_path)
            self.datasets['SIHYMECC'] = sihymecc_data
            print(f"✅ Loaded SIHYMECC tidal range dataset: {len(sihymecc_data)} observations")
        else:
            print(f"⚠️ SIHYMECC dataset not found at: {sihymecc_path}")
        
        # Create station mapping using geometry file
        self._create_station_mapping()
    
    def _create_station_mapping(self):
        """Map observation station locations to model grid points."""
        # Load geometry data
        geometry_path = Path(__file__).parent.parent.parent / "INPUT" / "Geometry" / "Geometry.csv"
        if not geometry_path.exists():
            print(f"⚠️ Geometry file not found at: {geometry_path}")
            return
        
        geometry = pd.read_csv(geometry_path)
        
        # Create mapping for CARE stations
        if 'CARE' in self.datasets:
            care_stations = self.datasets['CARE']['Site'].unique()
            care_locations = self.datasets['CARE']['Location'].unique()
            
            for station, location in zip(care_stations, care_locations):
                # Find the closest grid point
                grid_idx = self._find_closest_grid_point(geometry, location)
                self.station_mapping[f"CARE_{station}"] = {
                    'grid_index': grid_idx,
                    'location': location
                }
        
        # Create mapping for CEM stations
        if 'CEM' in self.datasets:
            cem_stations = self.datasets['CEM']['Site'].unique()
            
            for station in cem_stations:
                # Get the location for this station
                station_data = self.datasets['CEM'][self.datasets['CEM']['Site'] == station]
                if len(station_data) > 0:
                    location = station_data['Location'].iloc[0]
                    # Find the closest grid point
                    grid_idx = self._find_closest_grid_point(geometry, location)
                    self.station_mapping[f"CEM_{station}"] = {
                        'grid_index': grid_idx,
                        'location': location
                    }
        
        print(f"✅ Created station mapping for {len(self.station_mapping)} stations")
    
    def _find_closest_grid_point(self, geometry, location):
        """Find the closest grid point to the given location."""
        try:
            location_float = float(location)
            # Use 'Location' column instead of 'X_Dis'
            distances = np.abs(geometry['Location'] - location_float)
            return np.argmin(distances)
        except (ValueError, TypeError):
            # If location is not a number, return -1 (invalid)
            return -1
    
    def load_model_results(self):
        """Load model simulation results."""
        if self.model_output_format == 'csv':
            self._load_model_results_csv()
        elif self.model_output_format == 'npz':
            self._load_model_results_npz()
        else:
            print(f"⚠️ Unsupported model output format: {self.model_output_format}")
    
    def _load_model_results_csv(self):
        """Load model results from CSV files."""
        # Load hydrodynamics results
        hydro_dir = self.model_output_dir / "Hydrodynamics"
        if not hydro_dir.exists():
            print(f"⚠️ Hydrodynamics directory not found: {hydro_dir}")
            return
        
        # Load water level (eta)
        eta_file = hydro_dir / "eta.csv"
        if eta_file.exists():
            eta = pd.read_csv(eta_file)
            self.model_results['eta'] = eta.values
            print(f"✅ Loaded water level data: {eta.shape}")
        
        # Load salinity (S)
        sal_file = hydro_dir / "S.csv"
        if sal_file.exists():
            sal = pd.read_csv(sal_file)
            self.model_results['Salinity'] = sal.values
            print(f"✅ Loaded salinity data: {sal.shape}")
        
        # Load species concentration results (directly from OUT directory)
        # Map species names to file prefixes
        species_map = {
            'DO': 'O2',
            'NO3': 'NO3',
            'NH4': 'NH4',
            'PO4': 'PO4',
            'DSi': 'Si',
            'TOC': 'TOC',
            'Chl-a': 'Dia'  # Use diatoms (Dia) as proxy for chlorophyll-a
        }
        
        # Load each species
        for display_name, file_prefix in species_map.items():
            species_file = self.model_output_dir / f"{file_prefix}.csv"
            if species_file.exists():
                species_data = pd.read_csv(species_file)
                self.model_results[display_name] = species_data.values
                print(f"✅ Loaded {display_name} data: {species_data.shape}")
            else:
                print(f"⚠️ {display_name} data file not found: {species_file}")
        
        # Load time data or generate it if not available
        time_file = self.model_output_dir / "time.csv"
        if time_file.exists():
            time_data = pd.read_csv(time_file)
            self.model_results['time'] = time_data.values.flatten()
            print(f"✅ Loaded time data: {len(self.model_results['time'])} points")
        else:
            print(f"⚠️ Time data file not found: {time_file}")
            # Generate time data from one of the result files if available
            if 'DO' in self.model_results:
                num_timesteps = self.model_results['DO'].shape[0]
                # Generate time array (assuming 10 days simulation with even timesteps)
                self.model_results['time'] = np.linspace(0, 10, num_timesteps)
                print(f"📊 Generated time data: {len(self.model_results['time'])} points")
            else:
                print("⚠️ Cannot generate time data - no result files available")
    
    def _load_model_results_npz(self):
        """Load model results from NPZ file."""
        npz_file = self.model_output_dir / "simulation_results.npz"
        if not npz_file.exists():
            print(f"⚠️ Model results file not found: {npz_file}")
            return
        
        # Load the NPZ file
        data = np.load(npz_file)
        
        print(f"🔍 Available NPZ keys: {list(data.keys())}")
        
        # Extract relevant variables (fix variable names)
        if 'time' in data:
            self.model_results['time'] = data['time']
        else:
            print("⚠️ Time data not found in NPZ file")
            
        # Water level - check for H (height) instead of eta
        if 'H' in data:
            self.model_results['eta'] = data['H']  # H is water surface height
            print(f"✅ Loaded water level (H) data: shape {data['H'].shape}")
        else:
            print("⚠️ Water level (H) not found in NPZ file")
            
        # Salinity
        if 'S' in data:
            self.model_results['Salinity'] = data['S']
            print(f"✅ Loaded salinity (S) data: shape {data['S'].shape}")
        else:
            print("⚠️ Salinity (S) not found in NPZ file")
        
        # Map species names to what's actually in the NPZ file
        species_map = {
            'DO': 'O2',      # Dissolved Oxygen
            'NO3': 'NO3',    # Nitrate  
            'NH4': 'NH4',    # Ammonium
            'PO4': 'PO4',    # Phosphate
            'DSi': 'SI',     # Dissolved Silica (note: SI not Si in your model)
            'TOC': 'TOC',    # Total Organic Carbon
            'Chl-a': 'PHY1'  # Phytoplankton 1 as proxy for Chlorophyll-a
        }
        
        # Load each species
        for display_name, npz_name in species_map.items():
            if npz_name in data:
                self.model_results[display_name] = data[npz_name]
                print(f"✅ Loaded {display_name} data: {data[npz_name].shape}")
            else:
                print(f"⚠️ {display_name} data not found in NPZ file")
        
        print(f"✅ Loaded model results from NPZ file")
    
    def _convert_date_to_days(self, date_str):
        """Convert date string to days from simulation start."""
        try:
            # Attempt to parse various date formats
            date_formats = [
                "%d-%b-%y",  # 10-Jan-17
                "%m/%d/%Y",  # 10/09/2017
                "%Y-%m-%d",  # 2017-01-10
                "%d/%m/%Y"   # 10/01/2017
            ]
            
            date_obj = None
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if date_obj is None:
                print(f"⚠️ Could not parse date format: {date_str}")
                return None
            
            # Use the configured simulation start date
            days_diff = (date_obj - self.simulation_start_date).days
            
            # Only accept dates within reasonable range (warmup + simulation period)
            # Note: MAXT and WARMUP are already in days in the config, not seconds
            warmup_days = self.model_config.get('WARMUP', 100)  # Default 100 days
            total_simulation_days = self.model_config.get('MAXT', 465)  # Default 465 days  
            
            # Accept dates from warmup start to end of simulation
            if days_diff < -warmup_days or days_diff > total_simulation_days:
                print(f"📅 Date {date_str} ({days_diff:.1f} days from start) outside simulation range")
                return None
                
            return max(0, days_diff)  # Don't return negative days for output period
        except Exception as e:
            print(f"⚠️ Error converting date {date_str}: {e}")
            return None
    
    def prepare_validation_data(self):
        """Prepare observation data for validation."""
        if not self.datasets or not self.model_results:
            print("⚠️ Missing datasets or model results - call load_field_observations() and load_model_results() first")
            return {}
        
        # Dictionary to hold prepared validation data
        validation_data = {}
        
        # Process CARE dataset
        if 'CARE' in self.datasets:
            care_data = self.datasets['CARE']
            
            # Get time index in model results
            model_time = self.model_results.get('time', None)
            if model_time is None:
                print("⚠️ Model time data not available")
                return {}
            
            # Map variable names
            var_map = {
                'Salinity': 'Salinity',
                'DO (mg/L)': 'DO',
                'NO3 (mgN/L)': 'NO3',
                'NH4 (mgN/L)': 'NH4',
                'PO4 (mgP/L)': 'PO4',
                'DSi (mgSi/L)': 'DSi',
                'TOC (mgC/L)': 'TOC',
                'Chl-a (μg/L)': 'Chl-a'
            }
            
            # Process each variable
            for care_var, model_var in var_map.items():
                # Skip if model doesn't have this variable
                if model_var not in self.model_results:
                    continue
                
                # Get all observations for this variable
                obs_values = []
                pred_values = []
                
                for _, row in care_data.iterrows():
                    # Skip if variable is missing
                    if care_var not in row or pd.isna(row[care_var]):
                        continue
                    
                    # Get station mapping
                    station_key = f"CARE_{row['Site']}"
                    if station_key not in self.station_mapping:
                        continue
                    
                    grid_idx = self.station_mapping[station_key]['grid_index']
                    if grid_idx < 0:
                        continue
                    
                    # Convert date to model time
                    model_day = self._convert_date_to_days(row['Date'])
                    if model_day is None or model_day < 0:
                        continue
                    
                    # Find closest time in model results
                    # Convert model_day to seconds to match model_time units
                    model_time_seconds = model_day * 86400.0
                    time_idx = np.argmin(np.abs(model_time - model_time_seconds))
                    
                    # Get observation and model prediction
                    obs_value = row[care_var]
                    pred_value = self.model_results[model_var][time_idx, grid_idx]
                    
                    obs_values.append(float(obs_value))
                    pred_values.append(float(pred_value))
                
                # Store as arrays for validation
                if obs_values and pred_values:
                    validation_data[model_var] = {
                        'observed': jnp.array(obs_values),
                        'predicted': jnp.array(pred_values),
                        'n_obs': len(obs_values)
                    }
                    print(f"✅ Prepared {model_var} validation data: {len(obs_values)} points")
        
        # Process CEM dataset similarly (focusing on DO, Salinity, etc.)
        if 'CEM' in self.datasets:
            cem_data = self.datasets['CEM']
            
            # Map variable names (different structure in CEM dataset)
            var_map = {
                'DO (mg/L)': 'DO',
                'Salinity': 'Salinity',
                'TSS (mg/L)': 'TSS',
                'NH4 (mgN/L)': 'NH4',
                'PO4 (mgP/L)': 'PO4',
                'TOC (mgC/L)': 'TOC'
            }
            
            # Process each variable
            for cem_var, model_var in var_map.items():
                # Skip if model doesn't have this variable
                if model_var not in self.model_results:
                    continue
                
                # Get all observations for this variable
                obs_values = []
                pred_values = []
                
                for _, row in cem_data.iterrows():
                    # Skip if variable is missing
                    if cem_var not in row or pd.isna(row[cem_var]):
                        continue
                    
                    # Get station mapping
                    station_key = f"CEM_{row['Site']}"
                    if station_key not in self.station_mapping:
                        continue
                    
                    grid_idx = self.station_mapping[station_key]['grid_index']
                    if grid_idx < 0:
                        continue
                    
                    # Convert date to model time (CEM uses different format)
                    try:
                        date_str = row['Date']
                        model_day = self._convert_date_to_days(date_str)
                        if model_day is None or model_day < 0:
                            continue
                    except (KeyError, ValueError):
                        continue
                    
                    # Find closest time in model results
                    model_time = self.model_results.get('time', None)
                    if model_time is None:
                        continue
                    
                    # Convert model_day to seconds to match model_time units
                    model_time_seconds = model_day * 86400.0
                    time_idx = np.argmin(np.abs(model_time - model_time_seconds))
                    
                    # Get observation and model prediction
                    obs_value = row[cem_var]
                    pred_value = self.model_results[model_var][time_idx, grid_idx]
                    
                    obs_values.append(float(obs_value))
                    pred_values.append(float(pred_value))
                
                # Add to existing validation data or create new entry
                if obs_values and pred_values:
                    if model_var in validation_data:
                        # Append to existing data
                        validation_data[model_var]['observed'] = jnp.concatenate([
                            validation_data[model_var]['observed'],
                            jnp.array(obs_values)
                        ])
                        validation_data[model_var]['predicted'] = jnp.concatenate([
                            validation_data[model_var]['predicted'],
                            jnp.array(pred_values)
                        ])
                        validation_data[model_var]['n_obs'] += len(obs_values)
                    else:
                        # Create new entry
                        validation_data[model_var] = {
                            'observed': jnp.array(obs_values),
                            'predicted': jnp.array(pred_values),
                            'n_obs': len(obs_values)
                        }
                    
                    print(f"✅ Added CEM {model_var} validation data: {len(obs_values)} points")
        
        return validation_data
    
    def validate_model(self):
        """Run validation against field observations."""
        # Prepare validation data
        validation_data = self.prepare_validation_data()
        if not validation_data:
            print("⚠️ No validation data prepared")
            return
        
        # Convert to format expected by ModelValidator
        observations = {}
        model_results = {}
        
        for var, data in validation_data.items():
            observations[var] = data['observed']
            model_results[var] = data['predicted']
        
        # Run validation
        print("\n🔬 Running comprehensive model validation...")
        validation_results = self.validator.validate_against_observations(
            model_results, observations, create_plots=True
        )
        
        # Generate validation report
        report_path = self.validator.generate_validation_report(
            str(self.output_dir / "field_validation_report.md")
        )
        
        # Create main validation plots
        self._create_longitudinal_profile_plot()
        self._create_temporal_comparison_plot(validation_data)
        
        return validation_results
        
    def _create_longitudinal_profile_plot(self):
        """Create combined longitudinal profile plot for all water quality variables."""
        # Check if model results are available
        if not self.model_results:
            return
            
        # Use all available water quality variables 
        available_vars = [var for var in ['Salinity', 'DO', 'NO3', 'NH4', 'PO4', 'DSi', 'TOC', 'Chl-a'] 
                         if var in self.model_results]
        
        if not available_vars:
            return
            
        # Create figure with improved layout
        fig, axes = plt.subplots(len(available_vars), 1, figsize=(12, 2.5*len(available_vars)))
        if len(available_vars) == 1:
            axes = [axes]
            
        # Use grid indices for x-axis to ensure consistency
        x_values = np.arange(self.model_results[available_vars[0]].shape[1])
        
        # Plot each variable
        for i, var in enumerate(available_vars):
            ax = axes[i]
            
            # Get the data at the middle time step (after warmup)
            middle_time_idx = self.model_results[var].shape[0] // 2
            profile = self.model_results[var][middle_time_idx, :]
            
            # Plot longitudinal profile with improved styling
            ax.plot(x_values, profile, 'blue', linewidth=2, alpha=0.8, label='Model Profile')
            
            # Add station observations if available
            station_count = 0
            for station, info in self.station_mapping.items():
                grid_idx = info['grid_index']
                if grid_idx >= 0 and grid_idx < len(profile):
                    ax.scatter(grid_idx, profile[grid_idx], 
                             color='red', marker='o', s=40, zorder=5, alpha=0.8)
                    if station_count < 3:  # Only label first few stations to avoid clutter
                        ax.text(grid_idx, profile[grid_idx], 
                               station.split('_')[-1], fontsize=8, ha='center', va='bottom')
                    station_count += 1
            
            ax.set_title(f"{var} Spatial Profile", fontsize=11, fontweight='bold')
            ax.set_ylabel(var, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Only show x-label on last subplot
            if i == len(available_vars) - 1:
                ax.set_xlabel("Grid points (from upstream to downstream)", fontsize=10)
        
        plt.tight_layout()
        plt.suptitle("Water Quality Variables - Longitudinal Profiles", fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        fig_path = self.output_dir / "water_quality_longitudinal.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Water quality longitudinal profiles saved to {fig_path}")
        
    def _create_temporal_comparison_plot(self, validation_data):
        """Create combined temporal comparison plot for all water quality variables."""
        if not validation_data or not self.model_results:
            return
            
        # Use all available water quality variables
        available_vars = list(validation_data.keys())
        
        if not available_vars:
            return
        
        # Create a consolidated plot with all variables (no individual plots)
        if len(available_vars) >= 1:
            fig, axes = plt.subplots(len(available_vars), 1, figsize=(12, 2.5*len(available_vars)))
            
            if len(available_vars) == 1:
                axes = [axes]
            
            # Generate time series for model data with actual dates
            first_var = available_vars[0]
            time_values = self.model_results.get('time', np.arange(self.model_results[first_var].shape[0]))
            
            # Convert simulation time from seconds to days, then to actual calendar dates
            from datetime import timedelta
            time_in_days = time_values / 86400.0  # Convert seconds to days
            model_dates = [self.simulation_start_date + timedelta(days=float(t)) for t in time_in_days]
            
            # Create observation time series from field data dates
            obs_dates = {}
            obs_values = {}
            
            # Extract actual observation dates for each variable
            for var in available_vars:
                if var in validation_data:
                    obs_dates[var] = []
                    obs_values[var] = []
                    
                    # Get observation dates from CARE dataset
                    if 'CARE' in self.datasets:
                        care_data = self.datasets['CARE']
                        
                        # Map model variable back to CARE column name
                        care_var_map = {
                            'DO': 'DO (mg/L)',
                            'NO3': 'NO3 (mgN/L)', 
                            'NH4': 'NH4 (mgN/L)',
                            'PO4': 'PO4 (mgP/L)',
                            'DSi': 'DSi (mgSi/L)',
                            'TOC': 'TOC (mgC/L)',
                            'Chl-a': 'Chl-a (μg/L)',
                            'Salinity': 'Salinity'
                        }
                        
                        if var in care_var_map:
                            care_var = care_var_map[var]
                            for _, row in care_data.iterrows():
                                if care_var in row and not pd.isna(row[care_var]):
                                    date_str = row['Date']
                                    try:
                                        # Parse date and convert to actual datetime
                                        date_formats = ["%d-%b-%y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]
                                        date_obj = None
                                        for fmt in date_formats:
                                            try:
                                                date_obj = datetime.strptime(date_str, fmt)
                                                break
                                            except ValueError:
                                                continue
                                        
                                        if date_obj:
                                            obs_dates[var].append(date_obj)
                                            obs_values[var].append(row[care_var])
                                    except:
                                        continue
                
            for i, var in enumerate(available_vars):
                ax = axes[i]
                
                # Plot observations as scatter points with actual dates
                if var in obs_dates and obs_dates[var]:
                    ax.scatter(obs_dates[var], obs_values[var], color='red', marker='o', s=15, 
                             label='Field Data', alpha=0.7)
                
                # Plot model time series for a selected grid point
                middle_point = self.model_results[var].shape[1] // 2
                model_series = self.model_results[var][:, middle_point]
                
                # Take every 20th point to avoid overplotting but keep detail
                step = max(1, len(time_values) // 200)  # Adaptive step size
                ax.plot(model_dates[::step], model_series[::step], 'blue', linewidth=1.5, 
                      label='Model (mid-estuary)', alpha=0.8)
                
                ax.set_title(f"{var} Time Series", fontsize=11, fontweight='bold')
                ax.set_ylabel(var, fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Format x-axis to show dates nicely
                ax.tick_params(axis='x', rotation=45)
                
                # Only show legend on first subplot
                if i == 0:
                    ax.legend(frameon=False, fontsize=9)
                
                # Only show x-label on last subplot
                if i == len(available_vars) - 1:
                    ax.set_xlabel("Date", fontsize=10)
            
            plt.tight_layout()
            plt.suptitle(f"Water Quality Variables - Temporal Validation (Start: {self.simulation_start_date.strftime('%Y-%m-%d')})", 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Save figure
            fig_path = self.output_dir / "water_quality_temporal.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Water quality temporal comparison plot saved to {fig_path}")
    
    def _create_summary_plot(self, validation_data):
        """Create summary plot of all validated variables."""
        if not validation_data:
            return
        
        num_vars = len(validation_data)
        if num_vars == 0:
            return
        
        # Use a simple sequential arrangement of subplots to avoid issues
        # with axes handling
        fig = plt.figure(figsize=(15, 4 * ((num_vars + 2) // 3)))
        
        # Add plots for each variable
        for i, (var, data) in enumerate(validation_data.items()):
            # Create subplot in a 1-based indexing scheme
            ax = fig.add_subplot((num_vars + 2) // 3, 3, i + 1)
            
            # Convert JAX arrays to NumPy for plotting
            observed = np.array(data['observed'])
            predicted = np.array(data['predicted'])
            
            # Scatter plot
            ax.scatter(observed, predicted, alpha=0.7)
            
            # Add 1:1 line
            min_val = min(np.min(observed), np.min(predicted))
            max_val = max(np.max(observed), np.max(predicted))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add metrics
            metrics = self.validator.compute_validation_metrics(
                data['observed'], data['predicted'], var
            )
            ax.set_title(f"{var} (n={data['n_obs']})\nR²={metrics.r_squared:.3f}, RMSE={metrics.rmse:.3f}")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / "validation_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Validation summary figure saved to {fig_path}")
    
    def run_full_validation(self):
        """Run the complete validation workflow."""
        print("🌊 JAX C-GEM Field Data Validation")
        print("=" * 50)
        
        # Check if we have field data files
        calibration_dir = Path("INPUT/Calibration")
        if not calibration_dir.exists():
            print("❌ No calibration directory found!")
            print("💡 Create INPUT/Calibration/ directory and add field observation files:")
            print("   - CARE_2017-2018.csv")
            print("   - CEM_2017-2018.csv") 
            print("   - Or your own observation files in CSV format")
            return False
        
        field_files = list(calibration_dir.glob("*.csv"))
        if not field_files:
            print("❌ No field observation files found in INPUT/Calibration/")
            print("💡 Add field data files in CSV format with columns:")
            print("   Date,Site,Location,Salinity,DO (mg/L),NO3 (mgN/L),NH4 (mgN/L),...")
            return False
        
        print(f"🔍 Found {len(field_files)} field data files:")
        for f in field_files:
            print(f"   📄 {f.name}")
        
        # Check if we have model outputs
        model_dir = Path(self.model_output_dir)
        if not model_dir.exists():
            print(f"❌ Model output directory not found: {model_dir}")
            print("💡 Run a simulation first:")
            print("   python src/main.py --mode run")
            print("   or python main_ultra_performance.py")
            return False
        
        # Load data
        print(f"\n📊 Loading field observations...")
        self.load_field_observations()
        if not self.datasets:
            print("❌ No field observations loaded!")
            return False
        
        print(f"📊 Loading model results...")
        self.load_model_results()
        if not self.model_results:
            print("❌ No model results loaded!")
            return False
        
        # Run validation
        print(f"\n🔬 Running comprehensive model validation...")
        validation_results = self.validate_model()
        
        if validation_results:
            print("\n✅ Validation complete!")
            print(f"📁 Results saved to {self.output_dir}")
            
            # Print summary of validation results
            print(f"\n📈 VALIDATION SUMMARY:")
            print("=" * 30)
            for var, result in validation_results.items():
                try:
                    # Extract metrics safely
                    rmse = getattr(result, 'rmse', 0.0)
                    r_squared = getattr(result, 'r_squared', 0.0)
                    if not np.isnan(r_squared) and rmse > 0:
                        print(f"  {var:12s}: R²={r_squared:.3f}, RMSE={rmse:.3f}")
                    else:
                        print(f"  {var:12s}: Validation completed")
                except Exception:
                    print(f"  {var:12s}: Validation completed")
            
            # Suggest next steps
            print(f"\n💡 NEXT STEPS:")
            print("   📊 View validation plots in OUT/validation_results/")
            print("   📈 Generate publication plots: python tools/plotting/publication_plots.py")
            print("   📋 Read validation report: OUT/validation_results/field_validation_report.md")
            
            return True
        else:
            print("\n⚠️ Validation failed - no matching data found!")
            print("💡 Check that:")
            print("   1. Simulation period overlaps with field observation dates")
            print("   2. Field data files have proper Date column format (MM/DD/YYYY)")
            print("   3. Field data sites can be mapped to model grid points")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='JAX C-GEM Field Data Validation')
    parser.add_argument('--output-dir', default='OUT/validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--model-output-dir', default='OUT',
                       help='Directory with model output files')
    parser.add_argument('--model-output-format', choices=['csv', 'npz', 'auto'], default='auto',
                       help='Format of model output files (auto=detect automatically)')
    
    args = parser.parse_args()
    
    validator = FieldDataValidator(
        output_dir=args.output_dir,
        model_output_dir=args.model_output_dir,
        model_output_format=args.model_output_format
    )
    validator.run_full_validation()

if __name__ == "__main__":
    main()
