"""
Data loader for the JAX C-GEM model.
Handles reading and interpolation of all input time series data.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Dict, Any, Callable

# Try to import JAX, fall back to numpy if not available
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    HAS_JAX = False

class DataLoader:
    """Loads and manages all input data for the model."""
    
    def __init__(self, data_config: Dict[str, Any]):
        """Initialize data loader with configuration."""
        self.data_config = data_config
        self.interpolators = {}
        self._load_all_data()
    
    def _load_csv_data(self, filepath: str) -> tuple:
        """Load CSV data with time and value columns."""
        try:
            data = pd.read_csv(filepath, header=None, names=['time', 'value'])
            # Convert time from days to seconds
            time_seconds = np.array(data['time'].values) * 24 * 3600
            values = np.array(data['value'].values)
            return time_seconds, values
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {filepath}: {e}")

    def _load_hourly_csv_data(self, filepath: str) -> tuple:
        """Load CSV data with hourly time stamps."""
        try:
            data = pd.read_csv(filepath, header=None, names=['time', 'value'])
            # Convert time from hours to seconds (for hourly data)
            time_seconds = np.array(data['time'].values) * 3600  # hours to seconds
            values = np.array(data['value'].values)
            return time_seconds, values
        except Exception as e:
            raise RuntimeError(f"Failed to load hourly data from {filepath}: {e}")
    
    def _create_interpolator(self, time: np.ndarray, values: np.ndarray, 
                           kind: str = 'linear') -> Callable:
        """Create interpolation function."""
        # Use the first and last values for extrapolation
        fill_value = (float(values[0]), float(values[-1]))
        # type: ignore - scipy interp1d accepts tuple for fill_value
        return interp1d(time, values, kind=kind, bounds_error=False, 
                       fill_value=fill_value)  # type: ignore
    
    def _load_boundary_data(self):
        """Load boundary condition data."""
        for boundary in self.data_config['boundaries']:
            boundary_name = boundary['name']
            
            # Load each species data
            for species in ['Phy1', 'Phy2', 'Si', 'NO3', 'NH4', 'PO4', 
                          'O2', 'TOC', 'Sal', 'SPM', 'DIC', 'AT']:
                if species in boundary:
                    filepath = boundary[species]
                    time, values = self._load_csv_data(filepath)
                    
                    key = f"{boundary_name}_{species}"
                    self.interpolators[key] = self._create_interpolator(time, values)
    
    def _load_forcing_data(self):
        """Load forcing data (discharge, temperature, wind, elevation, light)."""
        for forcing in self.data_config['forcing']:
            forcing_name = forcing['name']
            
            # Load discharge data
            if 'DischargeFile' in forcing:
                time, values = self._load_csv_data(forcing['DischargeFile'])
                self.interpolators[f"{forcing_name}_Discharge"] = \
                    self._create_interpolator(time, values)
            
            # Load temperature data
            if 'TemperatureFile' in forcing:
                time, values = self._load_csv_data(forcing['TemperatureFile'])
                self.interpolators[f"{forcing_name}_Temperature"] = \
                    self._create_interpolator(time, values)
            
            # Load wind data
            if 'WindFile' in forcing:
                time, values = self._load_csv_data(forcing['WindFile'])
                self.interpolators[f"{forcing_name}_Wind"] = \
                    self._create_interpolator(time, values)
            
            # Load elevation data (hourly)
            if 'ElevationFile' in forcing:
                time, values = self._load_hourly_csv_data(forcing['ElevationFile'])
                self.interpolators[f"{forcing_name}_Elevation"] = \
                    self._create_interpolator(time, values)
            
            # Load light data (hourly)
            if 'LightFile' in forcing:
                time, values = self._load_hourly_csv_data(forcing['LightFile'])
                self.interpolators[f"{forcing_name}_Light"] = \
                    self._create_interpolator(time, values)
    
    def _load_tributary_data(self):
        """Load tributary data."""
        for tributary in self.data_config['tributaries']:
            trib_name = tributary['name']
            
            # Load discharge
            if 'discharge' in tributary:
                time, values = self._load_csv_data(tributary['discharge'])
                self.interpolators[f"{trib_name}_discharge"] = \
                    self._create_interpolator(time, values)
            
            # Load species data
            for species in ['Phy1', 'Phy2', 'Si', 'NO3', 'NH4', 'PO4', 
                          'O2', 'TOC', 'SPM', 'DIC', 'AT']:
                if species in tributary:
                    time, values = self._load_csv_data(tributary[species])
                    self.interpolators[f"{trib_name}_{species}"] = \
                        self._create_interpolator(time, values)
    
    def _load_all_data(self):
        """Load all data files specified in configuration."""
        self._load_boundary_data()
        self._load_forcing_data()
        self._load_tributary_data()
        
        print(f"✅ Loaded {len(self.interpolators)} data series")
    
    def get_value(self, key: str, time: float) -> float:
        """Get interpolated value at given time."""
        if key not in self.interpolators:
            raise KeyError(f"Data series '{key}' not found")
        
        return float(self.interpolators[key](time))
    
    def get_boundary_conditions(self, time: float) -> Dict[str, Dict[str, float]]:
        """Get all boundary conditions at given time."""
        bc = {}
        
        for boundary in self.data_config['boundaries']:
            boundary_name = boundary['name']
            bc[boundary_name] = {}
            
            for species in ['Phy1', 'Phy2', 'Si', 'NO3', 'NH4', 'PO4', 
                          'O2', 'TOC', 'Sal', 'SPM', 'DIC', 'AT']:
                key = f"{boundary_name}_{species}"
                if key in self.interpolators:
                    bc[boundary_name][species] = self.get_value(key, time)
        
        return bc
    
    def get_tributary_inputs(self, time: float) -> Dict[str, Dict[str, float]]:
        """Get all tributary inputs at given time."""
        tributaries = {}
        
        for tributary in self.data_config['tributaries']:
            trib_name = tributary['name']
            tributaries[trib_name] = {
                'cellIndex': int(tributary['cellIndex'])
            }
            
            # Get discharge
            discharge_key = f"{trib_name}_discharge"
            if discharge_key in self.interpolators:
                tributaries[trib_name]['discharge'] = self.get_value(discharge_key, time)
            
            # Get species concentrations
            for species in ['Phy1', 'Phy2', 'Si', 'NO3', 'NH4', 'PO4', 
                          'O2', 'TOC', 'SPM', 'DIC', 'AT']:
                key = f"{trib_name}_{species}"
                if key in self.interpolators:
                    tributaries[trib_name][species] = self.get_value(key, time)
        
        return tributaries
    
    def get_forcing_data(self, time: float) -> Dict[str, float]:
        """Get forcing data at given time."""
        forcing = {}
        
        for force in self.data_config['forcing']:
            forcing_name = force['name']
            
            for var in ['Discharge', 'Temperature', 'Wind', 'Elevation', 'Light']:
                key = f"{forcing_name}_{var}"
                if key in self.interpolators:
                    forcing[var] = self.get_value(key, time)
        
        return forcing