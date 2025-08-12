"""
Sparse data calibration module for the JAX C-GEM model.

This module implements sophisticated objective functions designed specifically
for calibrating against sparse field data from tidal estuaries, following the
core calibration philosophy of statistical aggregate comparison.

- Create multi-faceted objective function for realistic calibration
- Implement statistical comparison of longitudinal profiles
- Add seasonal cycle comparison (monthly means) at fixed stations
- Include variability magnitude comparison (standard deviations)
- Weight different data types appropriately with uncertainty quantification

Author: JAX C-GEM Team
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, NamedTuple, List
from dataclasses import dataclass
import numpy as np

@dataclass
class ObservationPoint:
    """Represents a field observation location."""
    name: str
    grid_index: int
    latitude: float
    longitude: float
    distance_from_mouth: float  # km
    depth: float  # m
    description: str

@dataclass
class FieldObservation:
    """Represents field measurement data."""
    location: ObservationPoint
    species: str
    values: jnp.ndarray  # Time series of observations
    times: jnp.ndarray   # Observation times (days)
    uncertainties: jnp.ndarray  # Measurement uncertainties
    quality_flags: jnp.ndarray  # Data quality indicators

class SparseDataManager:
    """
    Manager for sparse field observation data.
    
    This class handles the complex reality of sparse estuarine field data:
    - Irregular sampling in time and space
    - Missing data and data gaps
    - Variable measurement uncertainty
    - Different sampling frequencies for different variables
    - Seasonal bias in sampling campaigns
    """
    
    def __init__(self):
        """Initialize sparse data manager."""
        self.observations = {}
        self.observation_points = {}
        self.data_weights = {}
        
    def add_observation_point(self, point: ObservationPoint):
        """Add a monitoring station or sampling location."""
        self.observation_points[point.name] = point
        print(f"📍 Added observation point: {point.name} at {point.distance_from_mouth:.1f} km from mouth")
    
    def add_field_data(self, observation: FieldObservation):
        """Add field observation data."""
        key = f"{observation.location.name}_{observation.species}"
        self.observations[key] = observation
        
        # Calculate data weight based on quantity and quality
        n_points = len(observation.values)
        quality_score = jnp.mean(observation.quality_flags)
        uncertainty_score = 1.0 / (1.0 + jnp.mean(observation.uncertainties))
        
        weight = n_points * quality_score * uncertainty_score
        self.data_weights[key] = weight
        
        print(f"📊 Added {n_points} observations of {observation.species} at {observation.location.name}")
        print(f"   🎯 Data weight: {weight:.2f}")
    
    def compute_longitudinal_profiles(self, species: str, 
                                    time_window: Optional[Tuple[float, float]] = None) -> Dict[str, jnp.ndarray]:
        """
        Compute mean longitudinal profiles from sparse observations.
        
        This function creates spatial profiles by averaging temporal observations
        at each monitoring point, handling the reality that different locations
        have different amounts and quality of data.
        
        Args:
            species: Species name to analyze
            time_window: Optional time window to focus analysis (start_day, end_day)
            
        Returns:
            Dictionary with profile data and associated metadata
        """
        profile_data = {}
        
        for key, obs in self.observations.items():
            if obs.species == species:
                location = obs.location
                
                # Apply time window if specified
                if time_window is not None:
                    start_day, end_day = time_window
                    mask = (obs.times >= start_day) & (obs.times <= end_day)
                    values = obs.values[mask]
                    times = obs.times[mask]
                    uncertainties = obs.uncertainties[mask]
                else:
                    values = obs.values
                    times = obs.times
                    uncertainties = obs.uncertainties
                
                if len(values) > 0:
                    # Compute weighted mean (inverse variance weighting)
                    weights = 1.0 / (uncertainties**2 + 1e-12)
                    weighted_mean = jnp.sum(values * weights) / jnp.sum(weights)
                    
                    # Compute uncertainty of the mean
                    mean_uncertainty = 1.0 / jnp.sqrt(jnp.sum(weights))
                    
                    profile_data[location.name] = {
                        'distance_km': location.distance_from_mouth,
                        'grid_index': location.grid_index,
                        'mean_value': weighted_mean,
                        'uncertainty': mean_uncertainty,
                        'n_observations': len(values),
                        'time_span_days': jnp.max(times) - jnp.min(times) if len(times) > 1 else 0.0
                    }
        
        return profile_data
    
    def compute_seasonal_cycles(self, species: str, 
                              location: str) -> Dict[str, jnp.ndarray]:
        """
        Compute seasonal cycle from observations at a specific location.
        
        This function handles the common situation where observations are
        irregularly distributed across seasons, requiring careful temporal
        aggregation and gap handling.
        
        Args:
            species: Species name to analyze
            location: Observation location name
            
        Returns:
            Monthly means and uncertainties
        """
        key = f"{location}_{species}"
        
        if key not in self.observations:
            return {}
        
        obs = self.observations[key]
        
        # Group observations by month
        month_data = {month: [] for month in range(1, 13)}
        month_uncertainties = {month: [] for month in range(1, 13)}
        
        for i, time_day in enumerate(obs.times):
            # Convert time to month (simplified - assumes time is in days from start)
            # In reality, this would use proper datetime handling
            month = int((time_day % 365) // 30.4) + 1
            month = max(1, min(12, month))  # Ensure valid month
            
            month_data[month].append(obs.values[i])
            month_uncertainties[month].append(obs.uncertainties[i])
        
        # Compute monthly means
        monthly_means = []
        monthly_uncertainties = []
        months_with_data = []
        
        for month in range(1, 13):
            if len(month_data[month]) > 0:
                values = jnp.array(month_data[month])
                uncerts = jnp.array(month_uncertainties[month])
                
                # Weighted mean
                weights = 1.0 / (uncerts**2 + 1e-12)
                weighted_mean = jnp.sum(values * weights) / jnp.sum(weights)
                mean_uncertainty = 1.0 / jnp.sqrt(jnp.sum(weights))
                
                monthly_means.append(weighted_mean)
                monthly_uncertainties.append(mean_uncertainty)
                months_with_data.append(month)
        
        return {
            'months': jnp.array(months_with_data),
            'monthly_means': jnp.array(monthly_means),
            'monthly_uncertainties': jnp.array(monthly_uncertainties),
            'data_coverage': len(months_with_data) / 12.0
        }
    
    def compute_variability_metrics(self, species: str) -> Dict[str, float]:
        """
        Compute variability metrics across all observations.
        
        This provides system-wide measures of temporal and spatial variability
        that the model should reproduce.
        
        Args:
            species: Species name to analyze
            
        Returns:
            Variability metrics and statistics
        """
        all_values = []
        spatial_means = []
        temporal_stds = []
        
        for key, obs in self.observations.items():
            if obs.species == species:
                all_values.extend(obs.values.tolist())
                
                # Temporal variability at this location
                if len(obs.values) > 1:
                    temporal_std = jnp.std(obs.values)
                    temporal_stds.append(temporal_std)
                
                # Spatial contribution
                spatial_means.append(jnp.mean(obs.values))
        
        if len(all_values) == 0:
            return {}
        
        all_values = jnp.array(all_values)
        
        return {
            'overall_std': jnp.std(all_values),
            'overall_mean': jnp.mean(all_values),
            'coefficient_of_variation': jnp.std(all_values) / (jnp.mean(all_values) + 1e-12),
            'spatial_variability': jnp.std(jnp.array(spatial_means)) if len(spatial_means) > 1 else 0.0,
            'mean_temporal_variability': jnp.mean(jnp.array(temporal_stds)) if len(temporal_stds) > 0 else 0.0,
            'n_locations': len(spatial_means),
            'total_observations': len(all_values)
        }

class SparseDataObjective:
    """
    Advanced objective function for sparse data calibration.
    
    This class implements the core calibration philosophy: focus on statistical
    aggregates that capture the essential system behavior, rather than trying
    to match individual data points.
    """
    
    def __init__(self, data_manager: SparseDataManager):
        """Initialize objective function with sparse data."""
        self.data_manager = data_manager
        self.component_weights = self._setup_component_weights()
        
    def _setup_component_weights(self) -> Dict[str, float]:
        """Setup relative weights for different objective components."""
        return {
            # Longitudinal profiles (highest weight - these are most reliable)
            'salinity_profile': 3.0,      # Salinity is most reliable measurement
            'oxygen_profile': 2.0,        # Oxygen is well-measured but more variable
            'no3_profile': 1.5,           # Nutrients less reliable but important
            'nh4_profile': 1.5,
            'po4_profile': 1.0,
            
            # Seasonal cycles (medium weight - important for process validation)
            'oxygen_seasonal': 1.5,       # Oxygen seasonality is key indicator
            'phytoplankton_seasonal': 1.0, # Phytoplankton cycles important for productivity
            'no3_seasonal': 1.0,
            
            # Variability metrics (lower weight - helps with model realism)
            'salinity_variability': 0.5,
            'oxygen_variability': 0.8,
            'nutrient_variability': 0.3,
            
            # Data quality adjustments
            'data_coverage_bonus': 0.2,   # Bonus for good data coverage
            'uncertainty_penalty': 0.1    # Penalty for high uncertainty
        }
    
    @jax.jit
    def compute_profile_error(self, model_profile: jnp.ndarray,
                            observed_profile: jnp.ndarray,
                            observation_uncertainties: jnp.ndarray,
                            grid_indices: jnp.ndarray) -> float:
        """
        Compute error for longitudinal profile comparison.
        
        This function handles the spatial interpolation between model grid
        and observation locations, including proper uncertainty weighting.
        
        Args:
            model_profile: Model results along the grid
            observed_profile: Observed values at specific locations
            observation_uncertainties: Measurement uncertainties
            grid_indices: Grid indices corresponding to observation locations
            
        Returns:
            Weighted profile error
        """
        # Extract model values at observation locations
        model_at_obs = model_profile[grid_indices]
        
        # Compute residuals
        residuals = model_at_obs - observed_profile
        
        # Weight by inverse uncertainties (more weight to reliable observations)
        weights = 1.0 / (observation_uncertainties**2 + 1e-12)
        
        # Compute weighted mean squared error
        weighted_mse = jnp.sum(weights * residuals**2) / jnp.sum(weights)
        
        return weighted_mse
    
    @jax.jit
    def compute_seasonal_error(self, model_seasonal: jnp.ndarray,
                             observed_seasonal: jnp.ndarray,
                             seasonal_uncertainties: jnp.ndarray) -> float:
        """
        Compute error for seasonal cycle comparison.
        
        This function handles the comparison of seasonal patterns, which is
        crucial for validating the temporal dynamics of the model.
        
        Args:
            model_seasonal: Model monthly means
            observed_seasonal: Observed monthly means
            seasonal_uncertainties: Uncertainties in monthly means
            
        Returns:
            Weighted seasonal error
        """
        # Handle cases where not all months have data
        mask = ~jnp.isnan(observed_seasonal)
        
        if jnp.sum(mask) == 0:
            return 0.0
        
        model_masked = model_seasonal[mask]
        obs_masked = observed_seasonal[mask]
        uncert_masked = seasonal_uncertainties[mask]
        
        # Compute residuals
        residuals = model_masked - obs_masked
        
        # Weight by inverse uncertainties
        weights = 1.0 / (uncert_masked**2 + 1e-12)
        
        # Compute weighted mean squared error
        weighted_mse = jnp.sum(weights * residuals**2) / jnp.sum(weights)
        
        return weighted_mse
    
    @jax.jit
    def compute_variability_error(self, model_std: float,
                                observed_std: float,
                                std_uncertainty: float) -> float:
        """
        Compute error for variability magnitude comparison.
        
        This ensures the model reproduces the correct magnitude of temporal
        and spatial variability observed in the system.
        
        Args:
            model_std: Model standard deviation
            observed_std: Observed standard deviation
            std_uncertainty: Uncertainty in observed standard deviation
            
        Returns:
            Variability error
        """
        relative_error = (model_std - observed_std) / (observed_std + 1e-12)
        uncertainty_weight = 1.0 / (std_uncertainty + 1e-12)
        
        return uncertainty_weight * relative_error**2
    
    def compute_total_objective(self, model_results: Dict[str, jnp.ndarray],
                              simulation_time_days: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute the complete multi-objective function for sparse data calibration.
        
        This is the main objective function that combines all components
        according to the sparse data calibration philosophy.
        
        Args:
            model_results: Complete model simulation results
            simulation_time_days: Total simulation time for proper averaging
            
        Returns:
            Total objective value and breakdown by component
        """
        total_error = 0.0
        error_breakdown = {}
        
        # 1. LONGITUDINAL PROFILES
        profile_species = ['salinity', 'oxygen', 'no3', 'nh4', 'po4']
        
        for species in profile_species:
            if species in model_results:
                # Get sparse observations for this species
                profile_data = self.data_manager.compute_longitudinal_profiles(species)
                
                if len(profile_data) > 0:
                    # Extract observed data
                    locations = list(profile_data.keys())
                    grid_indices = jnp.array([profile_data[loc]['grid_index'] for loc in locations])
                    observed_values = jnp.array([profile_data[loc]['mean_value'] for loc in locations])
                    uncertainties = jnp.array([profile_data[loc]['uncertainty'] for loc in locations])
                    
                    # Compute model profile (time-averaged)
                    model_profile = jnp.mean(model_results[species], axis=0)
                    
                    # Compute profile error
                    profile_error = self.compute_profile_error(
                        model_profile, observed_values, uncertainties, grid_indices
                    )
                    
                    weight = self.component_weights.get(f'{species}_profile', 1.0)
                    total_error += weight * profile_error
                    error_breakdown[f'{species}_profile'] = profile_error
        
        # 2. SEASONAL CYCLES
        seasonal_species = ['oxygen', 'phytoplankton', 'no3']
        
        for species in seasonal_species:
            if species in model_results:
                # For each observation location
                for location_name in self.data_manager.observation_points.keys():
                    seasonal_data = self.data_manager.compute_seasonal_cycles(species, location_name)
                    
                    if len(seasonal_data) > 0 and seasonal_data['data_coverage'] > 0.25:  # At least 3 months
                        # Compute model seasonal cycle at this location
                        location = self.data_manager.observation_points[location_name]
                        grid_idx = location.grid_index
                        
                        # Extract model data at this location
                        model_at_location = model_results[species][:, grid_idx]
                        
                        # Compute monthly means (simplified)
                        n_months = min(12, len(model_at_location) // 30)
                        model_monthly = []
                        for month in range(n_months):
                            start_idx = month * (len(model_at_location) // n_months)
                            end_idx = (month + 1) * (len(model_at_location) // n_months)
                            monthly_mean = jnp.mean(model_at_location[start_idx:end_idx])
                            model_monthly.append(monthly_mean)
                        
                        model_seasonal = jnp.array(model_monthly)
                        
                        # Interpolate to match observed months
                        observed_months = seasonal_data['months']
                        observed_values = seasonal_data['monthly_means']
                        observed_uncerts = seasonal_data['monthly_uncertainties']
                        
                        # Simple interpolation (could be improved)
                        if len(model_seasonal) >= len(observed_values):
                            model_interp = model_seasonal[:len(observed_values)]
                        else:
                            # Pad with mean if needed
                            model_mean = jnp.mean(model_seasonal)
                            model_interp = jnp.concatenate([
                                model_seasonal, 
                                jnp.full(len(observed_values) - len(model_seasonal), model_mean)
                            ])
                        
                        # Compute seasonal error
                        seasonal_error = self.compute_seasonal_error(
                            model_interp, observed_values, observed_uncerts
                        )
                        
                        weight = self.component_weights.get(f'{species}_seasonal', 1.0)
                        total_error += weight * seasonal_error
                        error_breakdown[f'{species}_seasonal_{location_name}'] = seasonal_error
        
        # 3. VARIABILITY METRICS
        variability_species = ['salinity', 'oxygen', 'nutrient']
        
        for species in variability_species:
            species_key = 'no3' if species == 'nutrient' else species
            if species_key in model_results:
                # Compute model variability
                model_std = jnp.std(model_results[species_key])
                
                # Get observed variability
                var_metrics = self.data_manager.compute_variability_metrics(species_key)
                
                if 'overall_std' in var_metrics:
                    observed_std = var_metrics['overall_std']
                    # Estimate uncertainty in standard deviation (rough approximation)
                    n_obs = var_metrics.get('total_observations', 1)
                    std_uncertainty = observed_std / jnp.sqrt(2 * n_obs)
                    
                    variability_error = self.compute_variability_error(
                        model_std, observed_std, std_uncertainty
                    )
                    
                    weight = self.component_weights.get(f'{species}_variability', 0.5)
                    total_error += weight * variability_error
                    error_breakdown[f'{species}_variability'] = variability_error
        
        # 4. DATA QUALITY ADJUSTMENTS
        # Bonus for good data coverage
        total_observations = sum(len(obs.values) for obs in self.data_manager.observations.values())
        coverage_bonus = -self.component_weights['data_coverage_bonus'] * jnp.log(total_observations + 1)
        total_error += coverage_bonus
        error_breakdown['coverage_bonus'] = coverage_bonus
        
        return total_error, error_breakdown

def create_synthetic_sparse_data(model_config: Dict[str, Any]) -> SparseDataManager:
    """
    Create synthetic sparse observation data for testing the calibration system.
    
    This function creates realistic sparse data patterns that mimic real
    estuarine monitoring programs for testing and validation purposes.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        SparseDataManager with synthetic observation data
    """
    data_manager = SparseDataManager()
    
    # Create observation points at typical monitoring locations
    M = model_config.get('M', 101)
    EL = model_config.get('EL', 200000)  # Estuary length in meters
    
    # Common observation points: mouth, middle estuary, upstream
    observation_points = [
        ObservationPoint("mouth", 5, 10.5, 106.8, 5, 8, "Near estuary mouth"),
        ObservationPoint("middle", M//3, 10.6, 106.7, 40, 12, "Middle estuary"),
        ObservationPoint("upstream", 2*M//3, 10.7, 106.6, 80, 15, "Upstream location"),
        ObservationPoint("head", M-10, 10.8, 106.5, 150, 8, "Near estuary head"),
    ]
    
    for point in observation_points:
        data_manager.add_observation_point(point)
    
    # Create synthetic observations with realistic sparse patterns
    import numpy as np
    np.random.seed(42)  # For reproducible synthetic data
    
    species_configs = {
        'salinity': {'mean': 15, 'std': 8, 'uncertainty': 0.5, 'n_obs_range': (20, 50)},
        'oxygen': {'mean': 200, 'std': 50, 'uncertainty': 10, 'n_obs_range': (15, 30)},
        'no3': {'mean': 50, 'std': 20, 'uncertainty': 5, 'n_obs_range': (10, 25)},
        'nh4': {'mean': 20, 'std': 10, 'uncertainty': 2, 'n_obs_range': (8, 20)},
        'po4': {'mean': 5, 'std': 2, 'uncertainty': 0.5, 'n_obs_range': (5, 15)},
    }
    
    for point in observation_points:
        for species, config in species_configs.items():
            # Random number of observations (sparse!)
            n_obs = np.random.randint(*config['n_obs_range'])
            
            # Random observation times over 1 year
            times = np.sort(np.random.uniform(0, 365, n_obs))
            
            # Synthetic values with gradient and seasonality
            distance_factor = point.distance_from_mouth / 150.0  # Normalize to 0-1
            
            base_values = []
            for t in times:
                # Seasonal component
                seasonal = config['std'] * 0.3 * np.sin(2 * np.pi * t / 365)
                
                # Spatial gradient (different for each species)
                if species == 'salinity':
                    spatial = config['mean'] * (1 - 0.8 * distance_factor)  # Decreases upstream
                elif species == 'oxygen':
                    spatial = config['mean'] * (1 - 0.3 * distance_factor)  # Slight decrease upstream
                else:
                    spatial = config['mean'] * (1 + 0.5 * distance_factor)  # Increases upstream (nutrients)
                
                # Random noise
                noise = np.random.normal(0, config['std'] * 0.2)
                
                value = max(0, spatial + seasonal + noise)  # Ensure positive
                base_values.append(value)
            
            values = jnp.array(base_values)
            uncertainties = jnp.full_like(values, config['uncertainty'])
            quality_flags = jnp.ones_like(values)  # All good quality
            
            observation = FieldObservation(
                location=point,
                species=species,
                values=values,
                times=jnp.array(times),
                uncertainties=uncertainties,
                quality_flags=quality_flags
            )
            
            data_manager.add_field_data(observation)
    
    print(f"✅ Created synthetic sparse dataset:")
    print(f"   📍 {len(observation_points)} observation locations")
    print(f"   🧪 {len(species_configs)} species")
    print(f"   📊 Total observations: {sum(len(obs.values) for obs in data_manager.observations.values())}")
    
    return data_manager

# Export key functions
__all__ = [
    'SparseDataManager',
    'SparseDataObjective', 
    'ObservationPoint',
    'FieldObservation',
    'create_synthetic_sparse_data'
]
