"""
Configuration parser for the JAX C-G    # Validate required parameters (check if they exist, don't require all)
    core_params = ['MAXT', 'WARMUP', 'DELTI', 'TS', 'DELXI', 'EL', 'AMPL']
    
    for param in core_params:
        if param not in config:
            print(f"⚠️  Warning: Missing core parameter '{param}'")
    
    # Calculate derived parameters if core parameters exist
    if 'EL' in config and 'DELXI' in config:
        config['M'] = config['EL'] // config['DELXI'] + 1  # Number of grid points
    if 'MAXT' in config:
        config['MAXT_seconds'] = config['MAXT']  # MAXT is now in seconds directly
    if 'WARMUP' in config:
        config['WARMUP_seconds'] = config['WARMUP']  # WARMUP is now in seconds directly
"""
import re
from typing import Dict, Any, List, Tuple
import numpy as np

def parse_model_config(config_file: str) -> Dict[str, Any]:
    """Parse model configuration file."""
    config = {}
    
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line or '=' not in line:
            continue
            
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.split('#')[0].strip()  # Remove inline comments
        
        # Parse different data types - prioritize numeric over boolean
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            config[key] = int(value)
        elif '.' in value or 'e-' in value.lower():
            try:
                config[key] = float(value)
            except ValueError:
                config[key] = value.strip('"')
        elif value.lower() in ['true']:  # Only 'true', not '1' (which is numeric)
            config[key] = True
        elif value.lower() in ['false']:  # Only 'false', not '0' (which is numeric)
            config[key] = False
        elif value.startswith('"') and value.endswith('"'):
            config[key] = value.strip('"')
        else:
            config[key] = value.strip('"')
    
    # Validate required parameters
    required_params = [
        'MAXT', 'WARMUP', 'DELTI', 'TS', 'DELXI', 'EL', 'AMPL',
        'num_segments', 'index_1', 'B1', 'LC1', 'Chezy1', 'Rs1',
        'index_2', 'B2', 'LC2', 'Chezy2', 'Rs2'
    ]
    
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in config file")
    
    # Calculate derived parameters
    config['M'] = config['EL'] // config['DELXI'] + 1  # Number of grid points
    # Convert MAXT and WARMUP from days to seconds for simulation
    config['MAXT_seconds'] = config['MAXT'] * 24 * 60 * 60  # Convert days to seconds
    config['WARMUP_seconds'] = config['WARMUP'] * 24 * 60 * 60  # Convert days to seconds
    
    return config

def parse_input_data_config(config_file: str) -> Dict[str, Any]:
    """Parse input data configuration file."""
    config = {
        'tributaries': [],
        'boundaries': [],
        'forcing': []
    }
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Split into sections
    sections = re.split(r'\n(?=name=)', content)
    
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip() and not line.strip().startswith('#')]
        if not lines:
            continue
            
        section_data = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                section_data[key.strip()] = value.strip()
        
        if 'name' not in section_data:
            continue
            
        # Categorize sections
        if section_data.get('type') == 'Tributary':
            config['tributaries'].append(section_data)
        elif section_data.get('type') in ['UpperBoundary', 'LowerBoundary']:
            config['boundaries'].append(section_data)
        elif section_data.get('type') == 'Forcing':
            config['forcing'].append(section_data)
    
    # Parse global settings
    global_lines = content.split('\n')
    for line in global_lines:
        line = line.strip()
        if line.startswith('numTributaries='):
            config['numTributaries'] = int(line.split('=')[1])
        elif line.startswith('tributaryEnabled='):
            config['tributaryEnabled'] = bool(int(line.split('=')[1]))
    
    return config

def validate_configurations(model_config: Dict[str, Any], data_config: Dict[str, Any]) -> None:
    """Validate that configurations are consistent."""
    # Check grid consistency - C-GEM requires EVEN number of grid points
    M = model_config['M']
    if M % 2 != 0:
        raise ValueError(f"Number of grid points M={M} must be even to match C-GEM implementation")
    
    # Check tributary indices are valid
    for tributary in data_config['tributaries']:
        cell_index = int(tributary['cellIndex'])
        if cell_index >= M:
            raise ValueError(f"Tributary cell index {cell_index} exceeds grid size {M}")
    
    # Check segment indices
    if model_config['index_2'] >= M:
        raise ValueError(f"Segment 2 index {model_config['index_2']} exceeds grid size {M}")
    
    print("✅ Configuration validation passed")