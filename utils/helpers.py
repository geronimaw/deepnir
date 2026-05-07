"""
Utility Functions

Common utility functions used across the deep_nir package.

Example usage:
    from deep_nir.utils import load_config, setup_logging
    
    config = load_config("configs/model_config.yaml")
    logger = setup_logging()
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_config("configs/model_config.yaml")
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("deep_nir")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_data_shape(
    data: Any,
    expected_shape: Optional[tuple] = None,
    min_samples: int = 1
) -> bool:
    """
    Validate data shape and size.
    
    Args:
        data: Input data (numpy array or similar)
        expected_shape: Expected shape (None for any dimension)
        min_samples: Minimum number of samples required
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    raise NotImplementedError("Implement data validation")


__all__ = [
    "load_config",
    "save_config",
    "setup_logging",
    "validate_data_shape",
]
