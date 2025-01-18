import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path="configs/logging_config.yaml"):
    """Setup logging configuration from a YAML file."""
    config_path = Path(config_path)
    if config_path.is_file():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging configuration file not found: {config_path}. Using default configuration.")
