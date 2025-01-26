
# Registry for storing configurations
from typing import Callable, Dict


ConfigRegistry: Dict[str, dict] = {}

def register_config(config_id: str, metadata: dict = None):
    """
    Decorator for registering a configuration in the CONFIG_REGISTRY.
    :param config_id: Unique identifier for the configuration.
    :param metadata: Metadata describing the configuration.
    """
    def decorator(func: Callable):
        func.metadata = metadata  # Attach metadata to the configuration
        ConfigRegistry[config_id] = {"func": func, "metadata": metadata}
        return func
    return decorator