from typing import Callable, Dict
from core.config_loader import load_all_configs_via_master, check_and_load_config_zillow
import os
# Registry for storing processes
PROCESS_REGISTRY: Dict[str, Callable] = {}

def register_process(name: str, metadata: dict = None, save_to_file: bool = True, config_file: str = None):
    """
    Decorator for registering a process in the PROCESS_REGISTRY.
    :param name: Name of the process.
    :param metadata: Metadata describing the process.
    """
    loaded_configs = load_all_configs_via_master()
    
    if config_file and config_file in loaded_configs["configs_list"]:
        chosen_file= config_file
    else:
        master_config_path = os.getenv("MASTER_CONFIG_PATH")
        if not master_config_path:
            raise ValueError("MASTER_CONFIG_PATH not set in .env")
        chosen_file = os.path.basename(master_config_path)
    
    chosen_config = next(
            config["content"] 
            for config in loaded_configs["configs_list"]
            if config["content"]["config_name"] == chosen_file 
        )
    

    def decorator(func: Callable):
        func.metadata = metadata  # Attach metadata to the function
        PROCESS_REGISTRY[name] = func
        return func
    return decorator




from typing import Callable, Dict
from core.config_loader import load_all_configs_via_master, check_and_load_config_zillow
import os
# Registry for storing processes
PROCESS_REGISTRY: Dict[str, Callable] = {}

def register_process(name: str, metadata: dict = None, save_to_file: bool = True, config_file: str = None):
    """
    Decorator for registering a process in the PROCESS_REGISTRY.
    :param name: Name of the process.
    :param metadata: Metadata describing the process.
    """
    loaded_configs = load_all_configs_via_master()
    if config_file and config_file in loaded_configs["configs_list"]:
        chosen_file= config_file
    else:
        master_config_path = os.getenv("MASTER_CONFIG_PATH")
        if not master_config_path:
            raise ValueError("MASTER_CONFIG_PATH not set in .env")
        chosen_file = os.path.basename(master_config_path)
    
    chosen_config = next(
            config 
            for config in loaded_configs["configs_list"]
            if config == chosen_file 
        )
    

    def decorator(func: Callable):
        func.metadata = metadata  # Attach metadata to the function
        PROCESS_REGISTRY[name] = func
        return func
    return decorator
