from src.core.logger import logger
from pathlib import Path
from src.core.scraper import fetch_zillow_download_links
from src.core.metadata_utils import update_data_config
from src.core.file_utils import load_data
import os


def load_all_configs_via_master():
    """
    Loads all configurations specified in the master configuration file.

    This function reads the path to the master configuration file from the environment
    variable `MASTER_CONFIG_PATH`. It then loads the master configuration and all
    sub-configurations specified within it.

    The master configuration file should contain a section called `configs`, which includes
    a list of configuration types. Each type has associated configuration items that specify
    the file paths and optional metadata for the sub-configurations.

    Returns:
        dict: A dictionary containing the master configuration and all loaded sub-configurations.
            Example structure:
            ```python
            {
                "master_config": {...},
                "loaded_configs": {
                    "config_type_1": [
                        {
                            "metadata": {...},
                            "content": {...}
                        },
                        ...
                    ],
                    ...
                }
            }
            ```

    Raises:
        ValueError: If the `MASTER_CONFIG_PATH` is not set in the environment variables.
        RuntimeError: If the master configuration file cannot be loaded.

    Example:
        ```python
        # Assuming MASTER_CONFIG_PATH is set to "configs/master_config.json"
        all_configs = load_all_configs_via_master()
        print(all_configs["master_config"])
        print(all_configs["loaded_configs"]["config_type_1"][0]["content"])
        ```

    """
    # Load the path to the master configuration file from the environment variable
    master_config_path = os.getenv("MASTER_CONFIG_PATH")
    if not master_config_path:
        raise ValueError("MASTER_CONFIG_PATH not set in .env")
    
    master_config_path = Path(master_config_path).as_posix()
    
    # Load the master configuration file
    master_config = load_data(master_config_path, file_type="json", load_as="json", label="master_config")
    if not master_config:
        raise RuntimeError(f"Failed to load master config from {master_config_path}")

    # Initialize the structure for loaded configurations
    loaded_configs = {"master_config": master_config, "loaded_configs": {}}
    
    # Retrieve the section specifying sub-configurations
    configs_section = master_config.get("configs", {})
    configs_list = configs_section.get("configs_list",[])
    loaded_configs["configs_list"] = configs_list
    for config_type, config_data in configs_section.items():
        if isinstance(config_data, dict) and "items" in config_data:
            # Prepare a container for configurations of the current type
            loaded_configs["loaded_configs"][config_type] = []
            for config_item in config_data.get("items", []):
                # Retrieve the path and metadata for each sub-configuration
                config_path = config_item.get("path")
                config_label = config_item.get("tag")
                try:
                    # Load the sub-configuration using load_data()
                    config_content = load_data(config_path, file_type="json", load_as="json", label=config_label)
                    if config_content:
                        # Append the metadata and content of the sub-configuration
                        loaded_configs["loaded_configs"][config_type].append({
                            "metadata": config_item,
                            "content": config_content
                        })
                except Exception as e:
                    logger.error(f"Failed to load config '{config_label}' from {config_path}: {e}")
    
    return loaded_configs



def check_and_load_config_zillow(config):
    """
    Validates and loads the Zillow metadata configuration file. If the file is missing or invalid, it generates
    new data using `fetch_zillow_download_links` and updates the configuration.

    Args:
        config (dict): Configuration dictionary containing paths and settings for Zillow data processing.

    Returns:
        dict: Loaded JSON data from the file or newly fetched data.
    
    Raises:
        RuntimeError: If the configuration is invalid or the file cannot be loaded.
    """
    try:
        # Attempt to load the main config
        logger.info(f"Attempting to load main config from {config['data_config_path']}.")
        # Step 1: Load the main config
        main_config = load_data(
            config["data_config_path"], 
            file_type="json", 
            load_as="json", 
            label=f"{config['prefix']}_data_config"
        )
        if not main_config:
            logger.warning(f"Main config is empty or invalid: {config['data_config_path']}")
            main_config = {}
        else:
            logger.debug(f"Main config loaded successfully: {main_config}")

        # Step 2: Extract the file path from the loaded config
        file_key = f"{config['prefix']}_data_config"
        file_path = None
        if file_key in main_config and "file_path" in main_config[file_key]:
            file_path = main_config[file_key]["file_path"]
            logger.info(f"Found file path in config: {file_path}")

            # Step 3: Validate the file path and attempt to load the JSON file
            if os.path.exists(file_path):
                try:
                    logger.info(f"Loading JSON data from {file_path}.")
                    return load_data(file_path, file_type="json", load_as="json")
                except Exception as file_load_error:
                    logger.warning(f"Failed to load JSON file from {file_path}: {file_load_error}")
            else:
                logger.warning(f"File path {file_path} does not exist.")
        
        # Step 4: If no valid file path or file loading fails, fetch new data
        logger.info("File not found or invalid. Fetching new Zillow data.")
        fetch_result = fetch_zillow_download_links(config)
        
        if not fetch_result or "metadata" not in fetch_result or "file_path" not in fetch_result["metadata"]:
            logger.error("Fetch result did not return a valid metadata or file path.")
            return None
        
        # Return the newly fetched data as JSON
        if "data_path" not in fetch_result:
            fetch_result["data_path"] = config.get("data_path", "data/temp_data")
            
        # Save the new result file path in the main config
        new_file_path = fetch_result["metadata"].get("file_path")
        if new_file_path:
            logger.info(f"Updating config with new file path: {new_file_path}.")
            main_config[file_key] = {"file_path": new_file_path}
            update_data_config(config["data_config_path"], main_config)
        else:
            logger.error("Fetch result did not return a valid file path.")
            return None

        # Step 5: Return the newly fetched data
        try:
            return load_data(new_file_path, file_type="json", load_as="json")
        except Exception as e:
            logger.error(f"Failed to load newly fetched JSON file: {new_file_path}, error: {e}")
            return None

    except Exception as e:
        logger.error(f"Failed to check and load Zillow config: {e}")
        return None



