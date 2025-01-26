from src.core.logger import logger

from src.core.scraper import fetch_zillow_download_links
from src.core.metadata_utils import update_data_config
from src.core.file_utils import load_data
import os


def load_all_configs_via_master():
    # Загружаем путь к мастер-конфигу из .env
    master_config_path = os.getenv("MASTER_CONFIG_PATH")
    if not master_config_path:
        raise ValueError("MASTER_CONFIG_PATH not set in .env")
    
    # Загружаем мастер-конфиг
    master_config = load_data(master_config_path, file_type="json", load_as="json", label="master_config")
    if not master_config:
        raise RuntimeError(f"Failed to load master config from {master_config_path}")

    # Загружаем подконфиги
    loaded_configs = {"master_config": master_config, "loaded_configs": {}}
    configs_section = master_config.get("configs", {})
    for config_type, config_data in configs_section.items():
        loaded_configs["loaded_configs"][config_type] = []
        for config_item in config_data.get("items", []):
            config_path = config_item.get("path")
            config_label = config_item.get("tag")
            try:
                # Загружаем подконфиг с использованием load_data
                config_content = load_data(config_path, file_type="json", load_as="json", label=config_label)
                if config_content:
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



