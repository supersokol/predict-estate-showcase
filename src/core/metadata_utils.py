from src.core.logger import logger
import json
import os
import time
from src.core.file_utils import load_data, sanitize_filename


# Metadata functions
def get_metadata(metadata_config, status, timestamps):
    """
    Extracts metadata based on the provided configuration, status, and timestamps.

    This function generates a metadata dictionary by processing metadata configuration rules,
    fetch status, and timestamp values.

    Args:
        metadata_config (dict): Configuration dictionary containing metadata rules and patterns.
            Example:
            {
                "timestamps": [
                    {"timestamp_type": "string", "timestamp_name": "start", "timestamp_format": "%Y-%m-%d"}
                ],
                "version": "1.0.0",
                "save_to_json": True,
                "save_to_json_file": "example"
            }
        status (str): Status of the fetch process (e.g., "success", "failure").
        timestamps (dict): Dictionary containing timestamp values.
            Example:
            {
                "start": "2025-01-01T00:00:00",
                "end": "2025-01-01T00:05:00"
            }

    Returns:
        dict: A dictionary with processed metadata based on the configuration.
            Example:
            {
                "timestamps": {"start": "2025-01-01T00:00:00"},
                "version": "1.0.0",
                "fetch_status": "success",
                "save_to_json_file": "example_2025-01-01T00:00:00.json"
            }

    Example:
        ```python
        metadata_config = {
            "timestamps": [
                {"timestamp_type": "string", "timestamp_name": "start", "timestamp_format": "%Y-%m-%d"}
            ],
            "version": "1.0.0",
            "save_to_json": True,
            "save_to_json_file": "example"
        }
        timestamps = {"start": "2025-01-01T00:00:00"}
        metadata = get_metadata(metadata_config, "success", timestamps)
        print(metadata)
        ```
    """
    metadata = {}
    for key, value in metadata_config.items():
        if key == "timestamps":
            metadata[key] = {}
            for timestamp_config in value:
                if timestamp_config["timestamp_type"] == "string":
                    metadata[key][timestamp_config["timestamp_name"]] = timestamps[timestamp_config["timestamp_name"]]#.strftime(timestamp_config["timestamp_format"]) 
                elif timestamp_config["timestamp_type"] == "date":
                    if timestamp_config["timestamp_name"] == 'month':
                            metadata[key][timestamp_config["timestamp_name"]] =  int(time.strftime(f"%{timestamp_config['timestamp_name'][0]}"))
                    else:
                        pass
        elif key in ["version","save_to_json","save_to_json_path"]:
            metadata[key] = value
        elif key == "save_to_json_file":
            metadata[key] = value + timestamps["start"] 
            metadata[key] = sanitize_filename(metadata[key]) + '.json'
        else:
            metadata[key] = None
    metadata["fetch_status"] = status
    return metadata 

def update_data_config(file_path, metadata, config_path="data//current_data_config.json", label_prefix = 'current'):
    """
    Updates or creates a JSON configuration file with metadata for a specified file.

    This function ensures that metadata for a given file is stored in the `current_data_config.json`.
    If the file does not exist, it creates a new configuration file.

    Args:
        file_path (str): Path to the file whose metadata is being updated.
        metadata (dict): Dictionary containing metadata to associate with the file.
        config_path (str, optional): Path to the configuration file. Defaults to "data//current_data_config.json".
        label_prefix (str, optional): Prefix for the configuration label. Defaults to "current".

    Returns:
        None

    Raises:
        Exception: If an error occurs during the update process.

    Example:
        ```python
        metadata = {"version": "1.0.0", "timestamps": {"start": "2025-01-01T00:00:00"}}
        update_data_config("path/to/file.csv", metadata)
        ```
    """
    try:
        # Normalize the file path to avoid duplicate base directory paths
        base_dir = os.path.dirname(config_path)
        if file_path.startswith(base_dir):
            file_path = file_path[len(base_dir):].lstrip(os.sep)

        file_name = os.path.basename(file_path)  
        file_path = os.path.join(base_dir, file_name)  
        file_path = os.path.normpath(file_path.replace(base_dir, ""))  

        logger.debug(f"Normalized file path: {file_path}")    
        config = load_data(config_path, file_type="json", load_as="json", label=label_prefix+"_data_config")
        if not config:
            config = {}
            
        # Update the configuration with new metadata
        config[label_prefix+"_data_config"] = {"file_path":file_path, "metadata": metadata}
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Config updated for {file_path}: {metadata}")
    except Exception as e:
        logger.error(f"Failed to update config for {file_path}: {e}")
        raise
    
def update_statistics(glob_statistics, key, value=1, error_type=None):
    """
    Updates global statistics with new values.

    This function increments statistics counters or updates error counts in the provided
    global statistics dictionary.

    Args:
        glob_statistics (dict): Dictionary containing global statistics.
            Example:
            {
                "unique_features_processed": set(),
                "errors": {"missing_file": 2},
                "total_processed": 100
            }
        key (str): The key in the statistics dictionary to update.
        value (int, optional): The value to increment the counter by. Defaults to 1.
        error_type (str, optional): The type of error to update in the "errors" key. Required if `key` is "errors".

    Returns:
        None

    Example:
        ```python
        statistics = {"unique_features_processed": set(), "errors": {}, "total_processed": 0}
        update_statistics(statistics, "unique_features_processed", value="feature1")
        update_statistics(statistics, "errors", error_type="missing_file")
        print(statistics)
        # Output:
        # {
        #     "unique_features_processed": {"feature1"},
        #     "errors": {"missing_file": 1},
        #     "total_processed": 0
        # }
        ```
    """
    if key == "unique_features_processed":
        glob_statistics[key].add(value)
    elif key == "errors" and error_type:
        glob_statistics[key][error_type] = glob_statistics[key].get(error_type, 0) + 1
    else:
        glob_statistics[key] += value