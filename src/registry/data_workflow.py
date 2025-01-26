import os
import json
import pandas as pd
from src.core.file_utils import ensure_directory_exists, save_results_to_json, extract_timestamp_from_filename, get_files_for_last_timestamps
from src.core.config_loader import load_data
from src.core.logger import logger
from src.core.config_loader import check_and_load_config_zillow
from src.core.scraper import process_zillow_datasets
from src.registry import DataSourceRegistry
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional

def manage_data_processing(registry: DataSourceRegistry, config: Dict) -> List[str]:
    """
    Manage data processing:
    - Create a JSON file with metadata and download links.
    - Download datasets if not already downloaded.
    - Return paths to all datasets.

    Args:
        registry (DataSourceRegistry): The data source registry.
        config (Dict): Configuration for the datasets.

    Returns:
        List[str]: Paths to all downloaded datasets.
    """
    
    # Unpack configuration
    source_name = config["name"]
    description = config["description"]

    # Initialize metadata
    metadata = {
        "description": description,
        "local_files": [],
        "status": "pending"
    }
    # Register the source
    registry.register_source(source_name, metadata)
    try:
        # Step 1: Load Zillow configuration
        zillow_data = check_and_load_config_zillow(config)
        if not zillow_data:
            raise RuntimeError("Failed to load Zillow configuration.")
        logger.debug(f"Loaded Zillow data: {zillow_data}")
        # Step 2: Extract timestamp from the JSON filename
        json_file_path = zillow_data.get("metadata", {}).get("file_path", "")
        json_file_name = zillow_data.get("metadata", {}).get("save_to_json_file", "")
        json_file = json_file_path +'//'+ json_file_name
        if not json_file_path:
            raise ValueError("No file_path found in the loaded Zillow data.")
        logger.info(f"Extracting timestamp from filename: {json_file}")  
        timestamp = extract_timestamp_from_filename(os.path.basename(json_file))
        logger.info(f"Extracted timestamp: {timestamp}")
        # Step 3: Construct data path using extracted timestamp
        #base_data_path = zillow_data.get("data_path", "data/temp_data")
        #data_path = os.path.join(base_data_path, f"scraper_data_{timestamp}")
        #if not os.path.exists(data_path):
        #    ensure_directory_exists(data_path)
        data_path = json_file_path+'//'+'scraper_data_'+timestamp
        # Step 4: Check if data_path contains files with last 3 timestamps
        existing_files = get_files_for_last_timestamps(data_path, max_timestamps=3)

        if existing_files:
            logger.info(f"Found {len(existing_files)} existing files in {data_path}.")
            logger.info(f"Files: {existing_files}")
            # If files exist, assume the data has already been processed
            return {"files": existing_files}
        # Update registry with local paths
        registry.update_source(source_name, {"local_files": existing_files, "status": "complete"})
        # Step 5: If no valid result is found, process the datasets
        logger.info(f"No valid files found in {data_path}. Starting data processing.")
        processed_result = process_zillow_datasets(zillow_data)

        # Step 6: Save the result
        result_file_path = os.path.join(data_path, f"zillow_result_{timestamp}.json")
        save_results_to_json(processed_result, os.path.basename(result_file_path), os.path.dirname(result_file_path))

        # Step 7: Manage saved files (keep only the last 3 unique timestamps)
        all_files = [
            os.path.join(data_path, file)
            for file in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, file))
        ]

        # Extract unique timestamps from filenames
        file_timestamps = defaultdict(list)
        for file in all_files:
            file_timestamp = extract_timestamp_from_filename(file)
            if file_timestamp:
                file_timestamps[file_timestamp].append(file)

        # Sort timestamps and keep only the last 3
        sorted_timestamps = sorted(file_timestamps.keys(), reverse=True)
        if len(sorted_timestamps) > 3:
            timestamps_to_delete = sorted_timestamps[3:]
            for timestamp in timestamps_to_delete:
                for file in file_timestamps[timestamp]:
                    try:
                        os.remove(file)
                        logger.info(f"Deleted old result file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file}: {e}")

        return processed_result

    except Exception as e:
        logger.error(f"Error in manage_data_processing: {e}")
        raise