from src.core.logger import logger
import json
import os
import time
from src.core.file_utils import load_data, sanitize_filename


# Metadata functions
def get_metadata(metadata_config, status, timestamps):
    """
    Extracts metadata based on the configuration and status.

    :param metadata_config: Configuration dictionary containing metadata rules and patterns.
    :param status: Status of the fetch process.
    :return: Dictionary with metadata based on the configuration.
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
    Updates or creates the 'current_data_config.json' file with metadata for the given file.
    :param file_path: Path to the file being updated in the config.
    :param metadata: Dictionary containing metadata to update.
    :param config_path: Path to the config file.
    """
    try:
        # Если путь уже содержит базовую директорию, избегаем дублирования
        base_dir = os.path.dirname(config_path)
        if file_path.startswith(base_dir):
            file_path = file_path[len(base_dir):].lstrip(os.sep)

        file_name = os.path.basename(file_path)  # Извлекаем имя файла
        file_path = os.path.join(base_dir, file_name)  # Объединяем с базовой директорией
        file_path = os.path.normpath(file_path.replace(base_dir, ""))  # Удаляем лишние части базовой директории

        logger.debug(f"Normalized file path: {file_path}")    
        config = load_data(config_path, file_type="json", load_as="json", label=label_prefix+"_data_config")
        if not config:
            config = {}
        # Update config with new metadata
        config[label_prefix+"_data_config"] = {"file_path":file_path, "metadata": metadata}
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Config updated for {file_path}: {metadata}")
    except Exception as e:
        logger.error(f"Failed to update config for {file_path}: {e}")
        raise
    
# Функция обновления статистики
def update_statistics(glob_statistics, key, value=1, error_type=None):
        if key == "unique_features_processed":
            glob_statistics[key].add(value)
        elif key == "errors" and error_type:
            glob_statistics[key][error_type] = glob_statistics[key].get(error_type, 0) + 1
        else:
            glob_statistics[key] += value