import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from src.core.logger import logger
from src.core.file_utils import sanitize_filename, load_data, clean_text, ensure_directory_exists, save_results_to_json
from src.core.db_manager import DatabaseManager
from src.core.metadata_utils import get_metadata, update_data_config, update_statistics
from src.analysis.zillow_analysis import analyze_and_log_zillow_data
import time
import os
import re
import pandas as pd
import sqlite3

# Zillow Scraper Functions

def download_zillow_csv_file(csv_url, save_path="zillow_data.csv"):
    """
    Downloads a CSV file from the specified link and saves it to disk.

    :param csv_url: str - Link to the CSV file.
    :param save_path: str - Path to save the file. Default: zillow_data.csv.
    """
    try:
        logger.info(f"Starting file download from URL: {csv_url}")
        response = requests.get(csv_url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            save_path = save_path  # Path to save the file in the data folder
            # Save the file
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"File successfully downloaded and saved to: {save_path}")
        else:
            logger.error(f"File download error: response code {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"An error occurred while downloading the file: {e}")
    except Exception as e:
        logger.error(f"General error: {e}")

        #

def parse_full_name_zillow(full_name, zillow_dataset_label, config):
    """
    Parses the given full_name string to extract type, data_type, housing types, and features based on configuration.

    :param full_name: The full name string to parse.
    :param zillow_dataset_label: The dataset label determining parsing rules.
    :param config: Configuration dictionary containing parsing rules and patterns.
    :return: A dictionary containing parsed components.
    """
    # Extract configuration for the dataset label
    dataset_config = config.get("datasets", {}).get(zillow_dataset_label, {})
    
    # Extract patterns and options
    type_pattern = dataset_config.get("type_pattern", "")
    data_type_pattern = dataset_config.get("data_type_pattern", "")
    housing_types = dataset_config.get("housing_types", [])
    feature_patterns = dataset_config.get("feature_patterns", [])

    # Extract type
    match_type = re.search(type_pattern, full_name)
    parsed_type = match_type.group(1) if match_type else None

    # Extract data_type
    parsed_data_type = data_type_pattern if data_type_pattern in full_name else None

    # Extract housing types and features
    extracted_housing_types = []
    found_housing_type = False  # Track if a housing type is found

    for housing_type in housing_types:
        if housing_type in full_name:
            found_housing_type = True
            features = []

            # Extract features dynamically based on the configuration
            for feature_name, feature_config in feature_patterns.items():
                if isinstance(feature_config, list):  # Handle complex feature patterns like "measure"
                    for feature in feature_config:
                        if re.search(feature["condition"], full_name):
                            features.append({feature_name: feature["key"]})
                            break  # Stop checking further conditions for this feature
                else:  # Handle simple feature patterns
                    match = re.search(feature_config, full_name)
                    if match:
                        features.append({feature_name: match.group(1)})


            extracted_housing_types.append({
                "housing_type": housing_type,
                "features": [
                    {
                        "full_name": full_name,
                        "geography": None,  # This should be filled externally during processing
                        "additional_features": features
                    }
                ]
            })

            # Skip checking further housing types once one is found
            break

    return {
        "type": parsed_type,
        "data_type": parsed_data_type,
        "housing_types": extracted_housing_types
    }

def process_geography(geography, feature, dataset_name, glob_statistics, data_path):
    """
    Processes a single geography by downloading, loading, and saving data.

    This function downloads data for a specific geography, loads it into a DataFrame, and
    saves it to a file. It also updates global statistics.

    Args:
        geography (dict): Geography information containing download links and metadata.
        feature (dict): Feature information including additional attributes and full name.
        dataset_name (str): Name of the dataset being processed.
        glob_statistics (dict): Global statistics dictionary to track processing metrics.
        data_path (str): Path to save downloaded data files.

    Returns:
        tuple: A tuple containing:
            - dataset_name (str): Name of the dataset.
            - dataframe (pd.DataFrame): Loaded data as a DataFrame.
            - table_name (str): Name of the database table for the dataset.
            - file_path (str): Path to the saved file.

    Raises:
        Exception: If an error occurs during processing.

    Example:
        ```python
        geography = {"geography_value": "US", "download_link": "https://example.com/data.csv"}
        feature = {"additional_features": [], "full_name": "FeatureName"}
        dataset_name = "example_dataset"
        glob_statistics = {}
        data_path = "data/"
        result = process_geography(geography, feature, dataset_name, glob_statistics, data_path)
        ```
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.debug(f"Geography: {geography}")
    feature_additional_features = feature["additional_features"]
    logger.debug(f"Feature additional features: {feature_additional_features}")
    feature_download_link = geography["download_link"]
    logger.info(f"Feature download link: {feature_download_link}")
    feature_full_name = feature["full_name"]
    logger.debug(f"Feature full name: {feature_full_name}")
    
    file_name = f"{feature_full_name}_{geography['geography_value']}_{timestamp}"
    file_name = sanitize_filename(file_name)
    file_name += ".csv"
    ensure_directory_exists(data_path)
    file_path = os.path.join(data_path, file_name)
    logger.info(f"File path: {file_path}")
    update_statistics(glob_statistics, "unique_features_processed", value=feature_full_name)
    try:
        # Download and load the data
        dataframe = download_and_load_csv(
            download_link=feature_download_link ,
            file_name=file_name,
            file_path=file_path,
            label=feature_full_name
        )
        update_statistics(glob_statistics, "features_with_filename")
        update_statistics(glob_statistics, "files_downloaded")
        
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            logger.warning(f"Invalid or empty DataFrame for {file_name}. Skipping.")
            return None, None, None, None
        
        # Create a sanitized table name
        table_name = f"{dataset_name}_{feature_full_name}_{geography['geography_value']}_{timestamp}"
        update_statistics(glob_statistics, "dataframes_loaded")
        table_name = sanitize_filename(table_name)  
        return dataset_name, dataframe, table_name, file_path
    except Exception as e:
        logger.error(f"Error processing geography {geography['geography_value']} for feature {feature_full_name}: {e}")
        update_statistics(glob_statistics, "errors", error_type="geography_processing")
        return None, None, None, None

def download_and_load_csv(download_link, file_name, file_path, label):
    """
    Downloads a CSV file from a given link and loads it into a DataFrame.

    Args:
        download_link (str): URL to download the CSV file.
        file_name (str): Name of the file to save locally.
        file_path (str): Path to save the downloaded file.
        label (str): Label for the data being loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.

    Raises:
        Exception: If an error occurs during the download or loading process.

    Example:
        ```python
        dataframe = download_and_load_csv(
            download_link="https://example.com/data.csv",
            file_name="example.csv",
            file_path="data/example.csv",
            label="Example Label"
        )
        ```
    """
    try:
        # Download CSV file
        download_zillow_csv_file(download_link, save_path=file_path)
        logger.info(f"Downloaded file: {file_name}")
        
        # Load CSV into DataFrame
        dataframe = load_data(file_path, file_type="csv", load_as="dataframe", label=label)
        #logger.debug(f"Data loaded into DataFrame: {dataframe.shape}: {file_name}\n{dataframe.head()}")
        
        # Replace missing values with "N/A"
        dataframe = dataframe.fillna("N/A")
        return dataframe
    except Exception as e:
        logger.error(f"Error during download and loading of {file_name}: {e}")
        return None

def fetch_zillow_download_links(config):
    """
    Fetches all available download links for Zillow datasets based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing parsing rules, patterns,
            and options for fetching Zillow data.

    Returns:
        dict: A dictionary containing the downloaded Zillow datasets and metadata.

    Raises:
        Exception: If an error occurs during the fetching process.

    Example:
        ```python
        config = {
            "zillow_base_url": "https://www.zillow.com/research/data/",
            "zillow_data_labels_target": ["Home Values"],
            "geography_options": [["US"]],
            "metadata_config": {},
            "save_to_json": True,
            "save_to_json_path": "data/",
            "save_to_json_file": "zillow_results"
        }
        result = fetch_zillow_download_links(config)
        ```
    """
    timestamp_start = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"Starting Zillow data fetch process at {timestamp_start}.")
    
    data_path = config.get("data_path", "data/temp_data")
    data_path = os.path.join(data_path, f"scraper_data_{timestamp_start}")
    ensure_directory_exists(data_path)
    
    # Initialize Selenium WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    logger.info("Selenium WebDriver configured.")

    result = {"zillow_datasets": {}, "status": "success", "metadata": {}, "data_path": data_path}
    
    def merge_housing_types(zillow_datasets):
        """
        Merge housing_types for identical (type, data_type) pairs in the Zillow dataset.

        :param zillow_datasets: Dictionary containing the Zillow datasets.
        :return: Dictionary with merged housing_types for identical (type, data_type) pairs.
        """
        merged_results = {}

        for dataset_label, datasets in zillow_datasets.items():
            for dataset in datasets:
                key = (dataset["type"], dataset["data_type"])

                # Initialize entry if not exists
                if key not in merged_results:
                    merged_results[key] = {
                        "type": dataset["type"],
                        "data_type": dataset["data_type"],
                        "housing_types": []
                    }

                # Merge housing_types
                merged_results[key]["housing_types"].extend(dataset["housing_types"])

        # Convert merged_results back into the original structure
        final_results = []
        for key, value in merged_results.items():
            final_results.append(value)

        return final_results
    
    try:
        # Open the Zillow page
        driver.get(config["zillow_base_url"])
        time.sleep(5)  # Allow the page to load
        
        # Validate target labels
        zillow_data_labels_available = config.get("zillow_data_labels_available", [])
        logger.info(f"All available amount: {len(zillow_data_labels_available)}, labels:{zillow_data_labels_available}")
        zillow_data_labels_target = config.get("zillow_data_labels_target", [])
        logger.info(f"Target labels amount: {len(zillow_data_labels_target)}, labels: {zillow_data_labels_target}")
        valid_zillow_data_labels_target = [label for label in zillow_data_labels_target if label in zillow_data_labels_available]
        logger.info(f"Valid labels amount: {len(valid_zillow_data_labels_target)}, labels: {valid_zillow_data_labels_target}")
        
        # Process each target label
        for label in valid_zillow_data_labels_target:
            logger.info(f"Processing label: {label}")
            # Locate and process dropdowns and download links
            dropdown_id_1 = f"{label}dropdown-1"
            dropdown_id_2 = f"{label}dropdown-2"
            download_link_id = f"{label}download-link"

            try:
                # Locate the first dropdown and extract dataset label
                dropdown_element_1 = driver.find_element(By.ID, dropdown_id_1)
                dataset_label = dropdown_element_1.get_attribute("data-set")
                zillow_dataset_label = f"zillow_{clean_text(dataset_label).lower().replace(' ', '_')}"

                # Fetch data types from the first dropdown
                data_dropdown = Select(dropdown_element_1)
                data_types = [option.text for option in data_dropdown.options]

                dataset_results = []

                for data_type in data_types:
                    logger.info(f"Processing data type: {data_type}")
                    data_dropdown.select_by_visible_text(data_type)
                    time.sleep(1)

                    geography_list = []

                    for geography_group in config["geography_options"]:
                        for geography in geography_group:
                            try:
                                dropdown_element_2 = driver.find_element(By.ID, dropdown_id_2)
                                geo_dropdown = Select(dropdown_element_2)
                                geo_dropdown.select_by_visible_text(geography)
                                time.sleep(1)

                                # Extract the download link
                                download_link = driver.find_element(By.ID, download_link_id).get_attribute("href")
                                geography_normalized = geography_group[0]

                                if not any(entry['geography_value'] == geography_normalized and entry['download_link'] == download_link for entry in geography_list):
                                    geography_list.append({
                                        "geography_value": geography_normalized,
                                        "download_link": download_link
                                    })
                            except Exception as e:
                                pass
                                #logger.warning(f"Geography '{geography}' not available for data type '{data_type}': {e}")

                    # Parse full name using the zillow_dataset_label and enrich with geography
                    parsed_data = parse_full_name_zillow(data_type, zillow_dataset_label, config)
                    for housing_type in parsed_data["housing_types"]:
                        for feature in housing_type["features"]:
                            feature["geography"] = geography_list
                            feature["full_name"] = data_type

                    dataset_results.append(parsed_data)
                result["zillow_datasets"].setdefault(zillow_dataset_label, []).extend(dataset_results)
                result["zillow_datasets"][zillow_dataset_label] = merge_housing_types({zillow_dataset_label: result["zillow_datasets"][zillow_dataset_label]})
                # result["zillow_datasets"][zillow_dataset_label] = dataset_results

            except Exception as e:
                logger.error(f"Error processing label {label}: {e}")
        

        result["status"]="success"
        
        timestamp_end = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Zillow data fetch process completed at {timestamp_end}.")
        duration = time.time() - time.mktime(time.strptime(timestamp_start, "%Y%m%d-%H%M%S"))
        logger.info(f"Duration: {duration} seconds.")
        
        metadata_config = config.get("metadata_config", {})
        
        timestamps = {"start": timestamp_start, "end": timestamp_end, "duration": duration}
        
        result["metadata"] = get_metadata(metadata_config, result["status"], timestamps)
        result["metadata"]["file_path"] = config.get("save_to_json_path", "") + "/" + \
                                        config.get("save_to_json_file", "") 
        logger.info(f"Metadata collected:{result['metadata']}")
        logger.info(f"Analysis started...")
        result = analyze_and_log_zillow_data(result)
        logger.info("Analysis completed.")
        if config["save_to_json"]:
            logger.info("Saving results to JSON...")
            try:
                
                output_file_path = config["save_to_json_path"] 
                output_file_name = config["save_to_json_file"] + timestamps["start"] + ".json"  # Path to save the file in the data folder
                save_results_to_json(result, output_file_name, output_file_path)
            except Exception as save_error:
                logger.error(f"Error saving results to JSON: {save_error}")
        # Update config with fetched metadata
        update_data_config(config["save_to_json_file"] + timestamps["start"] + ".json", result["metadata"], config_path=config["data_config_path"], label_prefix = "zillow")
        logger.info(f"Config {config['data_config_path']} updated with fetched metadata.")
        return result

    except Exception as e:
        logger.error(f"An error occurred during the fetch process: {e}")
        if not result:
            result = {"zillow_datasets": {}}  # Initialize an empty result
        
        result["status"] = "error"
        timestamp_end = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Zillow data fetch process failed at {timestamp_end}.")
        duration = time.time() - time.mktime(time.strptime(timestamp_start, "%Y%m%d-%H%M%S"))
        logger.info(f"Duration: {duration} seconds.")
        
        metadata_config = config.get("metadata_config", {})
        
        timestamps = {"start": timestamp_start, "end": timestamp_end, "duration": duration}
        
        result["metadata"] = get_metadata(metadata_config, result["status"], timestamps)
        result["metadata"]["file_path"] = config.get("save_to_json_path", "") + "/" + \
                                        config.get("save_to_json_file", "") + \
                                        timestamps["start"] + ".json"
        logger.info(f"Metadata collected:{result['metadata']}")
        logger.info(f"Analysis started...")
        result = analyze_and_log_zillow_data(result)
        logger.info("Analysis completed.")
        if config["save_to_json"]:
            logger.info("Saving results to JSON...")
            try:
                output_file_path = config["save_to_json_path"] 
                output_file_name = config["save_to_json_file"] + timestamps["start"] + ".json"  # Path to save the file in the data folder
                save_results_to_json(result, output_file_name, output_file_path)
            except Exception as save_error:
                logger.error(f"Error saving results to JSON: {save_error}")
        
        return result

    finally:
        driver.quit()
        logger.info("Selenium WebDriver closed.")

def process_zillow_dataset(dataset_name, datasets, glob_statistics, data_path):
    """
    Processes a Zillow dataset by parsing and processing its components.

    Args:
        dataset_name (str): Name of the dataset.
        datasets (list): List of dataset configurations.
        glob_statistics (dict): Global statistics dictionary.
        data_path (str): Path to save downloaded files.

    Returns:
        generator: Yields the processed data for each geography.

    Example:
        ```python
        for result in process_zillow_dataset("example_dataset", datasets, {}, "data/"):
            print(result)
        ```
    """
    logger.info(f"Datasets: {datasets}")
    for dataset in datasets:
        logger.info(f"Dataset: {dataset}")
        dataset_type = dataset["type"]
        logger.info(f"Dataset type: {dataset_type}")
        dataset_data_type = dataset["data_type"]
        logger.info(f"Dataset data type: {dataset_data_type}")
        dataset_housing_types = dataset["housing_types"]
        logger.info(f"Dataset housing types: {dataset_housing_types}")
        for housing_type in dataset_housing_types:
            logger.info(f"Housing type: {housing_type['housing_type']}")
            housing_type_features = housing_type["features"]
            logger.info(f"Housing type features: {housing_type_features}")
            for feature in housing_type_features:
                logger.info(f"Feature: {feature}")
                feature_geography = feature["geography"]
                logger.info(f"Feature geography: {feature_geography}")
                for geography in feature_geography:
                    yield process_geography(geography, feature, dataset_name, glob_statistics, data_path)
                    
    
def process_zillow_datasets(zillow_datasets, db_file = "zillow_datasets.db"):
    # Инициализация глобальной статистики
    glob_statistics = {
        "unique_features_processed": set(),
        "features_with_filename": 0,
        "files_downloaded": 0,
        "dataframes_loaded": 0,
        "instances_parsed": 0,
        "successful_db_writes": 0,
        "errors": {}
    }
    result_dict = {"data":{}}
    data_path = os.getenv("DB_PATH")
    ensure_directory_exists(data_path)  # Ensure path exists
    logger.info(f"Processing Zillow datasets with data path: {data_path}")
    
    conn = sqlite3.connect(data_path+db_file)
    cursor = conn.cursor()
    assert sanitize_filename("Test:File/Name") == "Test_File_Name", "Filename sanitization test failed"
    # Create table of tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_tables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT UNIQUE NOT NULL,
        table_name TEXT UNIQUE NOT NULL,
        file_path TEXT NOT NULL
    )
    """)
    conn.commit()
    db_manager = DatabaseManager(db_file)
    for dataset_name, datasets in zillow_datasets['zillow_datasets'].items():
        try:
            dataset_name, dataframe, table_name, file_path = process_zillow_dataset(dataset_name, datasets, glob_statistics, data_path)
            assert isinstance(dataframe, pd.DataFrame), "Loaded data is not a DataFrame"
            result_dict["data"][dataset_name] = {"dataframe":dataframe, "table_name":table_name, "file_path":file_path}
            update_statistics(glob_statistics, "instances_parsed")
            DatabaseManager.save_dataframe_to_db(dataframe, table_name, conn)
            # Add entry to dataset_tables
            cursor.execute("""
                INSERT OR IGNORE INTO dataset_tables (dataset_name, table_name, file_path)
                VALUES (?, ?, ?)
                """, (str(dataset_name), str(table_name), str(file_path)))
            conn.commit()
        except Exception as e:
                    logger.error(f"Error processing dataset {dataset_name}, error: {e}")
    stats_dict = {
        "unique_features_processed": len(glob_statistics["unique_features_processed"]),
        "features_with_filename": glob_statistics["features_with_filename"],
        "files_downloaded": glob_statistics["files_downloaded"],
        "dataframes_loaded": glob_statistics["dataframes_loaded"],
        "instances_parsed": glob_statistics["instances_parsed"],
        "successful_db_writes": glob_statistics["successful_db_writes"],
        "error_counts": str(glob_statistics["errors"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    result_dict["statistics"] = stats_dict
    conn.close()
    logger.info("Processing complete.")
    # Save stats to DB
    conn = sqlite3.connect("data/zillow_datasets.db")
    stats_df = pd.DataFrame([stats_dict])
    stats_df.to_sql("processing_statistics", conn, if_exists="replace", index=False)
    logger.info(f"Saved statistics to database: {stats_df.to_dict(orient='records')}")

    # Close connection
    conn.close()
    return result_dict
