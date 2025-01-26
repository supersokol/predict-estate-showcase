from dotenv import load_dotenv
import time
from core.logger import logger
from core.config_loader import load_all_configs_via_master, check_and_load_config_zillow
from analysis.zillow_analysis import run_analysis
from core.scraper import process_zillow_datasets
from registry.data_workflow import manage_data_processing


# Загрузка переменных окружения
load_dotenv()
logger.info("Environment variables loaded.")

loaded_configs = load_all_configs_via_master()

# Извлекаем конфигурацию для Zillow
zillow_config = next(
            config["content"] 
            for config in loaded_configs["loaded_configs"]["data_sources"] 
            if config["content"]["config_name"] == "zillow_config"
        )

processing_result = manage_data_processing(zillow_config)

## Используем check_and_load_config_zillow для загрузки JSON
#zillow_json_data = check_and_load_config_zillow(zillow_config)

#run_analysis(zillow_json_data )

#DB_FILE = "data/db_files/zillow_datasets.db"

#zillow_datasets = zillow_json_data .copy()
#processing_result = process_zillow_datasets(zillow_datasets)