from src.core.logger import logger
import os
import json
import re
import yaml
import pandas as pd
from collections import defaultdict

import csv
import json
import os
from typing import Any, Dict, Optional, Union

import requests
from bs4 import BeautifulSoup
from wikipediaapi import Wikipedia


def get_context(context_type: Optional[str], **kwargs) -> Union[str, Dict[str, Any]]:
    """
    Получает контекст в зависимости от типа источника.

    :param context_type: Тип контекста (text, file, url, wiki).
    :param kwargs: Дополнительные параметры (text, file_path, url, term).
    :return: Контекст в виде строки или структурированных данных.
    """
    try:
        if context_type == None:
            logger.warning("No context type provided. Returning empty context.")
            return None
        elif context_type == "text":
            # Ручной ввод текста
            text = kwargs.get("text", "")
            logger.info(f"Text context received with length: {len(text)}:\n{text}\n\n")
            return text
        elif context_type == "file":
            file_path = kwargs.get("file_path")
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at path: {file_path}")
            logger.info(f"File context received. Path: {file_path}")
            return _process_file(file_path)
        elif context_type == "url":
            # Веб-скрапинг текста со страницы
            url = kwargs.get("url")
            if not url:
                raise ValueError("No URL provided for context extraction.")
            logger.info(f"Fetching content from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Проверка на ошибки HTTP
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text(strip=True)
            logger.info(f"Extracted content from URL with length: {len(content)}:\n{content}\n\n")
    
        elif context_type == "wiki":
            # Получение статьи из Wikipedia
            # Установка пользовательского агента
            wiki = Wikipedia(
                user_agent="MyAIWorkspace/1.0 (myemail@example.com) CustomLangGraphProcessor"
            )
            term = kwargs.get("term")
            if not term:
                raise ValueError("No Wikipedia term provided for context extraction.")
            logger.info(f"Fetching Wikipedia page for term: {term}")
            page = wiki.page(term)
            if page.exists():
                logger.info(f"Wikipedia page found for term '{term}'. Extracting content.")
                return page.text
            else:
                raise ValueError(f"Wikipedia page for term '{term}' not found.")
    
        else:
            raise ValueError(f"Unsupported context type: {context_type}")
    except Exception as e:
        logger.error(f"Error while extracting context: {e}", exc_info=True)
        raise e

def _process_file(file_path: str) -> Union[str, Dict[str, Any]]:
    """
    Обрабатывает файл и возвращает его содержимое.

    :param file_path: Путь к файлу.
    :return: Текстовое содержимое или структурированные данные.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Detected file extension: {file_extension}")

    try:
        if file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Processed .txt file with length: {len(content)}")
            return content

        elif file_extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Processed .json file with keys: {list(data.keys())}")
            return data

        elif file_extension == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = [row for row in reader]
            logger.info(f"Processed .csv file with {len(data)} rows.")
            return data

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    except Exception as e:
        logger.error(f"Error processing file: {file_path}. Error: {e}", exc_info=True)
        raise e





def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If it does not, creates it.

    :param directory: Path to the directory.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directory created: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory: {e}")    
        
def save_results_to_json(results, filename="results.json", directory_path="data"):
    """
    Saves results to a JSON file.

    :param results: List of results to save.
    :param filename: Name of the file to save.
    """
    try:
        ensure_directory_exists(directory_path)
        filename = directory_path+'//' + filename  # Path to save the file in the data folder
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Results successfully saved to file: {filename}")
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")

def clean_text(text):
    """
    Cleans text from unnecessary characters, replaces \n with spaces,
    sequences of spaces with a single space, and trims spaces from the edges.
    """
    if not text:
        return None
    # Remove newline characters and replace sequences of spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def sanitize_filename(filename):
    """
    Sanitizes the filename by removing or replacing invalid characters.
    """
    # Заменяем запрещенные символы на подчеркивание
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def extract_timestamp_from_filename(filename):
    """
    Extracts timestamp from a filename like 'zillow_datasets20250125-051342.json'.

    :param filename: Name of the file.
    :return: Extracted timestamp as a string.
    """
    match = re.search(r'\d{8}-\d{6}', filename)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"No valid timestamp found in filename: {filename}")

def get_files_for_last_timestamps(directory, max_timestamps=3):
    """
    Collects files for the last `max_timestamps` unique timestamps from a directory.

    :param directory: Directory to scan for files.
    :param max_timestamps: Maximum number of unique timestamps to consider.
    :return: List of file paths matching the last `max_timestamps`.
    """
    # Get all files in the directory
    all_files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    logger.info(f"Found {len(all_files)} files in directory: {directory}\nfiles: {all_files}")
    
    # Extract timestamps and group files by their base name
    files_by_base = defaultdict(list)
    for file in sorted(all_files):  # Sort by file name for grouping
        timestamp = extract_timestamp_from_filename(file)
        if timestamp:
            # Use the file name without the timestamp as the base name
            base_name = re.sub(r'_\d{8}-\d{6}', '', file)
            files_by_base[base_name].append((timestamp, file))

    # Collect files for the last `max_timestamps` unique timestamps
    selected_files = []
    for base_name, files in files_by_base.items():
        # Sort files by timestamp (newest first)
        files.sort(key=lambda x: x[0], reverse=True)
        # Take files for the last `max_timestamps` timestamps
        unique_timestamps = set()
        for timestamp, file in files:
            if len(unique_timestamps) < max_timestamps:
                unique_timestamps.add(timestamp)
                selected_files.append(os.path.join(directory, file))
    
    return selected_files

def load_data(file_path, file_type="csv", load_as="dataframe", encoding = "utf-8", label = None):
    logger.info(f"Loading data from file: {file_path}, type: {file_type}, load_as: {load_as}, encoding: {encoding}")
    try:
        if load_as == "dataframe":
            def load_data_as_dataframe(file_path, file_type):
                if file_type == "csv":
                    data = pd.read_csv(file_path)
                elif file_type == "json":
                    data = pd.read_json(file_path)
                else:
                    logger.error(f"Error loading data from file: {file_path}. Unsupported file type: {file_type}")
                    raise ValueError("Unsupported file type")
                if label:
                    logger.info(f"Adding label to data: {label}")
                    data["data_label"] = label
                return data
            data = load_data_as_dataframe(file_path, file_type)
            logger.info(f"{file_type} data loaded from file: {file_path}")
        elif load_as == "json":
            def load_data_as_json(file_path, file_type, encoding):
                if file_type == "json":
                    with open(file_path, "r", encoding=encoding) as f:
                        data = json.load(f)
                elif file_type == "csv":
                    data = pd.read_csv(file_path).to_dict(orient="records")
                elif file_type == "text":
                    with open(file_path, "r", encoding=encoding) as f:
                        data = json.loads(f.read())
                elif file_type == "yml":
                    with open(file_path, "r", encoding=encoding) as f:
                        data = json.dumps(yaml.safe_load(f))
                else:
                    logger.error(f"Error loading data from file: {file_path}. Unsupported file type: {file_type}")
                    raise ValueError("Unsupported file type")
                logger.info(f"JSON data loaded from file: {file_path}")
                return data
            data = load_data_as_json(file_path, file_type, encoding)
            if label:
                    logger.info(f"Adding label to JSON data: {label}")
                    data["data_label"]=label
        elif load_as == "text":
            with open(file_path, "r", encoding=encoding) as f:
                try:
                    data = f.read()
                    logger.info(f"Text data loaded from file: {file_path}")
                    if label:
                        logger.info(f"Adding label to text data: {label}")  
                        data=f"data_label: '{label}'\n\n"+data
                except Exception as e:
                    logger.error(f"Error loading text data from file: {file_path}: {e}")
                    return None
        elif load_as == "list": 
            with open(file_path, "r", encoding=encoding) as f:
                try:
                    data = f.readlines()
                    data = [line.strip() for line in data]
                    logger.info(f"List data loaded from file: {file_path}")
                    if label:
                        logger.info(f"Adding label to list data: {label}")
                        data = [f"data_label: '{label}'"] + data
                except Exception as e:
                    logger.error(f"Error loading list data from file: {file_path}: {e}")
                    return None
        elif load_as == "code":
            if file_type == "python":
                with open(file_path, "r", encoding=encoding) as f:
                    try:
                        data = eval(f.read())
                        logger.info(f"Python data loaded from file: {file_path}")
                        if label:
                            logger.info(f"Adding label to Python data: {label}")
                            data["data_label"] = label
                    except Exception as e:
                        logger.error(f"Error loading Python data from file: {file_path}: {e}")
            elif file_type == "yml":
                with open(file_path, "r", encoding=encoding) as f:
                    try:
                        data = yaml.safe_load(f)
                        logger.info(f"YAML data loaded from file: {file_path}")
                        if label:
                            logger.info(f"Adding label to YAML data: {label}")
                            data["data_label"] = label
                    except Exception as e:
                        logger.error(f"Error loading YAML data from file: {file_path}: {e}")
        else:
            logger.error(f"Error loading data from file: {file_path}. Unsupported load_as type: {load_as}")
            raise ValueError("Unsupported load_as type")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from file: {e}")
        try:
            if load_as == "dataframe":
                return pd.DataFrame()
            elif load_as == "json":
                return {}
            elif load_as == "text":
                return ""
            elif load_as == "list":
                return []
            elif load_as == "code":
                return {}
            else:
                logger.error(f"Unsupported load_as type: {load_as}")
                return None
        except Exception as e:
            logger.error(f"Error handling load_as type: {e}")
            return None
        