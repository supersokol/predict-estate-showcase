import os
import json
import sqlite3
import csv
from datetime import datetime
from typing import List, Dict, Optional
from src.core.logger import logger

class DataSourceRegistry:
    """
    Registry for managing data sources.

    The `DataSourceRegistry` class provides functionality to initialize, manage,
    and retrieve data source records from a SQLite database.

    Attributes:
        db_path (str): Path to the SQLite database file.
    """
    def __init__(self, db_file="data_sources.db"):
        """
        Initialize the DataSourceRegistry and set up the database.

        Args:
            db_file (str, optional): Name of the SQLite database file. Defaults to "data_sources.db".

        Example:
            ```python
            registry = DataSourceRegistry("my_data_sources.db")
            ```
        """
        self.db_path = os.getenv("DB_PATH") + db_file
        self._initialize_database()

    def _initialize_database(self):
        """
        Create the database table if it doesn't already exist.

        This method ensures that the `data_sources` table is present in the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                format TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _determine_file_type(self, file_path: str) -> str:
        """
        Determine the file type based on the path and extension.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The determined file type, e.g., "dataset", "config", or "uploaded".

        Example:
            ```python
            file_type = registry._determine_file_type("data/sample.csv")
            print(file_type)  # Output: "dataset"
            ```
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if "config" in file_path.lower():
            return "config"
        elif "data" in file_path.lower():
            if ext == ".csv":
                return "dataset"
            elif ext == ".json":
                return "config"
            else:
                return "uploaded"
        else:
            return "uploaded"

    def _get_file_metadata(self, file_path: str) -> Dict[str, str]:
        """
    Get metadata for a file, including size, last modified timestamp,
    number of characters, number of lines, and number of columns.

    Args:
        file_path (str): Path to the file.

    Returns:
        dict: Metadata including size in bytes, last modified timestamp,
            number of characters, number of lines, and number of columns.

    Example:
        ```python
        metadata = registry._get_file_metadata("data/sample.csv")
        print(metadata)
        ```
        """
        metadata = {
            "size_bytes": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
        }
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                metadata["num_chars"] = len(content)
                metadata["num_lines"] = content.count("\n") + 1

                # Определяем количество столбцов для CSV файлов
                if file_path.lower().endswith(".csv"):
                    file.seek(0)  # Возвращаемся в начало файла
                    reader = csv.reader(file)
                    first_row = next(reader, [])
                    metadata["num_columns"] = len(first_row)
                else:
                    metadata["num_columns"] = 1
                metadata["status"] = "success" 
                metadata["error"] = None
        except Exception as e:
            # Обработка ошибок чтения файлов
            metadata["num_chars"] = 0
            metadata["num_lines"] = 0
            metadata["num_columns"] = 1
            metadata["status"] = "error"
            metadata["error"] = str(e)
        return metadata

    def update_registry(self):
        """
        Update the database with files found in predefined directories.

        This method scans `data` and `configs` directories, determines file types,
        and updates the database with new or updated records.
        """
        files_to_register = self._get_all_files()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for file_path in files_to_register:
            file_type = self._determine_file_type(file_path)
            _, ext = os.path.splitext(file_path)
            ext = ext.lstrip(".").lower()
            timestamp = datetime.now().isoformat()
            metadata = json.dumps(self._get_file_metadata(file_path))

            # Check for the presence of a record in the database
            cursor.execute("""
                SELECT id FROM data_sources WHERE file_path = ?
            """, (file_path,))
            if not cursor.fetchone():
                # If the record is not present, add it
                cursor.execute("""
                    INSERT INTO data_sources (file_path, file_type, format, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, file_type, ext, timestamp, metadata))

        conn.commit()
        conn.close()

    def _get_all_files(self) -> List[str]:
        """
        Collect all file paths in the `data` and `configs` directories.

        Returns:
            list: A list of file paths.

        Example:
            ```python
            files = registry._get_all_files()
            print(files)
            ```
        """
        root_dirs = ["data", "configs"]
        supported_formats = {".csv", ".txt", ".md", ".pdf", ".json"}
        files = []

        for root_dir in root_dirs:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in supported_formats:
                        files.append(os.path.join(dirpath, filename))

        return files

    def get_table(self) -> List[tuple]:
        """
        Retrieve all records from the `data_sources` table.

        Returns:
            list[tuple]: A list of tuples representing rows in the table.

        Example:
            ```python
            table = registry.get_table()
            for row in table:
                print(row)
            ```
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM data_sources
        """)
        rows = cursor.fetchall()
        conn.close()

        return rows

    def display_table(self):
        """
        Display the `data_sources` table in a readable format.

        Returns:
            str: A formatted string of table data.

        Example:
            ```python
            print(registry.display_table())
            ```
        """
        rows = self.get_table()
        if not rows:
            logger.warning("The table is empty.")
            return

        s = f"{'ID':<5} {'File Path':<50} {'Type':<15} {'Format':<10} {'Timestamp':<25} {'Metadata'}\n"
        s += "-" * 120
        for row in rows:
            s+=f"\n{row[0]:<5} {row[1]:<50} {row[2]:<15} {row[3]:<10} {row[4]:<25} {row[5]}"
        logger.info(s)
        return s
