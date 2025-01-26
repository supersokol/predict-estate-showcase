import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from src.core.logger import logger

class DataSourceRegistry:
    def __init__(self, db_file="data_sources.db"):
        self.db_path = os.getenv("DB_PATH") + db_file
        self._initialize_database()

    def _initialize_database(self):
        """Create the table in the database upon initialization."""
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
        """Determine the file type based on the path and extension.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The determined file type.
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
        """Get file metadata (e.g., size).

        Args:
            file_path (str): The path to the file.

        Returns:
            Dict[str, str]: The metadata of the file.
        """
        metadata = {
            "size_bytes": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
        }
        return metadata

    def update_registry(self):
        """Update the database records based on the file structure."""
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
        """Collect paths to all files in the data and configs folders.

        Returns:
            List[str]: A list of file paths.
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
        """Get the table of all files from the database.

        Returns:
            List[tuple]: A list of tuples representing the rows in the table.
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
        """Display the table in a readable format."""
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
