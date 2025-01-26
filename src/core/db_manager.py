import sqlite3
from src.core.logger import logger
from src.core.metadata_utils import update_statistics
import pandas as pd

class DatabaseManager:
    def __init__(self, db_file: str):
        self.db_file = db_file
    
    def initialize_db(self):
        """
        Initializes the database with required tables.
        Creates tables if they do not exist.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            # Create a table to track datasets and their metadata
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT UNIQUE NOT NULL,
                table_name TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create a table for logging errors or events
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Additional tables can be added here
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        finally:
            conn.commit()
            conn.close()
    def save_dataframe_to_db(self, df, table_name, conn, glob_statistics):
        """
        Saves a dataframe to a SQLite database.

        :param df: DataFrame to save.
        :param table_name: Table name in the database.
        :param conn: SQLite connection object.
        """
        try:
            # Validate dataframe
            if df.empty:
                logger.warning(f"DataFrame is empty. Skipping table: {table_name}")
                return
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Data is not a DataFrame. Skipping table: {table_name}")
                update_statistics(glob_statistics, "errors", error_type="invalid_dataframe")
                return
        
            # Save to database
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            update_statistics(glob_statistics, "successful_db_writes")
            logger.info(f"Saved DataFrame to table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to table: {table_name}, error: {e}")
            update_statistics(glob_statistics, "errors", error_type="db_write_failure")

    def show_db_contents(self):
        """
        Shows the contents of the SQLite database: datasets and tables.

        :return: None
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Show all tables
        logger.info("Tables in the database:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            print(f"Table: {table[0]}")
    
        # Show dataset tables
        if "dataset_tables" in [t[0] for t in tables]:
            logger.info("Dataset table mappings:")
            cursor.execute("SELECT * FROM dataset_tables;")
            rows = cursor.fetchall()
            for row in rows:
                print(f"Dataset: {row[1]}, Table: {row[2]}, File Path: {row[3]}")
    
        conn.close()

    def get_table_from_db(self, table_name):
        """
        Retrieves a specific table from the database.

        :param table_name: Name of the table to retrieve.
        :return: DataFrame of the table.
        """
        conn = sqlite3.connect(self.db_file)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            logger.info(f"Retrieved table: {table_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve table: {table_name}, error: {e}")
            return None
        finally:
            conn.close()
