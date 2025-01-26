# Logging setup
from loguru import logger
import time

timestamp_start = time.strftime("%Y%m%d-%H%M%S")
logger.add("logs//scraper"+timestamp_start+".log", format="{time} {level} {message}", level="DEBUG", rotation="10 MB", compression="zip")
logger.info(f"New log session started at {timestamp_start}. Log file: logs//scraper{timestamp_start}.log")

