from src.core.logger import logger

class PipelineError(Exception):
    """Base class for pipeline-related errors."""
    pass

class ProcessError(PipelineError):
    """Error raised during the execution of a process."""
    def __init__(self, process_name, message="Error in process execution"):
        super().__init__(f"{message}: {process_name}")

class DataError(PipelineError):
    """Error raised for data-related issues."""
    def __init__(self, message="Invalid data"):
        super().__init__(message)


def handle_error(error, pipeline_name=None, process_name=None):
    """
    Handles and logs errors during pipeline or process execution.
    """
    if isinstance(error, ProcessError):
        logger.error(f"Process error in {process_name}: {error}")
    elif isinstance(error, DataError):
        logger.error(f"Data error: {error}")
    else:
        logger.error(f"Unknown error in pipeline {pipeline_name}: {error}")
