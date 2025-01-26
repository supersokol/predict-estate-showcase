from src.core.logger import logger

class PipelineError(Exception):
    """
    Base class for pipeline-related errors.

    This is the root exception for all pipeline-related issues, allowing specific
    exceptions to inherit from it for better categorization and handling.
    """
    pass

class ProcessError(PipelineError):
    """
    Error raised during the execution of a process.

    Attributes:
        process_name (str): Name of the process where the error occurred.
        message (str): Error message.
    """
    def __init__(self, process_name, message="Error in process execution"):
        """
        Initialize the ProcessError exception.

        Args:
            process_name (str): Name of the process causing the error.
            message (str, optional): Custom error message. Defaults to "Error in process execution".

        Example:
            ```python
            raise ProcessError("DataNormalization", "Normalization failed")
            ```
        """
        super().__init__(f"{message}: {process_name}")

class DataError(PipelineError):
    """
    Error raised for data-related issues.

    Attributes:
        message (str): Error message.
    """
    def __init__(self, message="Invalid data"):
        """
        Initialize the DataError exception.

        Args:
            message (str, optional): Custom error message. Defaults to "Invalid data".

        Example:
            ```python
            raise DataError("Missing required columns")
            ```
        """
        super().__init__(message)


def handle_error(error, pipeline_name=None, process_name=None):
    """
    Handles and logs errors during pipeline or process execution.

    This function categorizes errors into known types and logs them appropriately.
    If the error type is unknown, it logs it as a generic pipeline error.

    Args:
        error (Exception): The error object to handle.
        pipeline_name (str, optional): Name of the pipeline where the error occurred.
        process_name (str, optional): Name of the process causing the error.

    Returns:
        None

    Example:
        ```python
        try:
            some_pipeline.execute()
        except Exception as e:
            handle_error(e, pipeline_name="DataPipeline", process_name="NormalizationProcess")
        ```
    """
    if isinstance(error, ProcessError):
        logger.error(f"Process error in {process_name}: {error}")
    elif isinstance(error, DataError):
        logger.error(f"Data error: {error}")
    else:
        logger.error(f"Unknown error in pipeline {pipeline_name}: {error}")
