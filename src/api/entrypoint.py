import sys
import json
sys.path.append("Q:/SANDBOX/PredictEstateShowcase_dev/src")
sys.path.append("Q:/SANDBOX/PredictEstateShowcase_dev/")
print(sys.path)
from core.config_loader import load_all_configs_via_master
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union, List
from src.registry import (
    process_registry,
    pipeline_registry,
    model_registry,
    metaheuristics_registry,
    data_source_registry
)
from src.registry.data_workflow import manage_data_processing

from src.workflows.data_processing import *
from src.workflows.basic_eda import *
from src.workflows.report_generation import *
from src.workflows.model_processes import *

from src.models.regression_models import *
#from src.models.genetic_algorithms import *
#from src.models.econometric import *

#from src.metaheuristics.metaheuristics_examples import *

app = FastAPI(
    title="Intelligent Automation API",
    description="""
This API provides a flexible and extensible interface for managing and executing processes, pipelines, models, and metaheuristics.
Use it to:
- Automate data processing workflows.
- Train and use machine learning models.
- Execute high-level metaheuristics to dynamically generate new pipelines.
- Manage data files and configurations.

Each section below details the available endpoints and their functionality.
Swagger combined with MkDocs for comprehensive API documentation:
- [API Documentation](http://127.0.0.1:8000/docs)
- [Detailed User Guide](http://127.0.0.1:8000/mkdocs/index.html)
- [Streamlit Demo](http://localhost:8501/)
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Mount static files for MkDocs
app.mount("/mkdocs", StaticFiles(directory="site"), name="mkdocs")

@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint providing navigation links to documentation.

    Returns:
        dict: Links to Swagger UI and MkDocs documentation.

    Example:
        ```json
        {
            "swagger_ui": "/docs",
            "mkdocs_ui": "/mkdocs/index.html"
        }
        ```
    """
    return {
        "swagger_ui": "/docs",
        "mkdocs_ui": "/mkdocs/index.html"
    }

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

# Define a data registry for managing data sources
data_registry = data_source_registry

# Record model for file metadata
class FileRecord(BaseModel):
    """
    Represents a record for obtaining file metadata.

    Attributes:
        id (int): Unique identifier of the file record.
        file_path (str): Path to the file.
        file_type (str): Type of the file (e.g., "csv", "json").
        format (str): Format or structure of the file (e.g., "tabular", "raw").
        timestamp (str): Timestamp of when the file was processed or created.
        metadata (dict): Additional metadata associated with the file.
    """
    id: int
    file_path: str
    file_type: str
    format: str
    timestamp: str
    metadata: dict

@app.put("/data_sources", response_model=dict)
def update_registry():
    """
    Updates the data registry.

    This endpoint triggers an update to the data registry, ensuring that
    all data sources are synchronized with the current state.

    Returns:
        dict: A success message indicating that the registry was updated.

    Example:
        ```python
        response = client.put("/data_sources")
        print(response.json())  # Output: {"message": "Registry updated successfully."}
        ```
    """
    data_registry.update_registry()
    return {"message": "Registry updated successfully."}

@app.get("/data_sources", response_model=list[FileRecord])
def get_table():
    """
    Retrieves all file records from the data registry.

    This endpoint returns a list of file records stored in the data registry.
    Each record includes metadata such as file type, format, and associated metadata.

    Returns:
        list[FileRecord]: A list of file records.

    Example:
        ```python
        response = client.get("/data_sources")
        print(response.json())  
        # Output: [{"id": 1, "file_path": "data/file1.csv", ...}, ...]
        ```
    """
    rows = data_registry.get_table()
    records = []
    for row in rows:
        records.append({
            "id": row[0],
            "file_path": row[1],
            "file_type": row[2],
            "format": row[3],
            "timestamp": row[4],
            "metadata": json.loads(row[5] if row[5] else "{}"),
        })
    return records

# Helper functions for registry execution
def get_registry_metadata(registry):
    """
    Extracts metadata for all items in the specified registry.

    Args:
        registry (dict): The registry containing items with metadata.

    Returns:
        list[dict]: A list of dictionaries with metadata information.

    Example:
        ```python
        metadata = get_registry_metadata(process_registry.registry)
        print(metadata)
        # Output: [{"name": "Process1", "description": "Processes data", ...}, ...]
        ```
    """
    return [
        {
            "name": name,
            "description": meta.get("metadata", {}).get("description", "No description."),
            "parameters": meta.get("metadata", {}).get("parameters", {}),
        }
        for name, meta in registry.items()
    ]

# Input model for processes:
class ProcessInput(BaseModel):
    """
    Input model for executing a process.

    Attributes:
        parameters (dict): Dictionary of parameters required by the process.
    """
    parameters: dict

# Endpoints for Processes:

@app.get("/processes", tags=["Processes"], summary="List all registered processes")
async def list_processes():
    """
    Lists all processes registered in the system.

    Processes are reusable functions designed for data manipulation or analysis.
    Each process is registered with detailed metadata, including a description,
    required parameters, and expected outputs.

    Returns:
        list[str]: A list of process names registered in the system.

    Example:
        ```python
        response = client.get("/processes")
        print(response.json())  
        # Output: ["Process1", "Process2", ...]
        ```
    """
    return list(process_registry.keys())

@app.post("/processes/{process_name}", tags=["Processes"], summary="Execute a process")
async def execute_process(process_name: str, input_data: ProcessInput):
    """
    Executes a registered process by its name.

    This endpoint allows dynamic execution of a registered process by passing
    the required parameters as input. Processes can perform operations such as:
    - Normalizing numeric columns.
    - Removing duplicates from a dataset.
    - Generating correlation matrices.

    Args:
        process_name (str): The name of the process to execute.
        input_data (ProcessInput): Input parameters for the process.

    Returns:
        dict: A dictionary containing the execution status and result.

    Raises:
        HTTPException: If the process is not found or an error occurs during execution.

    Example:
        ```python
        input_data = {"parameters": {"column": "price"}}
        response = client.post("/processes/NormalizeColumns", json=input_data)
        print(response.json())  
        # Output: {"status": "success", "result": {...}}
        ```
    """
    if process_name not in process_registry.registry:
        raise HTTPException(status_code=404, detail=f"Process {process_name} not found.")
    process_func = process_registry.registry[process_name]["func"]
    try:
        result = process_func(**input_data.parameters)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Input model for pipelines:
class PipelineInput(BaseModel):
    data: dict
    
# Endpoints for Pipelines:

@app.get("/pipelines", tags=["Pipelines"], summary="List all registered pipelines")
async def list_pipelines():
    """
    Returns a list of all pipelines registered in the system.
    Pipelines are predefined workflows consisting of sequential steps (processes, models, or nested pipelines).
    Use pipelines to automate repetitive tasks such as:
    - Data cleaning.
    - Feature engineering.
    - Complex multi-step workflows.
    """
    return get_registry_metadata(pipeline_registry.registry)

@app.post("/pipelines/{pipeline_name}", tags=["Pipelines"], summary="Execute a pipeline")
async def execute_pipeline(pipeline_name: str, input_data: PipelineInput):
    """
    Execute a registered pipeline by its name.
    Pass the dataset to be processed as input.
    Pipelines dynamically execute their steps, which may include processes, models, and nested pipelines.
    """
    if pipeline_name not in pipeline_registry.registry:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_name} not found.")
    pipeline_config = pipeline_registry.registry[pipeline_name]
    try:
        from registry.pipeline_registry import PipelineExecutor
        executor = PipelineExecutor(pipeline_config)
        result = executor.execute(input_data.data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''
# Endpoints for Models

# Input model for model training and inference
class ModelInput(BaseModel):
    data: dict
    parameters: dict

@app.get("/models", tags=["Models"], summary="List all registered models")
async def list_models():
    """
    Returns a list of all models registered in the system.
    Models can be used for:
    - Training: Fit a model to a given dataset.
    - Inference: Make predictions using a trained model.
    Each model comes with metadata that describes its type, parameters, and usage.
    """
    return get_registry_metadata(model_registry.registry)

@app.post("/models/{model_name}/train", tags=["Models"], summary="Train a model")
async def train_model(model_name: str, input_data: ModelInput):
    """
    Train a registered model using the provided dataset and parameters.
    Example use cases:
    - Train a linear regression model on numeric data.
    - Fit a logistic regression model for classification tasks.
    """
    if model_name not in model_registry.registry:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
    model_func = model_registry.registry[model_name]["func"]
    try:
        model = model_func(input_data.data, **input_data.parameters)
        return {"status": "success", "model": str(model)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints for Metaheuristics
@app.get("/metaheuristics", tags=["Metaheuristics"], summary="List all registered metaheuristics")
async def list_metaheuristics():
    """
    Returns a list of all metaheuristics registered in the system.
    Metaheuristics are high-level strategies designed to dynamically:
    - Generate new pipelines based on data analysis.
    - Optimize workflows and suggest improvements.
    Each metaheuristic is described in detail with its usage and logic.
    """
    return get_registry_metadata(metaheuristics_registry.registry)

@app.post("/metaheuristics/{heuristic_name}", tags=["Metaheuristics"], summary="Execute a metaheuristic")
async def execute_metaheuristic(heuristic_name: str, input_data: dict):
    """
    Execute a registered metaheuristic by its name.
    Metaheuristics can analyze data, generate new pipelines, and produce high-level insights.
    """
    if heuristic_name not in metaheuristics_registry.registry:
        raise HTTPException(status_code=404, detail=f"Metaheuristic {heuristic_name} not found.")
    heuristic_func = metaheuristics_registry.registry[heuristic_name]["func"]
    try:
        result = heuristic_func(input_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''