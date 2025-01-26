import sys
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

from src.processes.data_processing import *
from src.processes.basic_eda import *
from src.processes.report_generation import *
from src.processes.model_processes import *

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


app.mount("/mkdocs", StaticFiles(directory="site"), name="mkdocs")

@app.get("/", include_in_schema=False)
async def root():
    return {
        "swagger_ui": "/docs",
        "mkdocs_ui": "/mkdocs/index.html"
    }

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


data_registry = data_source_registry
class AddFilesRequest(BaseModel):
    source_name: str
    new_files: List[str]

@app.post("/data/files/add", tags=["Data Sources"])
async def add_files_to_source(request: AddFilesRequest):
    """
    Add new files to an existing data source.
    """
    try:
        data_registry.add_to_uploaded(request.source_name, request.new_files)
        return {
            "status": "success",
            "message": f"Files added to the 'uploaded' section of data source '{request.source_name}'.",
            "uploaded": data_registry.get_source(request.source_name)["local_files"]["uploaded"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
class DataSourceConfig(BaseModel):
    name: str  
    description: str
    #zillow_data_labels_target: Optional[List[str]]


@app.post("/data/register", tags=["Data Sources"])
async def register_data_source(user_config: DataSourceConfig, **kwargs):
    """
    Register a new data source and download its datasets.
    """
    try:
        loaded_configs = load_all_configs_via_master()

        if user_config["data_type"] == "zillow_datasets":
            zillow_config = next(
                    config["content"] 
                    for config in loaded_configs["loaded_configs"]["data_sources"] 
                    if config["content"]["config_name"] == "zillow_config"
                )
            
            target_labels = kwargs.get("zillow_data_labels_target", None)
            if target_labels:
                zillow_config["zillow_data_labels_target"]=target_labels
            chosen_config = zillow_config
        local_files = manage_data_processing(data_registry, chosen_config)
        data_registry.add_to_datasets(chosen_config.name, local_files)
        return {
            "status": "success",
            "message": f"Data source '{chosen_config.name}' registered successfully.",
            "datasets": local_files,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/data/files/{source_name}", tags=["Data Sources"])
async def get_data_files(source_name: str):
    """
    Retrieve local files for a registered data source.
    """
    source = data_registry.get_source(source_name)
    if not source:
        raise HTTPException(status_code=404, detail=f"Data source '{source_name}' not found.")
    return {"local_files": source.get("local_files", [])}


@app.get("/data/sample/{source_name}", tags=["Data Sources"])
async def get_data_sample(source_name: str, rows: int = 5):
    """
    Get a preview (sample) of a dataset.
    """
    source = data_registry.get_source(source_name)
    if not source:
        raise HTTPException(status_code=404, detail=f"Data source '{source_name}' not found.")
    
    local_files = source.get("local_files", [])
    if not local_files:
        raise HTTPException(status_code=404, detail=f"No local files found for '{source_name}'.")

    # Загружаем файл и возвращаем образец
    import pandas as pd
    file_path = local_files[0]  # Возьмем первый файл
    df = pd.read_csv(file_path)
    return {"sample": df.head(rows).to_dict(orient="records")}

#######

# Helper functions for registry execution
def get_registry_metadata(registry):
    return [
        {
            "name": name,
            "description": meta.get("metadata", {}).get("description", "No description."),
            "parameters": meta.get("metadata", {}).get("parameters", {}),
        }
        for name, meta in registry.items()
    ]

# Endpoints for Processes
# Input model for processes
class ProcessInput(BaseModel):
    parameters: dict

@app.get("/processes", tags=["Processes"], summary="List all registered processes")
async def list_processes():
    """
    Returns a list of all processes registered in the system.
    Processes are reusable functions designed for data manipulation or analysis.
    Each process comes with detailed metadata, including a description, parameters, and expected outputs.
    """
    return list(process_registry.keys())

@app.post("/processes/{process_name}", tags=["Processes"], summary="Execute a process")
async def execute_process(process_name: str, input_data: ProcessInput):
    """
    Execute a registered process by its name.
    Pass parameters specific to the process as input.
    Example use cases:
    - Normalizing numeric columns.
    - Removing duplicates from a dataset.
    - Generating correlation matrices.
    """
    if process_name not in process_registry.registry:
        raise HTTPException(status_code=404, detail=f"Process {process_name} not found.")
    process_func = process_registry.registry[process_name]["func"]
    try:
        result = process_func(**input_data.parameters)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints for Pipelines
# Input model for pipelines
class PipelineInput(BaseModel):
    data: dict

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
        from core.pipeline_executor import PipelineExecutor
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