from .process_registry import PROCESS_REGISTRY
from .pipeline_registry import pipeline_registry
from .model_registry import ModelRegistry
from .metaheuristics_registry import MetaHeuristicsRegistry
from .data_source_registry import DataSourceRegistry
from .config_registry import ConfigRegistry

process_registry = PROCESS_REGISTRY
model_registry = ModelRegistry()
metaheuristics_registry = MetaHeuristicsRegistry
data_source_registry = DataSourceRegistry()
config_registry = ConfigRegistry


#data_source_registry.update_registry()
#data_source_registry.display_table()

#from .pipeline_registry import PipelineExecutor
#pipeline_registry.register_pipeline("example_pipeline", "pipelines/example_pipeline.json")
#executor = PipelineExecutor(pipeline_registry.get_pipeline("example_pipeline"))
#executor.execute(df)