from .process_registry import PROCESS_REGISTRY
from .pipeline_registry import PipelineRegistry
from .model_registry import ModelRegistry
from .metaheuristics_registry import MetaHeuristicsRegistry
from .data_source_registry import DataSourceRegistry
from .config_registry import ConfigRegistry

process_registry = PROCESS_REGISTRY
pipeline_registry = PipelineRegistry
model_registry = ModelRegistry()
metaheuristics_registry = MetaHeuristicsRegistry
data_source_registry = DataSourceRegistry
config_registry = ConfigRegistry