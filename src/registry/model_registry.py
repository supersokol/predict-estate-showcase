from typing import Callable, Dict
from src.core.logger import logger

class ModelRegistry:
    def __init__(self):
        self.registry: Dict[str, Dict] = {}

    def register(self, name: str, metadata: Dict):
        """
        Decorator to register a model with its training function and metadata.
        :param name: Name of the model.
        :param metadata: Metadata describing the model.
        """
        def decorator(func: Callable):
            if name in self.registry:
                logger.warning(f"Model {name} is already registered. Overwriting.")
            self.registry[name] = {"model_func": func, "metadata": metadata}
            return func
        return decorator

    def get_model(self, name: str):
        """
        Retrieve a registered model.
        :param name: Name of the model.
        :return: Dictionary with model function and metadata.
        """
        return self.registry.get(name, None)

    def list_models(self):
        """
        List all registered models.
        """
        return {name: meta["metadata"] for name, meta in self.registry.items()}
