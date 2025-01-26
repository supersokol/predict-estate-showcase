import os
import json
from typing import List, Dict, Optional

class DataSourceRegistry:
    def __init__(self, registry_file: str = "data_sources.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=4)

    def register_source(self, source_name: str, metadata: Dict):
        if source_name in self.registry:
            raise ValueError(f"Data source '{source_name}' is already registered.")
        metadata["local_files"] = {"datasets": [], "uploaded": []}
        self.registry[source_name] = metadata
        self.save_registry()

    def update_source(self, source_name: str, metadata: Dict):
        if source_name not in self.registry:
            raise ValueError(f"Data source '{source_name}' is not registered.")
        self.registry[source_name].update(metadata)
        self.save_registry()

    def add_to_datasets(self, source_name: str, files: List[str]):
        if source_name not in self.registry:
            raise ValueError(f"Data source '{source_name}' is not registered.")
        self.registry[source_name]["local_files"]["datasets"].extend(files)
        self.save_registry()

    def add_to_uploaded(self, source_name: str, files: List[str]):
        if source_name not in self.registry:
            raise ValueError(f"Data source '{source_name}' is not registered.")
        self.registry[source_name]["local_files"]["uploaded"].extend(files)
        self.save_registry()
    
    def get_source(self, source_name: str) -> Optional[Dict]:
        return self.registry.get(source_name)

    def list_sources(self) -> List[str]:
        return list(self.registry.keys())

