import importlib
import json

#CONFIG_PATH = "config/sections_config.json"
#ELEMENTS_CONFIG_PATH = "config/elements_config.json"

def load_sections(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    sections = {}
    for name, module_path in config["sections"].items():
        module = importlib.import_module(module_path)
        sections[name] = module
    return sections

def load_elements(ELEMENTS_CONFIG_PATH):
    with open(ELEMENTS_CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    elements = {"nodes": {}, "edges": {}}
    for name, params in config.get("nodes", {}).items():
        module = importlib.import_module(params["ui"])
        elements["nodes"][name] = module
    elements["edges"] = config.get("edges", {})
    return elements
