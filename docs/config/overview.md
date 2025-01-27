# Configurations Overview

This project heavily relies on various configuration files to manage its pipelines, settings, and processes. Each configuration serves a specific purpose, from defining dataset paths to orchestrating machine learning pipelines.

## Key Configuration Files

1. **Zillow Config (`zillow_config.json`)**:
    - Manages the structure and logic of Zillow dataset handling.
    - Contains file paths and metadata rules.

2. **Zillow Data Config (`zillow_data_config.json`)**  
    - Includes details about how to structure and process Zillow data.
    - Handles metadata, column mappings, and versioning.

3. **Master Config (`master_config.json`)**  
    - Centralizes all major configurations for the project.
    - Points to specific configuration files for pipelines, datasets, and sections.

4. **Pipeline Config (`pipeline_config.json`)**  
    - Defines the workflows for data preprocessing, model training, and evaluation.

5. **Sections Config (`sections_config.json`)**  
    - Handles configuration for UI sections and streamlit dashboards.

---

## Loading Configurations

To load any configuration file, use the `load_data` function from the `file_utils` module:

```python
from core.file_utils import load_data

config_path = "path/to/config.json"
config = load_data(config_path, file_type="json", load_as="json")
print(config)
```

---

## Best Practices for Managing Configurations

- **Consistency**: Always use clear and descriptive keys for configuration fields.
- **Version Control**: Store all configuration files in version control (e.g., Git).
- **Validation**: Validate configurations before passing them into pipelines.
- **Documentation**: Document the purpose of each configuration file and its fields.


---