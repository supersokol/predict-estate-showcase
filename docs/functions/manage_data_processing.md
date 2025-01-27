# Function: manage_data_processing

`manage_data_processing` handles the validation, processing, and storage of Zillow datasets. It ensures that up-to-date data is processed only when necessary and manages up to three saved results.

## Arguments
- `config` (*dict*): A dictionary containing configuration details for Zillow processing.

## Returns
- *dict*: Processed result or loaded result from existing files.

## Example
```python
from src.registry.data_workflow import manage_data_processing

config = {
    "prefix": "zillow",
    "data_config_path": "data/zillow_config.json",
    "data_path": "data/zillow/datasets"
}

result = manage_data_processing(config)
```

## Workflow
1. Loads Zillow configuration using `check_and_load_config_zillow`.
2. Extracts the timestamp from the JSON configuration file.
3. Constructs the `data_path` using the timestamp.
4. Checks for existing files in the directory.
5. If no valid files are found, processes the datasets using `process_zillow_datasets`.
6. Saves the processed result and manages old files (keeps up to 3 unique timestamps).
