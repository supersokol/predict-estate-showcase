# PredictEstate Showcase

This project processes and analyzes Zillow datasets, ensuring efficient storage and management of data.

## Features
- **Dynamic data processing:** Automatically handles Zillow data with timestamp-based management.
- **Efficient file handling:** Keeps only the latest results (up to 3) for better storage management.
- **Interactive Documentation:** Built using MkDocs with a Material theme.

## Installation
```bash
pip install -r requirements.txt
```
## Usafe
```python
from src.registry.data_workflow import manage_data_processing

# Example configuration
config = {
    "data_config_path": "data/zillow_config.json",
    "prefix": "zillow",
    "data_path": "data/zillow/datasets",
}

result = manage_data_processing(config)
print(result)
```

## Documentation
Visit the full documentation [here](https://example.com).
```

## Testing
Run the tests with:
```bash
pytest
```

---

python src/app/run_services.py 