# Module: file_utils

This module provides utility functions for working with files.

## Key Functions
- **load_data**: Loads data from a file based on its type.
- **save_results_to_json**: Saves data as a JSON file.

## Function: load_data
```python
def load_data(file_path, file_type="csv", load_as="dataframe"):
    """
    Loads data from the specified file path.

    Args:
        file_path (str): Path to the file.
        file_type (str): Type of the file (csv, json, etc.).
        load_as (str): Format to load the data into.

    Returns:
        object: Loaded data (e.g., DataFrame or dict).
    """
```
```

---
