```python
# Import the main function for data processing
from src.registry.data_workflow import manage_data_processing

# Configuration settings for data processing
config = {
    "prefix": "zillow",  # Prefix for the dataset
    "data_config_path": "../data/zillow_config.json",  # Path to the data configuration file
    "data_path": "../data/zillow_datasets"  # Path to the directory containing the datasets
}

# Execute the data processing workflow
result = manage_data_processing(config)

# Print the results of the data processing
print("Data processing results:")
print(result)

```
