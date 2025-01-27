# Zillow Config

The `zillow_config.json` file is the main configuration file for handling Zillow datasets. It defines data sources, metadata, paths, and processing rules.

---

## Full JSON Content

Below is the complete JSON content of the configuration file:

```json
{
    "config_name": "zillow_config",
    "description": "Configuration for loading data from the Zillow website",
    "prefix": "zillow",
    "zillow_base_url": "https://www.zillow.com/research/data/",
    "update_frequency": "Monthly",
    "last_updated": "24.01.2025",
    "data_config_path": "config/zillow_data_config.json",
    "data_path": "data//zillow//zillow_datasets",
    "zillow_data_labels_available": [
      "median-home-value-zillow-home-value-index-zhvi-",
      "home-values-forecasts-",
      "rentals-",
      "rental-forecasts-",
      "home-sales-listings-",
      "list-sale-prices-",
      "sales-",
      "market-heat-index-",
      "new-construction-",
      "affordability-"
    ],
    "zillow_data_labels_target": [
        "list-sale-prices-",
        "sales-",
        "market-heat-index-",
        ...
      ],  
    "save_to_json": true,
    "save_to_json_path": "data//zillow//zillow_datasets",
    "save_to_json_file": "zillow_datasets",
    "geography_options": [["U.S."],["Metro & U.S.","Metro & US"], ["State"], ["Country"], ["City"], ["ZIP Code"], ["Neighborhood"]],
    "datasets": {
      "zillow_home_values": {
        "type_pattern": "(Mortgage Payment|Total Monthly Payment|ZHVI)",
        "data_type_pattern": "Time Series",
        "housing_types": [
          "All Homes", "Single-Family Homes", "Condo[\/]Co-op", "1-Bedroom",
          "2-Bedroom", "3-Bedroom", "4-Bedroom", "5[+]-Bedroom"
        ],
        "feature_patterns": {
          "down_payment": "(\\d+%) down",
          "tier": "(Bottom Tier|Top Tier|Mid-Tier)",
          "measure": [
                {"key": "raw", "condition": "Raw"},
                {"key": "smoothed, seasonally adjusted", "condition": "Smoothed.*Seasonally Adjusted"}
            ]
        }
      },
      ...
    },
    "metadata_config" : {
        "timestamps": [{"timestamp_name":"start","timestamp_type":"string"},
                       {"timestamp_name":"end","timestamp_type":"string"},
                       {"timestamp_name":"duration","timestamp_type":"string"},
                       {"timestamp_name":"month","timestamp_type":"date"}],
        "version": "0.1.0",
        "save_to_json": true,
        "save_to_json_path": "data//zillow",
        "save_to_json_file": "zillow_datasets"
        
    }
}

```

For the full JSON file, refer to [zillow_config.json](api/zillow_config.json).

---
## Key Sections

### 1. General Information
- **`config_name`**: `"zillow_config"`  
  A unique identifier for this configuration file.

- **`description`**: `"Configuration for loading data from the Zillow website"`  
  Brief description of the configuration.

- **`prefix`**: `"zillow"`  
  A short prefix used in file naming and logging.

- **`zillow_base_url`**: `"https://www.zillow.com/research/data/"`  
  The base URL for accessing Zillow's datasets.

- **`update_frequency`**: `"Monthly"`  
  Indicates how often the data should be updated.

- **`last_updated`**: `"24.01.2025"`  
  The date when the configuration was last updated.

---

### 2. Paths and Metadata
- **`data_config_path`**: Path to the secondary configuration file for handling Zillow datasets.  
  Example: `"config/zillow_data_config.json"`

- **`data_path`**: Directory where the datasets are stored.  
  Example: `"data/zillow/zillow_datasets"`

- **`save_to_json_path`**: Path for saving processed results.  
  Example: `"data/zillow/zillow_datasets"`

- **`save_to_json_file`**: Base name for saving results.  
  Example: `"zillow_datasets"`

---

### 3. Data Labels
- **`zillow_data_labels_available`**: A list of all available Zillow data labels that can be processed.  
  Example:  
  ```json
  [
      "median-home-value-zillow-home-value-index-zhvi-",
      "home-values-forecasts-",
      ...
  ]
  ```

- **`zillow_data_labels_target`**: Specifies which data labels should be processed.

---

### 4. Geography Options
Defines the granularity of data processing:
```json
[
    ["U.S."],
    ["Metro & U.S.", "Metro & US"],
    ["State"],
    ["Country"],
    ["City"],
    ["ZIP Code"],
    ["Neighborhood"]
]
```

---

### 5. Datasets
This section defines specific patterns and options for Zillow datasets:
- **`type_pattern`**: Regex patterns to identify data types.
- **`data_type_pattern`**: Regex patterns for dataset categories.
- **`housing_types`**: Supported housing types.
- **`feature_patterns`**: Additional feature extraction rules.

Example:  
For the `zillow_home_values` dataset:
```json
{
    "type_pattern": "(Mortgage Payment|Total Monthly Payment|ZHVI)",
    "data_type_pattern": "Time Series",
    "housing_types": ["All Homes", "Single-Family Homes", "Condo/Co-op", "1-Bedroom", "2-Bedroom"],
    "feature_patterns": {
        "down_payment": "(\\d+%) down",
        "tier": "(Bottom Tier|Top Tier|Mid-Tier)",
        "measure": [
            {"key": "raw", "condition": "Raw"},
            {"key": "smoothed, seasonally adjusted", "condition": "Smoothed.*Seasonally Adjusted"}
        ]
    }
}
```

---

## Example Usage

To load and parse the Zillow configuration file:

```python
from core.file_utils import load_data

# Load the config
config_path = "config/zillow_config.json"
zillow_config = load_data(config_path, file_type="json", load_as="json")

# Access specific fields
print("Base URL:", zillow_config["zillow_base_url"])
print("Data Path:", zillow_config["data_path"])
```

---

## Tips for Customization

- **Add New Labels**: Update `zillow_data_labels_target` with additional datasets you want to process.
- **Change Geography Options**: Modify `geography_options` for different levels of granularity.
- **Update Metadata**: Keep the `last_updated` field accurate for better tracking.

---