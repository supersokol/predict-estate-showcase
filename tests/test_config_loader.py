import os
import pytest
from src.registry.data_workflow import check_and_load_config_zillow

@pytest.fixture
def valid_config(tmp_path):
    """Creates a temporary configuration with a valid data_config_path."""
    config_path = tmp_path / "zillow_config.json"
    config_data = {
        "prefix": "zillow",
        "data_config_path": str(config_path),
    }
    with open(config_path, "w") as f:
        f.write('{"zillow_data_config": {"file_path": "data/zillow_datasets.json"}}')
    return config_data

def test_load_valid_config(valid_config):
    """Test for successful configuration loading.
    
    This test ensures that the configuration is loaded correctly when the 
    data_config_path contains a valid file_path.
    """
    result = check_and_load_config_zillow(valid_config)
    assert result is not None, "Expected result to be loaded successfully."

def test_missing_file_path(valid_config):
    """Test for missing file_path key in the configuration.
    
    This test checks the behavior of the function when the main configuration 
    file does not contain the 'file_path' key. It should raise a ValueError.
    """
    config_path = valid_config["data_config_path"]
    with open(config_path, "w") as f:
        f.write('{}')  # Empty JSON
    with pytest.raises(ValueError, match="No file_path found in the main config."):
        check_and_load_config_zillow(valid_config)

def test_fetch_new_data(mocker, valid_config):
    """Test for scenario where the file is missing and new data needs to be fetched.
    
    This test simulates the situation where the specified file in the configuration 
    does not exist, and the function should fetch new data. It uses mocking to 
    simulate the fetch operation.
    """
    mock_fetch = mocker.patch("src.registry.data_workflow.fetch_zillow_download_links", return_value={
        "metadata": {"file_path": "data/zillow_new_data.json"}
    })
    result = check_and_load_config_zillow(valid_config)
    mock_fetch.assert_called_once()
    assert result is not None, "Expected new data to be fetched."
