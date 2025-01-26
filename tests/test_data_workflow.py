import os
import pytest
from src.registry.data_workflow import manage_data_processing
import json

@pytest.fixture
def config(tmp_path):
    """Creates a temporary configuration for testing."""
    config_path = os.path.join("config", "zillow_config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config_data["data_config_path"] = str(tmp_path / "zillow_config.json")
    config_data["data_path"] = str(tmp_path / "data/zillow")
    return config_data

@pytest.fixture
def invalid_config(tmp_path):
    """Creates an invalid configuration for testing."""
    return {
        "prefix": "zillow",
        "data_config_path": str(tmp_path / "invalid_config.json"),
        # Missing 'data_path' key to simulate invalid configuration
    }

def test_manage_existing_files(mocker, config, tmp_path):
    """Test scenario where files already exist and should be managed accordingly."""
    # Create a directory and a dummy existing file to simulate pre-existing data
    data_path = os.path.join(config["data_path"], "scraper_data_20250125-051342")
    os.makedirs(data_path)
    existing_file = os.path.join(data_path, "zillow_result_20250125-051342.json")
    with open(existing_file, "w") as f:
        f.write('{"test_key": "test_value"}')

    # Mock the function to return the existing file
    mocker.patch("src.registry.data_workflow.get_files_for_last_timestamps", return_value=[existing_file])

    # Call the function under test
    result = manage_data_processing(config)

    # Verify that the result contains the expected files
    assert "files" in result, "Expected result to contain 'files' key."
    assert existing_file in result["files"], "Expected existing file to be returned."

def test_manage_process_new_data(mocker, config):
    """Test scenario where no files exist and new data needs to be processed."""
    # Mock the data processing function to simulate processing new data
    mock_process = mocker.patch("src.registry.data_workflow.process_zillow_datasets", return_value={"processed": True})

    # Call the function under test
    result = manage_data_processing(config)

    # Verify that the data processing function was called and the result is as expected
    mock_process.assert_called_once()
    assert result == {"processed": True}, "Expected result from process_zillow_datasets."

with pytest.raises(RuntimeError):
    manage_data_processing(invalid_config)
