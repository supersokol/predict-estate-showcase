# Test Examples

## Test: manage_data_processing
```python
def test_manage_existing_files(mocker, config, tmp_path):
    data_path = os.path.join(config["data_path"], "scraper_data_20250125-051342")
    os.makedirs(data_path)
    existing_file = os.path.join(data_path, "zillow_result_20250125-051342.json")
    with open(existing_file, "w") as f:
        f.write('{"test_key": "test_value"}')

    mocker.patch("src.registry.data_workflow.get_files_for_last_timestamps", return_value=[existing_file])

    result = manage_data_processing(config)
    assert "files" in result
    assert existing_file in result["files"]
```