import streamlit as st
import pandas as pd
import os
import json
import requests

# Function to load data from API
@st.cache_data
def load_table():
    """
    Loads the table of files from the API.

    Returns:
        pd.DataFrame: DataFrame containing the data from the API.
    """
    response = requests.get("http://127.0.0.1:8000/data_sources")
    data = response.json()
    return pd.DataFrame(data)

# Function to process metadata
def parse_metadata(metadata):
    """
    Parses the metadata JSON string.

    Args:
        metadata (str): JSON string containing metadata.

    Returns:
        dict: Parsed metadata dictionary.
    """
    if isinstance(metadata, dict):
        return metadata
    try:
        return json.loads(metadata)
    except json.JSONDecodeError:
        return {}

# Function to load a file
def download_file(file_path):
    """
    Reads the file content from the given file path.

    Args:
        file_path (str): Path to the file.

    Returns:
        bytes: Content of the file.
    """
    with open(file_path, "rb") as f:
        return f.read()

# Main interface function
def render():
    """
    Renders the Streamlit interface for the filtered data source registry.
    """
    st.title("Filtered Data Source Registry")

    # Load data
    table = load_table()

    # Filters: Data types
    st.sidebar.header("Filters")
    file_types = table["file_type"].unique()
    selected_types = st.sidebar.multiselect("Select file types", file_types, default=file_types)

    # Filters: Formats
    formats = table["format"].unique()
    selected_formats = st.sidebar.multiselect("Select file formats", formats, default=formats)

    # Filter the table
    filtered_table = table[
        table["file_type"].isin(selected_types) & table["format"].isin(selected_formats)
    ]

    # Summary information
    st.header("Summary")
    total_files = len(table)
    filtered_files = len(filtered_table)
    total_size = filtered_table["metadata"].apply(lambda x: parse_metadata(x).get("size_bytes", 0)).sum()

    st.text(f"Total files: {total_files}")
    st.text(f"Filtered files: {filtered_files}")
    st.text(f"Total size of filtered files: {total_size} bytes")

    # Additional information for CSV/datasets
    if "csv" in selected_formats or "dataset" in selected_types:
        csv_data = filtered_table[filtered_table["format"] == "csv"]
        if not csv_data.empty:
            total_lines = csv_data["metadata"].apply(lambda x: parse_metadata(x).get("lines", 0)).sum()
            total_chars = csv_data["metadata"].apply(lambda x: parse_metadata(x).get("chars", 0)).sum()

            st.text(f"Total lines in filtered CSV files: {total_lines}")
            st.text(f"Total characters in filtered CSV files: {total_chars}")

            # Statistics for lines and columns
            sample_stats = pd.DataFrame({
                "Lines": csv_data["metadata"].apply(lambda x: parse_metadata(x).get("lines", 0)),
                "Chars": csv_data["metadata"].apply(lambda x: parse_metadata(x).get("chars", 0)),
            })
            stats = sample_stats.describe().transpose()
            st.dataframe(stats)

    # List of paths
    st.header("Filtered File Paths")
    file_paths = filtered_table["file_path"].tolist()
    st.text_area("Filtered File Paths", "\n".join(file_paths), height=200)

    # File selection
    st.header("File Details")
    selected_file = st.selectbox("Select a file", file_paths)

    if selected_file:
        file_details = filtered_table[filtered_table["file_path"] == selected_file].iloc[0]
        metadata = parse_metadata(file_details["metadata"])

        st.text(f"File Path: {selected_file}")
        st.text(f"File Type: {file_details['file_type']}")
        st.text(f"Format: {file_details['format']}")
        st.text(f"Size: {metadata.get('size_bytes', 'Unknown')} bytes")
        st.text(f"Timestamp: {file_details['timestamp']}")

        # Sample content of the file (for text/CSV files)
        if file_details["format"] in ["txt", "csv"]:
            try:
                with open(file_details["file_path"], "r", encoding="utf-8") as f:
                    content = f.read()
                    st.text_area("File Sample", content[:500], height=200)
            except Exception as e:
                st.error(f"Error reading file: {e}")

        # Download button
        if st.button("Download File"):
            file_content = download_file(selected_file)
            st.download_button(
                label="Download",
                data=file_content,
                file_name=os.path.basename(selected_file),
                mime="application/octet-stream",
            )

# Example usage as a section
if __name__ == "__main__":
    render()
