import streamlit as st
import pandas as pd
from src.registry import PROCESS_REGISTRY
from src.registry import DataSourceRegistry
import os

def render():
    st.title("Data Processing")

    # Выбор источника и процесса
    source_type = st.selectbox("Select Source Type", ["uploaded", "datasets"])
    sources = DataSourceRegistry.list_sources()
    selected_source = st.selectbox("Select a Source", sources)
    processes = list(PROCESS_REGISTRY.keys())
    selected_process = st.selectbox("Select a Process", processes)

    if st.button("Run Process"):
        source_metadata = DataSourceRegistry.get_source(selected_source)
        input_file = source_metadata["local_files"][source_type][0]
        with open(input_file, "r") as f:
            data = f.read()

        process_func = PROCESS_REGISTRY[selected_process]
        result = process_func(data)
        result_path = f"data/temp/{selected_source}_{selected_process}.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result)

        st.success("Process completed!")
        st.text_area("Result", result)
        with open(result_path, "rb") as f:
            st.download_button("Download Result", data=f, file_name=os.path.basename(result_path))
