import os
import streamlit as st
from registry import DataSourceRegistry

data_registry = DataSourceRegistry()
def render():
    st.title("Source Overview")

    # Категории источников
    source_type = st.selectbox("Select Source Type", ["datasets", "uploaded", "configs"])
    sources = data_registry.list_sources()
    
    # Выбор источника
    selected_source = st.selectbox("Select a Source", sources)
    if selected_source:
        source_metadata = data_registry.get_source(selected_source)
        st.subheader("Source Metadata")
        st.json(source_metadata)
    
        # Отображение файлов
        if source_metadata["local_files"].get(source_type):
            st.subheader(f"{source_type.capitalize()} Files")
            file_list = source_metadata["local_files"][source_type]
            for file_path in file_list:
                st.write(file_path)
                with open(file_path, "rb") as f:
                    st.download_button("Download File", data=f, file_name=os.path.basename(file_path))
