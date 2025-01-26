import streamlit as st
from src.registry import PROCESS_REGISTRY
from src.registry import DataSourceRegistry

def render():
    st.subheader("Data Files")

    # Список файлов
    file_options = list(DataSourceRegistry.registry.keys())
    selected_file = st.selectbox("Select a Data File", file_options)

    if selected_file:
        file_info = DataSourceRegistry.registry[selected_file]
        st.write("**File Path:**", file_info["path"])
        st.write("**Metadata:**", file_info.get("metadata", {}))

        # Загрузка и просмотр данных
        if st.button("Load Data"):
            import pandas as pd
            df = pd.read_csv(file_info["path"])
            st.write("**Preview of Data:**", df.head())
