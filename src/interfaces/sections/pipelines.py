import streamlit as st
from src.registry import pipeline_registry


def render():
    st.subheader("Pipelines")

    # Список пайплайнов
    pipeline_options = list(pipeline_registry.registry.keys())
    selected_pipeline = st.selectbox("Select a Pipeline", pipeline_options)

    if selected_pipeline:
        pipeline_config = pipeline_registry.registry[selected_pipeline]
        st.write("**Description:**", pipeline_config.get("description", "No description available"))
        st.write("**Steps:**", pipeline_config.get("steps", []))

        # Выбор файла данных
        uploaded_file = st.file_uploader("Upload a file to process")
        if uploaded_file:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.write("**Preview of Data:**", df.head())


