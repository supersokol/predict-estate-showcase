import os
import streamlit as st
from src.core.file_utils import get_context
from src.registry import DataSourceRegistry

data_registry = DataSourceRegistry()

def render():
    st.title("Local Source Management")

    # Добавление нового источника
    st.subheader("Add New Source")
    source_name = st.text_input("Source Name")
    description = st.text_area("Description")
    content_type = st.selectbox("Content Type", ["text", "file", "url", "wiki"])
    context_data = None

    if content_type == "text":
        context_data = st.text_area("Enter Text")
    elif content_type == "file":
        uploaded_file = st.file_uploader("Upload File")
        if uploaded_file:
            save_path = f"data/uploaded/{uploaded_file.name}"
            os.makedirs("data/uploaded", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File uploaded to {save_path}")
            context_data = {"file_path": save_path}
    elif content_type == "url":
        url = st.text_input("Enter URL")
        if url:
            context_data = {"url": url}
    elif content_type == "wiki":
        term = st.text_input("Enter Wikipedia Term")
        if term:
            context_data = {"term": term}

    if st.button("Process and Add Source"):
        try:
            processed_context = get_context(content_type, **context_data)
            save_path = f"data/uploaded/{source_name}.txt"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(processed_context if isinstance(processed_context, str) else str(processed_context))
            data_registry.register_source(source_name, {"description": description, "local_files": {"uploaded": [save_path]}})
            st.success(f"Source '{source_name}' added and saved to {save_path}")
        except Exception as e:
            st.error(f"Error processing source: {e}")
