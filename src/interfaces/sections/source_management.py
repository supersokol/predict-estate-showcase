import os
import streamlit as st
from core.file_utils import get_context
from registry import DataSourceRegistry

data_registry = DataSourceRegistry()

def render():
    st.title("Source Management")

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
            st.success(f"File saved to {save_path}")
            context_data = {"file_path": save_path}
    elif content_type == "url":
        context_data = {"url": st.text_input("Enter URL")}
    elif content_type == "wiki":
        context_data = {"term": st.text_input("Enter Wikipedia Term")}

    if st.button("Add Source"):
        try:
            processed_context = get_context(content_type, **context_data)
            file_path = f"data/uploaded/{source_name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(processed_context)
            st.success(f"Source '{source_name}' added successfully!")
        except Exception as e:
            st.error(f"Error processing source: {e}")
