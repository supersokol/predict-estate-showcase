from src.core.file_utils import get_context
import streamlit as st
import os
from typing import Optional


def render():
    """Renders the Streamlit interface for uploading data."""
    st.title("Upload Data")

    # Select directory for upload
    st.subheader("Select Upload Path")
    base_path = "data"
    all_directories = [base_path]
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            all_directories.append(os.path.join(root, d))

    upload_path = st.selectbox("Upload Path", all_directories)

    # Select data source
    st.subheader("Select Source Option")
    source_option = st.radio("Select Source Option", ["url", "wiki term", "text", "file"])

    # Process input data using get_context
    input_data = None
    try:
        if source_option == "url":
            url = st.text_input("Enter URL")
            if url and st.button("Preview"):
                input_data = get_context(context_type="url", url=url)
                st.text_area("Preview of the URL content", input_data[:1000], height=300)
                st.write(f"Full response length: {len(input_data)} characters")
        elif source_option == "wiki term":
            wiki_term = st.text_input("Enter Wikipedia Term")
            if wiki_term and st.button("Preview"):
                input_data = get_context(context_type="wiki", term=wiki_term)
                st.text_area("Preview of the Wikipedia article", input_data[:1000], height=300)
                st.write(f"Full response length: {len(input_data)} characters")
        elif source_option == "text":
            input_data = st.text_area("Enter Text Content", height=300)
        elif source_option == "file":
            uploaded_file = st.file_uploader("Upload File", type=["txt", "csv", "json", "pdf", "md"])
            if uploaded_file:
                # Save temporary file for get_context
                temp_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                input_data = get_context(context_type="file", file_path=temp_path)
    except Exception as e:
        st.error(f"Error while processing data: {e}")

    # Upload data button
    if st.button("Upload"):
        if not input_data:
            st.error("No data to upload. Please provide input.")
        else:
            # Generate path for saving file
            file_name = {
                "url": "uploaded_url_data.txt",
                "wiki term": "uploaded_wiki_data.txt",
                "text": "uploaded_text_data.txt",
                "file": uploaded_file.name if source_option == "file" else "uploaded_file_data.txt",
            }.get(source_option, "uploaded_data.txt")
            save_path = os.path.join(upload_path, file_name)

            # Save data to file
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(input_data)
                st.success(f"Data uploaded successfully to {save_path}.")
                st.info("The uploaded data is now registered and available for further processing.")
            except Exception as e:
                st.error(f"Failed to save data: {e}")

# Example usage as a section
if __name__ == "__main__":
    render()
