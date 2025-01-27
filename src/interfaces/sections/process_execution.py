import streamlit as st
from src.registry import process_registry
import os

def render():
    st.subheader("Processes")
    
    process_options = list(process_registry.registry.keys())
    selected_process = st.selectbox("Select a Process", process_options)

    if selected_process:
        process_func = process_registry.registry[selected_process]["func"]
        metadata = process_registry.registry[selected_process]["metadata"]

        st.write("**Description:**", metadata.get("description", "No description available"))
        st.write("**Parameters:**", metadata.get("parameters", "No parameters available"))

        params = {}
        if "parameters" in metadata:
            for param_name, param_meta in metadata["parameters"].items():
                param_type = param_meta.get("type", "str")
                param_desc = param_meta.get("description", "")
                if param_type == "int":
                    params[param_name] = st.number_input(f"{param_name} ({param_desc})", step=1)
                elif param_type == "float":
                    params[param_name] = st.number_input(f"{param_name} ({param_desc})", format="%.2f")
                elif param_type == "bool":
                    params[param_name] = st.checkbox(f"{param_name} ({param_desc})")
                else:
                    params[param_name] = st.text_input(f"{param_name} ({param_desc})")

        if st.button("Run Process"):
            result = process_func(**params)
            st.write("**Result:**", result)

def render_process_execution():
    st.title("Process Execution")

    selected_process = st.selectbox("Select Process", list(process_registry.keys()))
    process_func = process_registry[selected_process]

    st.subheader(f"Parameters for {selected_process}")
    params = {}
    for param, meta in process_func.metadata["parameters"].items():
        if meta["type"] == "str":
            params[param] = st.text_input(meta["description"])
        elif meta["type"] == "float":
            params[param] = st.slider(meta["description"], meta["min"], meta["max"], meta["default"])
        elif meta["type"] == "int":
            params[param] = st.number_input(meta["description"], value=meta["default"])

    if st.button("Run Process"):
        result = process_func(**params)
        st.success("Process executed successfully!")
        st.text_area("Result", result["result"])
        if result["result_file"]:
            with open(result["result_file"], "rb") as f:
                st.download_button("Download Result", data=f, file_name=os.path.basename(result["result_file"]))