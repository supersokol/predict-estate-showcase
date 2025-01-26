from src.api.api_client import APIClient
import streamlit as st
import json

def render():
    MASTER_CONFIG_PATH = "config/master_config.json"
    with open(MASTER_CONFIG_PATH, "r") as f:
        master_config = json.load(f)

    API_URL = master_config["api_url"]
    client = APIClient(API_URL)
    st.title("API Client Interface")
    methods = client.list_methods()

    # Select method
    selected_method = st.selectbox("Select an API Method", list(methods.keys()))
    if selected_method:
        metadata = methods[selected_method]
        st.write("**Description:**", metadata["description"])
        st.write("**Path:**", metadata["path"])
        st.write("**HTTP Method:**", metadata["method"])

        # Input parameters
        params = {}
        for param in metadata["parameters"]:
            param_name = param["name"]
            param_type = param.get("schema", {}).get("type", "string")
            if param_type == "integer":
                params[param_name] = st.number_input(param_name, step=1)
            elif param_type == "boolean":
                params[param_name] = st.checkbox(param_name)
            else:
                params[param_name] = st.text_input(param_name)

        # Execute method
        if st.button("Execute"):
            try:
                func = getattr(client, selected_method)
                result = func(**params)
                st.success("Execution Successful!")
                st.write("**Result:**", result)
                st.json(result)
            except Exception as e:
                st.error(f"Error: {e}")
