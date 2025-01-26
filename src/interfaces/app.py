import sys
sys.path.append("Q:/SANDBOX/PredictEstateShowcase_dev/src")
sys.path.append("Q:/SANDBOX/PredictEstateShowcase_dev/")
print(sys.path)
import os
import json
from sdk.api_client import APIClient
import streamlit as st
import requests
from registry import data_source_registry
from core.logger import logger
from interfaces.section_loader import load_sections

def render_api_methods():
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
                st.json(result)
            except Exception as e:
                st.error(f"Error: {e}")



from src.core.db_manager import DatabaseManager

# Path to the database file
db_file = "data/app_database.db"

# Create an instance of the DatabaseManager
db_manager = DatabaseManager(db_file)

# Initialize the database
db_manager.initialize_db()
logger.info("App initialized")

# Load configuration
MASTER_CONFIG_PATH = "config/master_config.json"
with open(MASTER_CONFIG_PATH, "r") as f:
    master_config = json.load(f)

API_URL = master_config["api_url"]
ELEMENTS_CONFIG_PATH = master_config["configs"]["elements_config_path"]

# Initialize API client and data registry
client = APIClient(API_URL)
data_registry = data_source_registry()

# Load sections dynamically
sections = load_sections(ELEMENTS_CONFIG_PATH)


theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=0)
st.markdown(f"<style>body {{ background-color: {'#ffffff' if theme == 'Light' else '#333333'}; }}</style>", unsafe_allow_html=True)

# Sidebar for section navigation
st.sidebar.header("Navigation")
section_name = st.sidebar.radio("Select Section", list(sections.keys()))

# Sidebar links
st.sidebar.header("Quick Links")
st.sidebar.markdown("[GitHub](https://github.com/supersokol/predict-estate-showcase)", unsafe_allow_html=True)
st.sidebar.markdown("[Detailed User Guide](http://127.0.0.1:8000/mkdocs/index.html)", unsafe_allow_html=True)
st.sidebar.markdown("[API Documentation](http://127.0.0.1:8000/docs)", unsafe_allow_html=True)

st.title("Workspace Automation")
# Render the selected section
if section_name in sections:
    try:
        sections[section_name].render()
    except Exception as e:
        logger.exception(f"Error rendering section '{section_name}': {e}")
        st.error(f"An error occurred while rendering the section '{section_name}'.")
else:
    st.error(f"Section '{section_name}' not found.")


st.title("Data Sources Management")

# Вывод списка зарегистрированных источников данных
st.header("Registered Data Sources")
response = requests.get(f"{API_URL}/data/sources")
if response.status_code == 200:
    sources = response.json()
    selected_source = st.selectbox("Select a Data Source", sources)

    if selected_source:
        # Получение файлов
        files_response = requests.get(f"{API_URL}/data/files/{selected_source}")
        if files_response.status_code == 200:
            st.write("**Current Files:**", files_response.json().get("local_files", []))
        # Добавление новых файлов
        st.write("### Add New Files")
        new_files = st.text_area("Enter paths to new files (comma-separated)", "")
        if st.button("Add Files"):
            new_files_list = [f.strip() for f in new_files.split(",") if f.strip()]
            if new_files_list:
                add_files_response = requests.post(
                    f"{API_URL}/data/files/add",
                    json={"source_name": selected_source, "new_files": new_files_list}
                )
                if add_files_response.status_code == 200:
                    st.success("Files added successfully!")
                    updated_files = add_files_response.json().get("updated_files", [])
                    st.write("**Updated Files:**", updated_files)
                else:
                    st.error(f"Failed to add files: {add_files_response.json()['detail']}")
        # Получение образца данных
        if st.button("Get Sample"):
            sample_response = requests.get(f"{API_URL}/data/sample/{selected_source}")
            if sample_response.status_code == 200:
                st.write("**Sample Data:**")
                st.dataframe(sample_response.json()["sample"])
else:
    st.error("Failed to fetch data sources.")

# Инициализация клиента
client = APIClient(API_URL)

# Список методов
st.title("API Client Interface")
methods = client.list_methods()

# Отображение доступных методов
selected_method = st.selectbox("Select an API method", list(methods.keys()))
if selected_method:
    metadata = methods[selected_method]
    st.write("**Description:**", metadata["description"])
    st.write("**Path:**", metadata["path"])
    st.write("**HTTP Method:**", metadata["method"])

    # Ввод параметров
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

    # Выполнение метода
    if st.button("Execute"):
        try:
            func = getattr(client, selected_method)
            result = func(**params)
            st.write("**Result:**", result)
        except Exception as e:
            st.error(f"Error: {e}")
