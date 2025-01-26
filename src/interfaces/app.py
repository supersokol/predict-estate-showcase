import sys
sys.path.append("Q:/SANDBOX/PredictEstateShowcase_dev/src")
sys.path.append("Q:/SANDBOX/PredictEstateShowcase_dev/")
import streamlit as st
from core.logger import logger
from core.config_loader import load_all_configs_via_master
from interfaces.section_loader import load_sections

#db_path = os.getenv("DB_PATH")
# Path to the database file
#db_file = db_path + "app_database.db"
# Create an instance of the DatabaseManager
#db_manager = DatabaseManager(db_file)
# Initialize the database
#db_manager.initialize_db()

logger.info("App initialized")

loaded_configs = load_all_configs_via_master()
ELEMENTS_CONFIG_PATH = loaded_configs['master_config']['configs']['elements_config_path']

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

st.title("PredictStateShowcase")
# Render the selected section
if section_name in sections:
    try:
        sections[section_name].render()
    except Exception as e:
        logger.exception(f"Error rendering section '{section_name}': {e}")
        st.error(f"An error occurred while rendering the section '{section_name}'.")
else:
    st.error(f"Section '{section_name}' not found.")
