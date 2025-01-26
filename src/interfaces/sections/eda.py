import streamlit as st
import pandas as pd
from src.registry import PROCESS_REGISTRY

def render():
    st.title("Exploratory Data Analysis (EDA)")
    
    # File selection
    file_options = ["file1.csv", "file2.csv"]  # Заменить на реальные данные из `current_data_config.json`
    selected_file = st.selectbox("Select a file for analysis", file_options)
    
    # Load data
    if selected_file:
        data = pd.read_csv(selected_file)
        st.write(f"Data loaded: {selected_file}")
        st.write(data.head())
        
        # Select process
        selected_process = st.selectbox("Select an EDA process", list(PROCESS_REGISTRY.keys()))
        if selected_process:
            process_func = PROCESS_REGISTRY[selected_process]
            st.write(f"Running: {selected_process}")
            result = process_func(data=data)
            st.write(result)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            elif hasattr(result, "show"):
                st.plotly_chart(result)
