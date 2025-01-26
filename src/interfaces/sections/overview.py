from src.registry import metaheuristics_registry, model_registry, pipeline_registry, process_registry
from src.registry import MetaHeuristicsRegistry
import streamlit as st


def render():
    st.subheader("Overview of Registries")
    
    # Processes
    with st.expander("Processes"):
        for name, meta in process_registry.registry.items():
            st.write(f"**{name}**: {meta.get('metadata', {}).get('description', 'No description')}")
    
    # Pipelines
    with st.expander("Pipelines"):
        for name, path in pipeline_registry.registry.items():
            st.write(f"**{name}**: {path}")

    # Models
    with st.expander("Models"):
        for name, meta in model_registry.registry.items():
            st.write(f"**{name}**: {meta.get('metadata', {}).get('description', 'No description')}")

    # Metaheuristics
    with st.expander("Metaheuristics"):
        for name, meta in metaheuristics_registry.registry.items():
            st.write(f"**{name}**: {meta.get('metadata', {}).get('description', 'No description')}")
