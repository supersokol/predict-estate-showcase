import streamlit as st
from src.registry import metaheuristics_registry

def render():
    st.subheader("Metaheuristics")

    # Список метаэвристик
    heuristic_options = list(metaheuristics_registry.registry.keys())
    selected_heuristic = st.selectbox("Select a MetaHeuristic", heuristic_options)

    if selected_heuristic:
        heuristic_func = metaheuristics_registry.registry[selected_heuristic]["func"]
        metadata = metaheuristics_registry.registry[selected_heuristic]["metadata"]

        # Отображение метаданных
        st.write("**Description:**", metadata.get("description", "No description available"))

        # Запуск эвристики
        if st.button("Run MetaHeuristic"):
            import pandas as pd
            uploaded_file = st.file_uploader("Upload Data for Analysis")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                result = heuristic_func(df)
                st.write("**Result:**", result)
