import streamlit as st
from src.registry import model_registry

def render():
    st.subheader("Models")

    # Список моделей
    model_options = list(model_registry.registry.keys())
    selected_model = st.selectbox("Select a Model", model_options)

    if selected_model:
        model_func = model_registry.registry[selected_model]["func"]
        metadata = model_registry.registry[selected_model]["metadata"]

        # Отображение метаданных
        st.write("**Description:**", metadata.get("description", "No description available"))
        st.write("**Parameters:**", metadata.get("parameters", "No parameters available"))

        # Ввод параметров
        params = {}
        if "parameters" in metadata:
            for param_name, param_meta in metadata["parameters"].items():
                param_desc = param_meta.get("description", "")
                params[param_name] = st.text_input(f"{param_name} ({param_desc})")

        # Обучение модели
        if st.button("Train Model"):
            import pandas as pd
            uploaded_file = st.file_uploader("Upload Training Data")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                target_column = st.selectbox("Select Target Column", df.columns)
                X = df.drop(columns=[target_column])
                y = df[target_column]
                model = model_func(X, y, **params)
                st.write("**Trained Model:**", model)
