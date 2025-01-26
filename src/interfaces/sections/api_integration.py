from sdk.api_client import APIClient
import streamlit as st

def render(api_url):
    client = APIClient(api_url)
    st.title("API Methods")

    # Выбор метода
    methods = client.list_methods()
    selected_method = st.selectbox("Select Method", list(methods.keys()))

    if selected_method:
        method_meta = methods[selected_method]
        st.write("**Description:**", method_meta["description"])
        st.write("**Path:**", method_meta["path"])

        # Параметры метода
        params = {}
        for param in method_meta["parameters"]:
            param_name = param["name"]
            if param["type"] == "string":
                params[param_name] = st.text_input(param_name)
            elif param["type"] == "integer":
                params[param_name] = st.number_input(param_name)

        # Выполнение метода
        if st.button("Run Method"):
            result = client.execute_method(selected_method, params)
            st.subheader("Result")
            st.json(result)
