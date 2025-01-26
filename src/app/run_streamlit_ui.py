import os

def launch_streamlit():
    os.system("streamlit run src/interfaces/streamlit_dashboard.py") #streamlit run src/interfaces/streamlit_dashboard.py
    
    
    ###uvicorn src.api.entrypoint:app --reload
    
    # mkdocs build --clean
    
    # mkdocs serve --dev-addr=