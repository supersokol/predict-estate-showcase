import subprocess

def run_fastapi():
    subprocess.run(["uvicorn", "src.api.entrypoint:app", "--reload"])

def run_streamlit():
    subprocess.run(["streamlit", "run", "src/interfaces/app.py"])

if __name__ == "__main__":
    from multiprocessing import Process

    fastapi_process = Process(target=run_fastapi)
    streamlit_process = Process(target=run_streamlit)

    fastapi_process.start()
    streamlit_process.start()

    fastapi_process.join()
    streamlit_process.join()