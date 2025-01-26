# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# Команда для запуска обоих сервисов
CMD ["sh", "-c", "uvicorn src.api.entrypoint:app --host 0.0.0.0 --port 8000 & streamlit run src/streamlit_app.py --server.port 8501"]
