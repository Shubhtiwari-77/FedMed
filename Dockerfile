FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8501
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]
