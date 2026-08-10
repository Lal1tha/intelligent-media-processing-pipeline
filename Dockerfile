FROM python:3.11-slim
WORKDIR /app
# Console-script entry points (notably Alembic) do not guarantee that WORKDIR is
# on sys.path. Keep the application package importable for migrations, API, and RQ.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data/uploads
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
