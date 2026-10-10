
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt ./

# Install CPU-only PyTorch with a compatible torchvision version.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchvision==0.21.0

# Keep dependency resolution from upgrading PyTorch to a CUDA build.
RUN pip install --no-cache-dir \
    -c constraints.txt \
    -r requirements.txt

COPY . .

RUN mkdir -p /data/uploads

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
