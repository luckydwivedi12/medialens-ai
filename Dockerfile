# Production Dockerfile for MediaLens AI backend on Render / Linux containers
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_FORCE_DNN_ENGINE=2 \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies required by OpenCV and HTTP downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements (CPU PyTorch and application dependencies)
COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# Copy application code
COPY backend/app ./app
COPY backend/scripts ./scripts

# Create model directory with appropriate permissions
RUN mkdir -p /app/models/medialens_mobilenetv3 /app/models/yunet

EXPOSE 8000

# Listen on 0.0.0.0 using Render's dynamic PORT environment variable
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
