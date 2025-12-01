FROM pytorch/pytorch:2.2.0-cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install Python dependencies (torch already present in base image)
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Default command; override with container command args.
CMD ["python", "train_multimodal_massive.py", "--manifest", "manifests/multimodal_manifest.csv", "--epochs", "10"]
FROM python:3.9-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements if present
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt || true

# Copy repository
COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "train_multimodal_rcma.py"]
