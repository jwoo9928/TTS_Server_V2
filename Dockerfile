# Use Python 3.11 slim image as a base
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH="/app:${PYTHONPATH}"

# Install system dependencies
# - espeak-ng is required by kokoro
# - libsndfile1 is required by soundfile
# - git is needed if any pip package installs from git
# - build-essential and python3-dev for potential compilation requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    git \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Create a requirements.txt file if not already existing
RUN echo "fastapi>=0.104.1\n\
uvicorn>=0.24.0\n\
kokoro\n\
soundfile>=0.12.1\n\
numpy>=1.24.0" > /app/requirements.txt

# --- CPU Stage ---
FROM base as cpu
ARG TORCH_VERSION=2.1.0
ARG TORCHVISION_VERSION=0.16.0
ARG TORCHAUDIO_VERSION=2.1.0

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cpu \
    torchvision==${TORCHVISION_VERSION}+cpu \
    torchaudio==${TORCHAUDIO_VERSION}+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- GPU Stage ---
FROM base as gpu
ARG TORCH_VERSION=2.1.0
ARG TORCHVISION_VERSION=0.16.0
ARG TORCHAUDIO_VERSION=2.1.0
ARG CUDA_VERSION=11.8

# Install PyTorch GPU version
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final CPU Target Stage ---
FROM cpu as final-cpu
WORKDIR /app

# Copy application code
COPY main.py .

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port and set default command
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# --- Final GPU Target Stage ---
FROM gpu as final-gpu
WORKDIR /app

# Copy application code
COPY main.py .

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port and set default command
EXPOSE 8080
# NVIDIA CUDA toolkit environment variables
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# To build, use --target flag:
# For CPU: docker build --target final-cpu -t tts-server:cpu .
# For GPU: docker build --target final-gpu -t tts-server:gpu .
# 
# To run with GPU:
# docker run --gpus all -p 8080:8080 tts-server:gpu
#
# To run with CPU:
# docker run -p 8080:8080 tts-server:cpu