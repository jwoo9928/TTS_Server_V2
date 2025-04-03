# Use Python 3.9 slim image as a base
ARG PYTHON_VERSION=3.12.0
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
ARG TORCH_VERSION=2.3.1 # Specify desired torch CPU version
ARG TORCHVISION_VERSION=0.18.0
ARG TORCHAUDIO_VERSION=2.3.1

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cpu \
    torchvision==${TORCHVISION_VERSION}+cpu \
    torchaudio==${TORCHAUDIO_VERSION}+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# --- GPU Stage ---
# For GPU, we ideally need a base image with CUDA drivers.
# Using nvidia/cuda image is recommended for simplicity.
# Example: FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
# However, sticking to the python base and installing torch+cuda for flexibility as requested.
# NOTE: This requires the host to have NVIDIA drivers and nvidia-docker/nvidia-container-toolkit.
FROM base as gpu
ARG TORCH_VERSION=2.1.0 # Specify desired torch GPU version
ARG TORCHVISION_VERSION=0.16.0
ARG TORCHAUDIO_VERSION=2.1.0
ARG CUDA_VERSION=11.8 # Specify CUDA version compatible with host and torch version

# Install PyTorch GPU version (matching CUDA version)
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.} # Format CUDA version like cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final CPU Target Stage ---
# This stage uses the 'cpu' build stage as its base
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
# This stage uses the 'gpu' build stage as its base
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
# docker builder prune -a --force && docker build --target final-cpu -t tts-server:cpu . --no-cache

# 
# To run with GPU:
# docker run --gpus all -p 8080:8080 tts-server:gpu
#
# To run with CPU:
# docker run -p 8080:8080 tts-server:cpu