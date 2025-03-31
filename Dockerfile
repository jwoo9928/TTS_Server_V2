# Use Python 3.9 slim image as a base
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
# - espeak-ng is required by kokoro
# - libsndfile1 is required by soundfile
# - git is needed if any pip package installs from git
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# --- CPU Stage ---
FROM base as cpu
ARG TORCH_VERSION=2.1.0 # Specify desired torch CPU version
ARG TORCHVISION_VERSION=0.16.0
ARG TORCHAUDIO_VERSION=2.1.0

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cpu \
    torchvision==${TORCHVISION_VERSION}+cpu \
    torchaudio==${TORCHAUDIO_VERSION}+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .

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

# Copy the application code
COPY main.py .

# --- Final CPU Target Stage ---
# This stage uses the 'cpu' build stage as its base
FROM cpu as final-cpu
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# --- Final GPU Target Stage ---
# This stage uses the 'gpu' build stage as its base
FROM gpu as final-gpu
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# To build, use --target flag:
# For CPU: docker build --target final-cpu -t tts-server:cpu .
# For GPU: docker build --target final-gpu -t tts-server:gpu .
