# Use Python 3.12 slim image as a base
ARG PYTHON_VERSION=3.12.0
FROM python:${PYTHON_VERSION}-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH="/app:${PYTHONPATH:-}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    git \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all application files (including main.py and project files)
COPY . .

# (Optional) If the project uses editable install, ensure a setup.py or pyproject.toml exists.
# Here, we assume the project is installed via requirements.
# Install additional dependencies using pip from requirements.txt.
# RUN echo \"fastapi>=0.104.1\nuvicorn>=0.24.0\nsoundfile>=0.12.1\nnumpy>=1.24.0\nzonos\" > requirements.txt

# Install project dependencies in editable mode (if setup.py exists)
RUN pip install --upgrade pip && pip install -e . || echo 'No setup.py found, skipping editable install.'
RUN pip install --no-build-isolation -e .[compile] || echo 'No [compile] extras found, skipping.'

# --- CPU Stage ---
FROM base AS cpu
ARG TORCH_VERSION=2.3.0
ARG TORCHVISION_VERSION=0.18.0
ARG TORCHAUDIO_VERSION=2.3.0

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cpu \
    torchvision==${TORCHVISION_VERSION}+cpu \
    torchaudio==${TORCHAUDIO_VERSION}+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- GPU Stage ---
FROM base AS gpu
ARG TORCH_VERSION=2.3.0
ARG TORCHVISION_VERSION=0.18.0
ARG TORCHAUDIO_VERSION=2.3.0
ARG CUDA_VERSION=11.8

SHELL ["/bin/bash", "-c"]
# Install PyTorch GPU version (with CUDA)
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cu${CUDA_VERSION//.} \
    torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION//.} \
    torchaudio==${TORCHAUDIO_VERSION}+cu${CUDA_VERSION//.} \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Final CPU Target Stage ---
FROM cpu AS final-cpu
WORKDIR /app

# Copy the main server code (main.py)
COPY main.py .

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for API
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# --- Final GPU Target Stage ---
FROM gpu AS final-gpu
WORKDIR /app

# Copy the main server code (main.py)
COPY main.py .

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos \"\" appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for API
EXPOSE 8080
# Set NVIDIA GPU environment variables
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# Build commands:
# For CPU: docker build --target final-cpu -t tts-server:cpu .
# For GPU: docker build --target final-gpu -t tts-server:gpu .
#
# Run commands:
# For CPU: docker run -p 8080:8080 tts-server:cpu
# For GPU: docker run --gpus all -p 8080:8080 tts-server:gpu
