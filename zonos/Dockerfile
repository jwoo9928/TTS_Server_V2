# Define Python version
ARG PYTHON_VERSION=3.10

# Base image
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH="/app:${PYTHONPATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install dependencies using wheels
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# -------------------------------------
# CPU Build Stage
# -------------------------------------
FROM base as final-cpu

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder stage
COPY --from=base /wheels /wheels

# Install from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy application source code
COPY . .

# Create a directory for speaker reference files
RUN mkdir -p /app/speaker_references

# Expose port
EXPOSE 8080

# Set environment variables
ENV MODEL_TYPE=hybrid \
    DEVICE=cpu

# Entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command
CMD ["python", "main.py"]

# -------------------------------------
# GPU Build Stage
# -------------------------------------
FROM base as final-gpu

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set environment variables
ENV MODEL_TYPE=hybrid \
    DEVICE=cuda

# Reuse the same setup as CPU build
COPY --from=final-cpu /app /app

# Expose port
EXPOSE 8080

# Entrypoint script
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command
CMD ["python", "main.py"]
