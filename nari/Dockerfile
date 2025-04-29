# Use the recommended PyTorch base image from the reference Dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (including venv tools)
# Keep libsndfile1-dev just in case soundfile compilation needs it
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user and set up directories
# Using UID 1000 as it's a common default, adjust if needed
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser nari/requirements.txt .

# Install Python dependencies within the venv
# Upgrade pip first
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=appuser:appuser nari/app.py .

# Set environment variables (adapted from reference)
# LD_LIBRARY_PATH might be inherited or set correctly by the base image,
# but explicitly setting it based on the reference is safer.
ENV USE_GPU=true
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

# Expose the FastAPI port
EXPOSE 8080

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
