# Use an official NVIDIA CUDA base image compatible with PyTorch 2.x and CUDA 12.x
# Using 12.4 as a stable base, generally compatible with 12.6 runtime needs for PyTorch
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and common dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies, including PyTorch for CUDA 12.1
# This step might take a while
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY app.py .
# If the 'dia' model needs to be copied or downloaded, add steps here
# e.g., COPY ./path/to/dia/model /app/dia/model
# Or RUN commands to download it

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using Uvicorn with multiple workers
# The number of workers can be adjusted based on the specific GPU and workload
# Start with a reasonable number like 4, but monitor performance
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
