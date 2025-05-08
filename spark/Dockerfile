# ---------- Dockerfile ----------
    FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

    # Set work directory
    WORKDIR /app
    
    # Install Python and pip
    RUN apt-get update && \
        apt-get install -y python3 python3-pip && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy and install dependencies
    COPY requirements.txt ./
    RUN pip3 install --upgrade pip && \
        pip3 install --no-deps -r requirements.txt
    
    # Copy application code
    COPY main.py ./
    
    # Expose the port the app runs on
    EXPOSE 8080
    
    # Start the FastAPI server
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]