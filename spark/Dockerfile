FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    soundfile \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    huggingface_hub

# Spark TTS 설치
# RUN git clone https://github.com/sparkandtemple/Spark-TTS.git && \
#     cd Spark-TTS && \
#     pip install -e .

# 애플리케이션 파일 복사
COPY main.py /app/

# 모델 및 결과 디렉토리 생성
RUN mkdir -p /app/pretrained_models /app/results

# Hugging Face에서 모델 다운로드
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')"

# 환경 변수 설정
ENV MODEL_DIR="/app/pretrained_models/Spark-TTS-0.5B" \
    DEVICE_ID=0 \
    RESULTS_DIR="/app/results"

# 포트 노출
EXPOSE 8000

# 진입점 명령
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]