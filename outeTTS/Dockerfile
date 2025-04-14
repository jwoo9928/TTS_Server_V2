FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CUDA 지원 옵션으로 outetts 설치
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install outetts --upgrade

# 스피커 캐시 디렉토리 생성
RUN mkdir -p /app/speakers
RUN mkdir -p /tmp/tts_outputs

# 애플리케이션 파일 복사
COPY app.py .

# 포트 설정
EXPOSE 8080

# 애플리케이션 실행
CMD ["python", "app.py"]