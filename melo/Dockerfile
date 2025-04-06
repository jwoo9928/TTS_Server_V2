# 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치 요구사항 파일 복사
COPY requirements.txt .

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# melo TTS 설치
RUN pip install --no-cache-dir git+https://github.com/melothebig/melo-tts.git

# 서버 코드 복사
COPY main.py .

# 포트 노출
EXPOSE 8080

# 서버 실행 명령
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]