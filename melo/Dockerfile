# 베이스 이미지 사용
FROM python:3.9-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# MeloTTS 클론
RUN git clone https://github.com/myshell-ai/MeloTTS.git

# MeloTTS 설치 (editable)
RUN pip install --no-cache-dir -e ./MeloTTS && \
    python -m unidic download

# 필요 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# main.py 복사
COPY main.py .

# 포트 노출
EXPOSE 8080

# 실행 명령어
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
