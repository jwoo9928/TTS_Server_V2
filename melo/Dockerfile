FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# MeloTTS 클론 + 설치
RUN git clone https://github.com/myshell-ai/MeloTTS.git /app/MeloTTS
WORKDIR /app/MeloTTS
RUN pip install --use-deprecated=legacy-resolver -e . && python -m unidic download

# 다시 app으로 작업 디렉토리 변경
WORKDIR /app

# 나머지 파일 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
