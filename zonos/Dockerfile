# PyTorch 기반 이미지 사용 (CUDA 지원 버전)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 필수 패키지 설치
RUN apt update && \
    apt install -y git espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Zonos 저장소 클론 및 폴더 이동
RUN git clone https://github.com/Zyphra/Zonos.git /Zonos
WORKDIR /Zonos

# uv 설치
RUN pip install uv

# 프로젝트 의존성 설치
RUN uv pip install --system -e . && uv pip install --system -e .[compile]

# /Zonos 위치에 main.py 복사 (같은 디렉토리에 있는 경우)
COPY main.py /Zonos/main.py

# 컨테이너에서 열어줄 포트
EXPOSE 8080

# Uvicorn 실행 명령어
CMD ["uv","run","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

#docker build -t zonos .
