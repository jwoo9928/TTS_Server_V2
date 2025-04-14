import os
import io
import time
import uuid
import torch
import torchaudio
import threading
import multiprocessing
from typing import Dict, Optional, List, Union
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# 캐시 디렉토리 설정
CACHE_DIR = os.path.join(os.getcwd(), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 각 프로세스마다 고유한 모델 인스턴스를 가지도록 함
MODEL_LOCK = threading.Lock()
MODEL_INSTANCES = {}

# 요청 큐와 프로세스 풀 설정
REQUEST_QUEUE = multiprocessing.Queue()
MAX_WORKERS = min(multiprocessing.cpu_count(), 4)  # CPU 코어 수와 원하는 최대 동시 처리 수 중 작은 값 선택

class TTSRequest(BaseModel):
    text: str
    language: str = "en-us"
    speaker_audio_url: Optional[str] = None
    cached_speaker_id: Optional[str] = None

class SpeakerCache:
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()
    
    def get(self, speaker_id: str) -> Optional[torch.Tensor]:
        with self.lock:
            return self.cache.get(speaker_id)
    
    def set(self, speaker_id: str, embedding: torch.Tensor):
        with self.lock:
            self.cache[speaker_id] = embedding
            
    def has(self, speaker_id: str) -> bool:
        with self.lock:
            return speaker_id in self.cache

speaker_cache = SpeakerCache()

def get_or_create_model():
    """현재 프로세스에 모델이 없으면 로드하고, 있으면 재사용"""
    pid = os.getpid()
    with MODEL_LOCK:
        if pid not in MODEL_INSTANCES:
            print(f"프로세스 {pid}에 모델 로드 중...")
            # 메모리를 효율적으로 사용하기 위해 필요할 때만 모델 로드
            model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
            MODEL_INSTANCES[pid] = model
            print(f"프로세스 {pid}에 모델 로드 완료")
        return MODEL_INSTANCES[pid]

def process_tts_request(request_id: str, text: str, language: str, speaker_embedding: torch.Tensor):
    """TTS 생성 처리를 수행"""
    model = get_or_create_model()
    
    # 캐시 파일 경로 생성
    output_path = os.path.join(CACHE_DIR, f"{request_id}.wav")
    
    # TTS 생성
    torch.manual_seed(int(time.time()))  # 매번 다른 결과를 위한 시드 설정
    cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=language)
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes).cpu()
    
    # 파일 저장
    torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
    return output_path

def get_speaker_embedding(model, speaker_audio_path: str):
    """스피커 오디오로부터 임베딩 생성"""
    wav, sampling_rate = torchaudio.load(speaker_audio_path)
    return model.make_speaker_embedding(wav, sampling_rate)

def download_audio(url: str, save_path: str):
    """URL에서 오디오 다운로드"""
    import requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return True
    return False

# FastAPI 앱 생성
app = FastAPI(title="Zonos TTS API Server")

# 스레드 풀과 프로세스 풀 생성
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)

@app.post("/tts")
async def tts_endpoint(request: TTSRequest, background_tasks: BackgroundTasks):
    """TTS API 엔드포인트"""
    try:
        request_id = str(uuid.uuid4())
        
        # 스피커 임베딩 처리
        speaker_embedding = None
        
        # 캐시된 스피커 ID 사용
        if request.cached_speaker_id and speaker_cache.has(request.cached_speaker_id):
            speaker_embedding = speaker_cache.get(request.cached_speaker_id)
        # 오디오 URL 제공된 경우
        elif request.speaker_audio_url:
            # 스피커 오디오 다운로드
            temp_audio_path = os.path.join(CACHE_DIR, f"speaker_{request_id}.wav")
            if not download_audio(request.speaker_audio_url, temp_audio_path):
                raise HTTPException(status_code=400, detail="스피커 오디오 다운로드 실패")
            
            # 스피커 임베딩 생성
            model = get_or_create_model()
            speaker_embedding = get_speaker_embedding(model, temp_audio_path)
            
            # 임베딩 캐시 (나중에 사용할 수 있도록)
            if request.cached_speaker_id:
                speaker_cache.set(request.cached_speaker_id, speaker_embedding)
                
            # 임시 파일 정리 (백그라운드로)
            background_tasks.add_task(lambda: os.remove(temp_audio_path) if os.path.exists(temp_audio_path) else None)
        else:
            # 기본 임베딩 사용 (텍스트만 있는 경우)
            model = get_or_create_model()
            default_audio_path = "assets/exampleaudio.mp3"  # 예시 오디오 경로
            speaker_embedding = get_speaker_embedding(model, default_audio_path)
        
        # TTS 처리를 프로세스 풀에서 실행
        output_path = await app.state.loop.run_in_executor(
            process_pool, 
            process_tts_request, 
            request_id, 
            request.text, 
            request.language, 
            speaker_embedding
        )
        
        # 파일 응답
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{request_id}.wav",
            background=background_tasks.add_task(lambda: os.remove(output_path) if os.path.exists(output_path) else None)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 처리 오류: {str(e)}")

@app.post("/tts-stream")
async def tts_stream_endpoint(request: TTSRequest):
    """스트리밍 TTS API 엔드포인트"""
    try:
        request_id = str(uuid.uuid4())
        
        # 스피커 임베딩 처리 (기본 로직은 /tts와 동일)
        speaker_embedding = None
        
        if request.cached_speaker_id and speaker_cache.has(request.cached_speaker_id):
            speaker_embedding = speaker_cache.get(request.cached_speaker_id)
        elif request.speaker_audio_url:
            temp_audio_path = os.path.join(CACHE_DIR, f"speaker_{request_id}.wav")
            if not download_audio(request.speaker_audio_url, temp_audio_path):
                raise HTTPException(status_code=400, detail="스피커 오디오 다운로드 실패")
            
            model = get_or_create_model()
            speaker_embedding = get_speaker_embedding(model, temp_audio_path)
            
            if request.cached_speaker_id:
                speaker_cache.set(request.cached_speaker_id, speaker_embedding)
                
            # 임시 파일 삭제 (스트리밍에서는 바로 삭제)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        else:
            model = get_or_create_model()
            default_audio_path = "assets/exampleaudio.mp3"
            speaker_embedding = get_speaker_embedding(model, default_audio_path)
        
        # 프로세스 풀에서 TTS 처리 실행
        output_path = await app.state.loop.run_in_executor(
            process_pool, 
            process_tts_request, 
            request_id, 
            request.text, 
            request.language, 
            speaker_embedding
        )
        
        # 스트리밍 응답 함수
        async def stream_audio():
            with open(output_path, "rb") as audio_file:
                chunk_size = 4096  # 적절한 청크 크기
                while True:
                    chunk = audio_file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            
            # 파일 스트리밍 후 삭제
            if os.path.exists(output_path):
                os.remove(output_path)
        
        return StreamingResponse(
            stream_audio(),
            media_type="audio/wav"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스트리밍 TTS 처리 오류: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행할 작업"""
    app.state.loop = asyncio.get_event_loop()
    
    # 캐시 디렉토리 정리
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"캐시 파일 정리 중 오류: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행할 작업"""
    # 스레드 풀과 프로세스 풀 정리
    thread_pool.shutdown(wait=False)
    process_pool.shutdown(wait=False)

if __name__ == "__main__":
    import asyncio
    
    # 필요한 asyncio import
    if not hasattr(app, "state") or not hasattr(app.state, "loop"):
        app.state = type('state', (), {})()
        app.state.loop = asyncio.get_event_loop()
    
    # 서버 시작
    uvicorn.run(app, host="0.0.0.0", port=8000)