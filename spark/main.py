import os
import torch
import soundfile as sf
import logging
import platform
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

# Spark TTS 관련 임포트
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Spark TTS API", description="Spark TTS를 이용한 음성 합성 API")

# 전역 변수 설정
model = None
model_dir = os.environ.get("MODEL_DIR", "pretrained_models/Spark-TTS-0.5B")
device_id = int(os.environ.get("DEVICE_ID", 0))
results_dir = os.environ.get("RESULTS_DIR", "results")

# 결과 디렉토리 생성
os.makedirs(results_dir, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    prompt_text: Optional[str] = None
    gender: Optional[str] = None
    pitch: Optional[int] = None
    speed: Optional[int] = None

def initialize_model(model_dir=model_dir, device_id=device_id):
    """모델 초기화 함수"""
    logger.info(f"Loading model from: {model_dir}")

    # 플랫폼에 따른 적절한 디바이스 결정
    if platform.system() == "Darwin":
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{device_id}")
        logger.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # CUDA 지원 시스템
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Using CUDA device: {device}")
    else:
        # CPU 폴백
        device = torch.device("cpu")
        logger.info("GPU acceleration not available, using CPU")

    return SparkTTS(model_dir, device)

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 모델 로드"""
    global model
    try:
        model = initialize_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/")
def read_root():
    """루트 경로 API"""
    return {"status": "Spark TTS API is running", "model_dir": model_dir}

@app.post("/tts/")
async def text_to_speech(request: TTSRequest):
    """텍스트를 음성으로 변환하는 API"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 타임스탬프 기반 고유 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(results_dir, f"{timestamp}.wav")
        
        # pitch와 speed가 제공된 경우 LEVELS_MAP_UI로 매핑
        pitch_val = None
        speed_val = None
        
        if request.pitch is not None:
            if request.pitch < 1 or request.pitch > 5:
                raise HTTPException(status_code=400, detail="Pitch should be between 1 and 5")
            pitch_val = LEVELS_MAP_UI[request.pitch]
            
        if request.speed is not None:
            if request.speed < 1 or request.speed > 5:
                raise HTTPException(status_code=400, detail="Speed should be between 1 and 5")
            speed_val = LEVELS_MAP_UI[request.speed]
        
        # SparkTTS 추론 실행
        with torch.no_grad():
            wav = model.inference(
                request.text,
                prompt_speech=None,
                prompt_text=request.prompt_text if request.prompt_text and len(request.prompt_text) > 1 else None,
                gender=request.gender,
                pitch=pitch_val,
                speed=speed_val,
            )
            
            sf.write(save_path, wav, samplerate=16000)
        
        logger.info(f"Audio saved at: {save_path}")
        
        return FileResponse(save_path, media_type="audio/wav", filename=f"spark_tts_{timestamp}.wav")
    
    except Exception as e:
        logger.error(f"Error in TTS processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")

@app.post("/voice_clone/")
async def voice_clone(
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_speech: UploadFile = File(...)
):
    """음성 복제 API"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 업로드된 오디오 파일 임시 저장
        temp_audio_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await prompt_speech.read())
        
        # 타임스탬프 기반 고유 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(results_dir, f"{timestamp}.wav")
        
        # SparkTTS 추론 실행
        with torch.no_grad():
            wav = model.inference(
                text,
                prompt_speech=temp_audio_path,
                prompt_text=prompt_text if prompt_text and len(prompt_text) > 1 else None,
                gender=None,
                pitch=None,
                speed=None,
            )
            
            sf.write(save_path, wav, samplerate=16000)
        
        # 임시 파일 삭제
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        logger.info(f"Audio saved at: {save_path}")
        
        return FileResponse(save_path, media_type="audio/wav", filename=f"voice_clone_{timestamp}.wav")
    
    except Exception as e:
        logger.error(f"Error in voice cloning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.get("/health")
def health_check():
    """헬스 체크 API"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_dir": model_dir}

@app.get("/switch_device")
def switch_device(device_id: int):
    """디바이스 전환 API"""
    global model, device_id
    
    if device_id < 0:
        raise HTTPException(status_code=400, detail="Device ID should be non-negative")
    
    try:
        # 새 디바이스에 모델 다시 로드
        model = initialize_model(model_dir=model_dir, device_id=device_id)
        return {"message": f"Model reloaded on device {device_id}"}
    except Exception as e:
        logger.error(f"Error switching device: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to switch device: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)