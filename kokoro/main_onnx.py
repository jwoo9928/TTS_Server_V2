# main.py
import asyncio
import io
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
# kokoro_onnx 임포트
from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수로 모델 인스턴스 저장
kokoro = None
tokenizer = None
SUPPORTED_LANGUAGES = ["en-us"]  # 지원하는 언어 목록

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작시 모델 로드
    global kokoro, tokenizer
    logger.info("Kokoro ONNX 모델 로딩 중...")
    try:
        tokenizer = Tokenizer()
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        logger.info("Kokoro ONNX 모델 로딩 완료.")
    except Exception as e:
        logger.error(f"Kokoro ONNX 모델 로딩 실패: {e}", exc_info=True)
        kokoro = None
        tokenizer = None
    yield
    # 종료시 리소스 정리
    logger.info("서버 종료 중...")
    kokoro = None
    tokenizer = None

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_sky'  # 기본 보이스
    language: str = 'en-us'  # 기본 언어
    speed: float = 1.0
    blend_voice_name: str = None  # 블렌딩용 보이스 (선택사항)

def generate_audio_sync(text: str, voice: str, language: str, speed: float, blend_voice_name: str = None):
    """kokoro_onnx를 사용하여 오디오 생성하는 동기 함수."""
    if kokoro is None or tokenizer is None:
        raise RuntimeError("Kokoro ONNX 모델이 초기화되지 않았습니다.")
    
    try:
        logger.info(f"오디오 생성 중: 보이스={voice}, 언어={language}, 스피드={speed}")
        
        # 텍스트를 음소로 변환
        phonemes = tokenizer.phonemize(text, lang=language)
        
        # 보이스 블렌딩 처리
        if blend_voice_name:
            logger.info(f"보이스 블렌딩: {voice}와 {blend_voice_name}")
            first_voice = kokoro.get_voice_style(voice)
            second_voice = kokoro.get_voice_style(blend_voice_name)
            voice = np.add(first_voice * (50 / 100), second_voice * (50 / 100))
        
        # 오디오 생성
        samples, sample_rate = kokoro.create(
            phonemes, voice=voice, speed=speed, is_phonemes=True
        )
        
        # 오디오를 WAV 파일로 변환
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV')
        buffer.seek(0)
        logger.info("오디오 생성 완료.")
        return buffer
    except Exception as e:
        logger.error(f"오디오 생성 중 오류 발생: {e}", exc_info=True)
        raise

@app.post("/tts")
async def text_to_speech(request_data: TTSRequest):
    """
    TTS 오디오를 생성하고 전체 WAV 파일을 반환합니다.
    asyncio.to_thread를 사용하여 동시 요청을 처리합니다.
    """
    if kokoro is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="TTS 서비스 사용 불가: 모델이 로드되지 않았습니다.")

    try:
        # 동기 생성 함수를 별도 스레드에서 실행
        audio_buffer = await asyncio.to_thread(
            generate_audio_sync,
            request_data.text,
            request_data.voice,
            request_data.language,
            request_data.speed,
            request_data.blend_voice_name
        )
        
        if audio_buffer is None:
            raise HTTPException(status_code=500, detail="TTS 생성 실패: 오디오가 생성되지 않았습니다.")

        # 오디오 스트림 반환
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"TTS 서비스 사용 불가: {str(e)}")
    except Exception as e:
        logger.error(f"/tts 엔드포인트에서 처리되지 않은 예외 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")

@app.get("/voices")
async def get_voices():
    """
    사용 가능한 모든 보이스 목록을 반환합니다.
    """
    if kokoro is None:
        raise HTTPException(status_code=503, detail="TTS 서비스 사용 불가: 모델이 로드되지 않았습니다.")
    
    try:
        voices = sorted(kokoro.get_voices())
        return {"voices": voices}
    except Exception as e:
        logger.error(f"/voices 엔드포인트에서 처리되지 않은 예외 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")

@app.get("/languages")
async def get_languages():
    """
    지원되는 언어 목록을 반환합니다.
    """
    return {"languages": SUPPORTED_LANGUAGES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)