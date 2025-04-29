import asyncio
import io
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from kokoro import KPipeline
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 파이프라인 인스턴스를 저장할 전역 변수
pipeline = None
# CUDA 가용성에 따른 디바이스 결정
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"사용 디바이스: {device}")

# 병렬 처리를 위한 ThreadPoolExecutor 생성
# 최대 워커 수는 시스템 사양에 맞게 조정
thread_pool = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 모델 로드 (싱글톤 패턴)
    global pipeline
    logger.info("Kokoro 파이프라인 로드 중...")
    try:
        # 파이프라인 초기화 - 필요시 lang_code 조정 ('a'는 미국 영어)
        # 결정된 디바이스를 파이프라인에 전달
        pipeline = KPipeline(lang_code='a', device=device)
        logger.info("Kokoro 파이프라인 로드 완료.")
    except Exception as e:
        logger.error(f"Kokoro 파이프라인 로드 실패: {e}", exc_info=True)
        pipeline = None
    yield
    # 앱 종료 시 리소스 정리
    logger.info("종료 중...")
    global thread_pool
    thread_pool.shutdown()
    pipeline = None  # 메모리 해제

app = FastAPI(lifespan=lifespan)

class TTSItem(BaseModel):
    text: str
    voice: str = 'af_heart'  # 기본 음성
    speed: float = Field(1.0, ge=0.5, le=2.0)

class TTSBatchRequest(BaseModel):
    items: List[TTSItem]
    split_pattern: str = r'\n+'

class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_heart'  # 기본 음성
    speed: float = Field(1.0, ge=0.5, le=2.0)
    split_pattern: str = r'\n+'

def generate_audio_sync(text: str, voice: str, speed: float, split_pattern: str):
    """Kokoro를 사용하여 오디오를 생성하는 동기 함수."""
    if pipeline is None:
        raise RuntimeError("Kokoro 파이프라인이 초기화되지 않았습니다.")
    
    try:
        logger.info(f"오디오 생성 중: 음성={voice}, 속도={speed}")
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
        
        # 제너레이터에서 모든 오디오 청크 수집
        all_audio_chunks = []
        for i, (gs, ps, audio_chunk) in enumerate(generator):
            logger.debug(f"청크 {i} 생성됨")
            all_audio_chunks.append(audio_chunk)
        
        if not all_audio_chunks:
            logger.warning("생성된 오디오 청크가 없습니다.")
            return None

        # 청크 병합
        if isinstance(all_audio_chunks[0], torch.Tensor):
            full_audio = torch.cat(all_audio_chunks)
            full_audio_np = full_audio.cpu().numpy()
        else:
            full_audio_np = np.concatenate(all_audio_chunks)

        # 메모리 내 WAV 파일로 저장
        buffer = io.BytesIO()
        sf.write(buffer, full_audio_np, 24000, format='WAV')
        buffer.seek(0)
        logger.info("오디오 생성 완료.")
        return buffer
    except Exception as e:
        logger.error(f"오디오 생성 중 오류 발생: {e}", exc_info=True)
        raise

async def process_batch_item(item: TTSItem, split_pattern: str) -> Optional[io.BytesIO]:
    """배치 요청의 각 항목을 처리하는 비동기 함수"""
    try:
        return await asyncio.to_thread(
            generate_audio_sync,
            item.text,
            item.voice,
            item.speed,
            split_pattern
        )
    except Exception as e:
        logger.error(f"배치 항목 처리 중 오류 발생: {e}", exc_info=True)
        return None

@app.post("/tts")
async def text_to_speech(request_data: TTSRequest, background_tasks: BackgroundTasks):
    """
    TTS 오디오를 생성하고 전체 WAV 파일을 반환합니다.
    asyncio.to_thread를 사용하여 요청을 동시에 처리합니다.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS 서비스 사용 불가: 모델이 로드되지 않았습니다.")

    try:
        # 동기 생성 함수를 별도 스레드에서 실행
        audio_buffer = await asyncio.to_thread(
            generate_audio_sync,
            request_data.text,
            request_data.voice,
            request_data.speed,
            request_data.split_pattern
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

@app.post("/tts-batch")
async def text_to_speech_batch(request_data: TTSBatchRequest):
    """
    배치 처리를 위한 엔드포인트.
    여러 TTS 요청을 병렬로 처리하고 ZIP 파일로 반환합니다.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS 서비스 사용 불가: 모델이 로드되지 않았습니다.")

    try:
        # 모든 배치 항목을 비동기적으로 처리
        tasks = [
            process_batch_item(item, request_data.split_pattern)
            for item in request_data.items
        ]
        
        # 모든 태스크 동시 실행
        results = await asyncio.gather(*tasks)
        
        # 실패한 항목 체크
        if all(result is None for result in results):
            raise HTTPException(status_code=500, detail="모든 배치 항목 처리에 실패했습니다.")
        
        # 여러 개의 오디오 파일을 ZIP으로 압축
        import zipfile
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, audio_buffer in enumerate(results):
                if audio_buffer:
                    audio_buffer.seek(0)
                    zip_file.writestr(f"audio_{i}.wav", audio_buffer.read())
        
        # ZIP 파일 반환
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=audio_batch.zip"}
        )
        
    except Exception as e:
        logger.error(f"/tts-batch 엔드포인트에서 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"배치 처리 오류: {str(e)}")

@app.post("/tts-stream")
async def text_to_speech_stream(request_data: TTSRequest):
    """
    TTS 오디오를 생성하고 생성되는 대로 WAV 오디오 청크를 스트리밍합니다.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS 서비스 사용 불가: 모델이 로드되지 않았습니다.")

    async def audio_stream_generator():
        try:
            # 비동기 스레드를 사용하여 블로킹 kokoro 제너레이터 실행
            async def process_chunks():
                generator = pipeline(
                    request_data.text, 
                    voice=request_data.voice, 
                    speed=request_data.speed, 
                    split_pattern=request_data.split_pattern
                )
                
                # WAV 헤더 먼저 전송
                yield b'RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\\\0\0\0\x01\x18\0data\0\0\0\0'
                
                # 각 청크 처리
                for i, (gs, ps, audio_chunk) in enumerate(generator):
                    logger.debug(f"청크 {i} 스트리밍 중")
                    # 필요시 텐서를 numpy로 변환
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy()
                    else:
                        audio_np = audio_chunk
                        
                    # 바이트로 변환 후 yield
                    audio_bytes = audio_np.tobytes()
                    yield audio_bytes
                    
                logger.info("스트리밍 생성 완료.")
            
            # 별도 스레드에서 제너레이터 시작 및 결과 스트리밍
            chunks_generator = process_chunks()
            async for chunk in chunks_generator:
                yield chunk
                
        except Exception as e:
            logger.error(f"스트리밍 오디오 생성 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"스트리밍 오류: {str(e)}")

    try:
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
        )
    except Exception as e:
        logger.error(f"/tts-stream 엔드포인트에서 처리되지 않은 예외 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)