# app.py
import os
import json
import time
import asyncio
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import tempfile
import logging
from functools import partial
import uuid

import outetts
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tts-server")

# 환경 변수 설정
CPU_CORES = os.cpu_count() or 4
WORKER_PROCESSES = max(1, CPU_CORES - 1)  # 하나는 메인 프로세스용으로 남겨둠
TTS_OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "/tmp/tts_outputs")

# 출력 디렉토리가 없으면 생성
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# 스피커 캐시 디렉토리
SPEAKER_CACHE_DIR = os.environ.get("SPEAKER_CACHE_DIR", "/app/speakers")
os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)

# 애플리케이션 초기화
app = FastAPI(title="High Performance TTS Server")

# TTS 인터페이스 설정을 위한 모델
class TTSModelConfig(BaseModel):
    model: str = "VERSION_1_0_SIZE_1B"
    backend: str = "LLAMACPP"
    quantization: str = "Q4_K_S"
    use_cuda: bool = False

# TTS 요청을 위한 모델
class TTSRequest(BaseModel):
    text: str
    speaker_name: Optional[str] = "EN-FEMALE-1-NEUTRAL"
    temperature: float = 0.4
    repetition_penalty: float = 1.1
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05
    mirostat: bool = False
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    output_format: str = "wav"

# 배치 TTS 요청을 위한 모델
class BatchTTSRequest(BaseModel):
    texts: List[str]
    speaker_name: Optional[str] = "EN-FEMALE-1-NEUTRAL"
    temperature: float = 0.4
    repetition_penalty: float = 1.1
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05
    mirostat: bool = False
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    output_format: str = "wav"

class ProcessManager:
    """TTS 작업을 관리하기 위한 프로세스 매니저"""
    
    def __init__(self, max_workers=WORKER_PROCESSES, config=None):
        self.max_workers = max_workers
        self.config = config or TTSModelConfig()
        
        # 워커 프로세스 시작 시 모델 로드를 위한 초기화 함수
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=self._initialize_worker,
            initargs=(self.config,)
        )
        logger.info(f"프로세스 매니저 초기화: 최대 워커 수 {max_workers}")
    
    @staticmethod
    def _initialize_worker(config):
        """워커 프로세스 초기화 함수 - 모델 미리 로드"""
        pid = os.getpid()
        logger.info(f"워커 프로세스 {pid} 초기화 중: TTS 모델 사전 로드")
        
        # CUDA 설정
        if config.use_cuda:
            os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        
        # 모델 초기화 및 로드
        model_enum = getattr(outetts.Models, config.model)
        backend_enum = getattr(outetts.Backend, config.backend)
        quantization_enum = getattr(outetts.LlamaCppQuantization, config.quantization)
        
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=model_enum,
                backend=backend_enum,
                quantization=quantization_enum
            )
        )
        
        # 프로세스 전역 변수에 인터페이스 저장
        global _tts_interface
        _tts_interface = interface
        
        logger.info(f"워커 프로세스 {pid}에서 TTS 모델 로드 완료")
    
    def get_tts_interface(self):
        """현재 프로세스의 TTS 인터페이스 가져오기"""
        pid = os.getpid()
        global _tts_interface
        
        if '_tts_interface' in globals():
            logger.debug(f"프로세스 {pid}에서 기존 TTS 인터페이스 반환")
            return _tts_interface
        
        # 혹시 초기화되지 않은 경우를 위한 폴백
        logger.warning(f"프로세스 {pid}에서 TTS 인터페이스가 초기화되지 않음. 동적 로드 중...")
        self._initialize_worker(self.config)
        return _tts_interface
    
    def submit_tts_task(self, tts_request, speaker_path=None):
        """TTS 작업을 프로세스 풀에 제출"""
        return self.executor.submit(
            process_tts_request, 
            tts_request, 
            speaker_path
        )
    
    def shutdown(self):
        """프로세스 풀 종료"""
        self.executor.shutdown()

# 프로세스 매니저 인스턴스 생성
process_manager = ProcessManager()

# 전역 스피커 캐시
speaker_cache = {}

def load_speaker(interface, speaker_name):
    """스피커 프로필 로드"""
    if speaker_name.startswith("EN-") or speaker_name.startswith("FR-"):
        # 기본 스피커 사용
        return interface.load_default_speaker(speaker_name)
    else:
        # 사용자 정의 스피커 파일 경로
        speaker_path = os.path.join(SPEAKER_CACHE_DIR, f"{speaker_name}.json")
        if os.path.exists(speaker_path):
            return interface.load_speaker(speaker_path)
        else:
            raise ValueError(f"스피커 프로필을 찾을 수 없습니다: {speaker_name}")

def process_tts_request(tts_request, speaker_path=None):
    """TTS 요청 처리 (별도 프로세스에서 실행)"""
    try:
        pid = os.getpid()
        start_time = time.time()
        logger.info(f"프로세스 {pid}에서 TTS 요청 처리 시작: {tts_request.text[:30]}...")
        
        # 이미 로드된 TTS 인터페이스 가져오기
        interface = process_manager.get_tts_interface()
        
        # 스피커 로드
        if speaker_path and os.path.exists(speaker_path):
            speaker = interface.load_speaker(speaker_path)
        else:
            speaker = load_speaker(interface, tts_request.speaker_name)
        
        # 샘플러 설정
        sampler_config = outetts.SamplerConfig(
            temperature=tts_request.temperature,
            repetition_penalty=tts_request.repetition_penalty,
            top_k=tts_request.top_k,
            top_p=tts_request.top_p,
            min_p=tts_request.min_p,
            mirostat=tts_request.mirostat,
            mirostat_tau=tts_request.mirostat_tau,
            mirostat_eta=tts_request.mirostat_eta
        )
        
        # TTS 생성
        output = interface.generate(
            config=outetts.GenerationConfig(
                text=tts_request.text,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=speaker,
                sampler_config=sampler_config,
            )
        )
        
        # 출력 파일 저장
        output_filename = f"{uuid.uuid4()}.{tts_request.output_format}"
        output_path = os.path.join(TTS_OUTPUT_DIR, output_filename)
        output.save(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"프로세스 {pid}에서 TTS 처리 완료: {elapsed_time:.2f}초 소요")
        
        return {
            "success": True,
            "file_path": output_path,
            "processing_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"TTS 처리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

async def process_batch_with_semaphore(sem, tts_request, speaker_path=None):
    """세마포어를 사용하여 비동기적으로 TTS 작업 처리"""
    async with sem:
        # 함수를 실행할 이벤트 루프
        loop = asyncio.get_event_loop()
        
        # 프로세스 풀에 작업 제출
        return await loop.run_in_executor(
            None,  # 기본 실행자 사용
            process_manager.submit_tts_task,
            tts_request,
            speaker_path
        )

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 이벤트"""
    logger.info(f"TTS 서버 시작: CPU 코어 {CPU_CORES}개, 작업자 프로세스 {WORKER_PROCESSES}개")
    
@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 이벤트"""
    logger.info("TTS 서버 종료 중...")
    process_manager.shutdown()

@app.get("/")
async def read_root():
    """루트 엔드포인트"""
    return {"status": "running", "cores": CPU_CORES, "workers": WORKER_PROCESSES}

@app.post("/tts")
async def create_tts(tts_request: TTSRequest):
    """단일 TTS 요청 처리 엔드포인트"""
    try:
        # TTS 요청 처리를 프로세스 풀에 제출
        future = process_manager.submit_tts_task(tts_request)
        
        # 결과 대기
        result = future.result()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 파일 응답 반환
        return FileResponse(
            path=result["file_path"],
            media_type=f"audio/{tts_request.output_format}",
            filename=f"tts_output.{tts_request.output_format}"
        )
    except Exception as e:
        logger.error(f"TTS 요청 처리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_tts")
async def create_batch_tts(batch_request: BatchTTSRequest):
    """배치 TTS 요청 처리 엔드포인트"""
    try:
        # 동시 처리 수를 제한하는 세마포어
        # CPU 코어 수에 맞게 조정
        sem = asyncio.Semaphore(WORKER_PROCESSES)
        
        # 각 텍스트에 대한 TTS 요청 생성
        tasks = []
        for text in batch_request.texts:
            tts_request = TTSRequest(
                text=text,
                speaker_name=batch_request.speaker_name,
                temperature=batch_request.temperature,
                repetition_penalty=batch_request.repetition_penalty,
                top_k=batch_request.top_k,
                top_p=batch_request.top_p,
                min_p=batch_request.min_p,
                mirostat=batch_request.mirostat,
                mirostat_tau=batch_request.mirostat_tau,
                mirostat_eta=batch_request.mirostat_eta,
                output_format=batch_request.output_format
            )
            
            # 비동기 작업 생성
            task = process_batch_with_semaphore(sem, tts_request)
            tasks.append(task)
        
        # 모든 작업 동시 실행
        futures = await asyncio.gather(*tasks)
        
        # 결과 대기 및 수집
        results = []
        for future in futures:
            result = await asyncio.wrap_future(future)
            results.append(result)
        
        # 결과 반환
        file_paths = []
        for i, result in enumerate(results):
            if result["success"]:
                file_paths.append({
                    "index": i,
                    "file_path": result["file_path"],
                    "processing_time": result["processing_time"]
                })
            else:
                file_paths.append({
                    "index": i,
                    "error": result["error"]
                })
        
        return {"results": file_paths}
    except Exception as e:
        logger.error(f"배치 TTS 요청 처리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_speaker")
async def upload_speaker(
    speaker_name: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """새로운 스피커 프로필 업로드 엔드포인트"""
    try:
        # 오디오 파일 임시 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file.close()
            
            # 이미 로드된 TTS 인터페이스 사용
            interface = process_manager.get_tts_interface()
            
            # 스피커 프로필 생성
            speaker = interface.create_speaker(temp_file.name)
            
            # 스피커 프로필 저장
            speaker_path = os.path.join(SPEAKER_CACHE_DIR, f"{speaker_name}.json")
            interface.save_speaker(speaker, speaker_path)
            
            return {"status": "success", "speaker_name": speaker_name}
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    except Exception as e:
        logger.error(f"스피커 업로드 오류: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speakers")
async def list_speakers():
    """사용 가능한 스피커 목록 반환 엔드포인트"""
    try:
        # 기본 스피커
        default_speakers = [
            "EN-FEMALE-1-NEUTRAL", "EN-FEMALE-2-NEUTRAL", 
            "EN-MALE-1-NEUTRAL", "EN-MALE-2-NEUTRAL",
            "FR-FEMALE-1-NEUTRAL", "FR-MALE-1-NEUTRAL"
        ]
        
        # 사용자 정의 스피커
        custom_speakers = []
        for filename in os.listdir(SPEAKER_CACHE_DIR):
            if filename.endswith(".json"):
                custom_speakers.append(filename[:-5])  # .json 확장자 제거
        
        return {
            "default_speakers": default_speakers,
            "custom_speakers": custom_speakers
        }
    except Exception as e:
        logger.error(f"스피커 목록 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/speakers/{speaker_name}")
async def delete_speaker(speaker_name: str):
    """스피커 프로필 삭제 엔드포인트"""
    try:
        # 기본 스피커는 삭제 불가
        default_speakers = [
            "EN-FEMALE-1-NEUTRAL", "EN-FEMALE-2-NEUTRAL", 
            "EN-MALE-1-NEUTRAL", "EN-MALE-2-NEUTRAL",
            "FR-FEMALE-1-NEUTRAL", "FR-MALE-1-NEUTRAL"
        ]
        
        if speaker_name in default_speakers:
            raise HTTPException(status_code=400, detail="기본 스피커는 삭제할 수 없습니다")
        
        # 스피커 파일 경로
        speaker_path = os.path.join(SPEAKER_CACHE_DIR, f"{speaker_name}.json")
        
        # 파일 존재 여부 확인
        if not os.path.exists(speaker_path):
            raise HTTPException(status_code=404, detail=f"스피커 '{speaker_name}'을 찾을 수 없습니다")
        
        # 파일 삭제
        os.remove(speaker_path)
        
        return {"status": "success", "message": f"스피커 '{speaker_name}'이 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"스피커 삭제 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 환경 변수에서 포트 설정 (기본값: 8080)
    port = int(os.environ.get("PORT", 8080))
    
    # 서버 실행
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port,
        workers=1,  # FastAPI 자체 워커 수 (내부적으로 ProcessPoolExecutor를 사용하므로 1로 설정)
        log_level="info"
    )