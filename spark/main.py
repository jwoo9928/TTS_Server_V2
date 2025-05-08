import asyncio
import itertools
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# MLX-Audio TTS 라이브러리 직접 임포트
from mlx_audio.tts.generate import generate_audio

# -------------------- Configuration --------------------
MODEL_NAME = "mlx-community/Spark-TTS-0.5B-fp16"
DEFAULT_MODEL_DIR = None   # 기본 Hugging Face hub 경로 사용
DEVICE_ID = 0
N_WORKERS = 4  # 병렬 워커 수
MAX_BATCH_SIZE = 4
MAX_WAIT_TIME = 0.01  # 10ms

# -------------------- Request/Response Schemas --------------------
class TTSRequest(BaseModel):
    text: str
    save_dir: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_speech_path: Optional[str] = None

class TTSResponse(BaseModel):
    audio: bytes
    sample_rate: int

# -------------------- 워커 정의 --------------------
class TTSWorker:
    def __init__(self):
        pass  # 상태 없음

    def run_batch(
        self,
        batch: List[Tuple[str, Optional[str], Optional[str], Optional[str]]]
    ) -> List[Tuple[bytes, int]]:
        outputs = []
        for text, prompt_text, prompt_path, save_dir in batch:
            # generate_audio 호출 시 옵션 매핑
            wav_np = generate_audio(
                text=text,
                model_path=MODEL_NAME,
                device=DEVICE_ID,
                save_dir=save_dir,
                prompt_text=prompt_text,
                prompt_speech_path=prompt_path,
                join_audio=True,
                audio_format="wav",
                sample_rate=16000,
                verbose=False
            )
            # numpy array to bytes
            audio_bytes = wav_np.tobytes()
            outputs.append((audio_bytes, 16000))
        return outputs

# -------------------- 글로벌 배칭 & 서버 --------------------
app = FastAPI()
workers = [TTSWorker() for _ in range(N_WORKERS)]
rr = itertools.cycle(range(N_WORKERS))
queue: List[Tuple[Tuple[str, Optional[str], Optional[str], Optional[str]], asyncio.Future]] = []
lock = asyncio.Lock()
evt = asyncio.Event()

@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_handler())

async def batch_handler():
    while True:
        await evt.wait()
        await asyncio.sleep(MAX_WAIT_TIME)
        async with lock:
            batch = list(queue)
            queue.clear()
            evt.clear()
            
        if not batch:  # 빈 배치 체크
            continue
            
        inputs, futures = zip(*batch)
        worker = workers[next(rr)]
        try:
            results = await run_in_threadpool(worker.run_batch, list(inputs))
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
            continue
        for (audio, sr), fut in zip(results, futures):
            if not fut.done():
                fut.set_result({"audio": audio, "sample_rate": sr})

@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    if not req.text:
        raise HTTPException(400, "text is required")
    item = (
        req.text,
        req.prompt_text,
        req.prompt_speech_path,
        req.save_dir
    )
    fut = asyncio.get_event_loop().create_future()
    async with lock:
        queue.append((item, fut))
        if len(queue) >= MAX_BATCH_SIZE or len(queue) == 1:
            evt.set()
    res = await fut
    return TTSResponse(audio=res["audio"], sample_rate=res["sample_rate"])

# uvicorn 실행 예시
# uvicorn optimized_spark_tts_server:app --host 0.0.0.0 --port 8000 --workers 1