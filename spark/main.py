import asyncio
import itertools
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import pipeline
from starlette.concurrency import run_in_threadpool

# -------------------- Configuration --------------------
MODEL_NAME = "mlx-community/Spark-TTS-0.5B-fp16"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_WORKERS = 4                        # Number of model instances
MAX_BATCH_SIZE = 4                   # Max requests per batch
MAX_WAIT_TIME = 0.01                 # Max wait time (seconds) for batching

# -------------------- Request/Response Schemas --------------------
class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio: bytes                     # Raw WAV bytes
    sample_rate: int

# -------------------- Worker Definition --------------------
class TTSWorker:
    def __init__(self, device: torch.device):
        self.device = device
        # Initialize huggingface TTS pipeline in FP16
        self.pipe = pipeline(
            task="text-to-speech",
            model=MODEL_NAME,
            torch_dtype=torch.float16,
            device=self.device.index if self.device.type == "cuda" else -1
        )

    def run_batch(self, texts: List[str]) -> List[Tuple[bytes, int]]:
        # Blocking call: generate audio batch
        results = self.pipe(texts)
        return [(r["audio"], r["sample_rate"]) for r in results]

# -------------------- Global Batcher --------------------
app = FastAPI()

# Pool of workers & round-robin selector
workers = []
round_robin = itertools.cycle(range(N_WORKERS))

# Queues and synchronization
batch_queue: List[Tuple[str, asyncio.Future]] = []
queue_lock = asyncio.Lock()
batch_event = asyncio.Event()

# -------------------- Background Batching Task --------------------
@app.on_event("startup")
async def startup_event():
    # Initialize workers
    for _ in range(N_WORKERS):
        workers.append(TTSWorker(DEVICE))
    # Start batch handler
    asyncio.create_task(batch_handler())

async def batch_handler():
    while True:
        # Wait until batch_event is set
        await batch_event.wait()
        # Allow time to collect additional requests
        await asyncio.sleep(MAX_WAIT_TIME)

        async with queue_lock:
            batch = list(batch_queue)
            batch_queue.clear()
            batch_event.clear()

        if not batch:
            continue

        texts, futures = zip(*[(text, fut) for text, fut in batch])
        worker_idx = next(round_robin)
        worker = workers[worker_idx]

        # Offload blocking inference to thread pool
        try:
            results = await run_in_threadpool(worker.run_batch, list(texts))
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
            continue

        # Map results back to futures
        for (audio, sr), fut in zip(results, futures):
            if not fut.done():
                fut.set_result({"audio": audio, "sample_rate": sr})

# -------------------- API Endpoint --------------------
@app.post("/tts", response_model=TTSResponse)
async def tts_endpoint(req: TTSRequest):
    # Reject too long texts
    if len(req.text) > 500:
        raise HTTPException(status_code=400, detail="Text length exceeds limit.")

    fut = asyncio.get_event_loop().create_future()
    async with queue_lock:
        batch_queue.append((req.text, fut))
        # Trigger batch processing when full or first request
        if len(batch_queue) >= MAX_BATCH_SIZE or len(batch_queue) == 1:
            batch_event.set()

    # Await result
    res = await fut
    return TTSResponse(audio=res["audio"], sample_rate=res["sample_rate"])

# -------------------- Run with Uvicorn --------------------
# uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
