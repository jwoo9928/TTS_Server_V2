import asyncio
import io
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from kokoro import KPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the pipeline instance
pipeline = None
# Determine device based on CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model during startup
    global pipeline
    logger.info("Loading Kokoro pipeline...")
    try:
        # Initialize pipeline - adjust lang_code if needed, 'a' for American English
        # Pass the determined device to the pipeline
        pipeline = KPipeline(lang_code='a', device=device)
        logger.info("Kokoro pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Kokoro pipeline: {e}", exc_info=True)
        # You might want to prevent the app from starting if the model fails to load
        # For now, we'll let it start but endpoints will fail.
        pipeline = None
    yield
    # Clean up resources if needed during shutdown (optional)
    logger.info("Shutting down...")
    pipeline = None # Release memory if possible

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_heart' # Default voice
    speed: float = 1.0
    split_pattern: str = r'\n+'

async def generate_audio_sync(text: str, voice: str, speed: float, split_pattern: str):
    """Synchronous function to generate audio using kokoro."""
    if pipeline is None:
        raise RuntimeError("Kokoro pipeline is not initialized.")
    try:
        logger.info(f"Generating audio for voice: {voice}, speed: {speed}")
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
        
        # Concatenate all audio chunks from the generator
        all_audio_chunks = []
        for i, (gs, ps, audio_chunk) in enumerate(generator):
            logger.debug(f"Generated chunk {i}")
            all_audio_chunks.append(audio_chunk)
        
        if not all_audio_chunks:
            logger.warning("No audio chunks generated.")
            return None

        # Concatenate chunks - assuming they are numpy arrays or similar
        full_audio = torch.cat(all_audio_chunks) if isinstance(all_audio_chunks[0], torch.Tensor) else \
                     numpy.concatenate(all_audio_chunks) # Add numpy import if needed

        # Save to an in-memory WAV file
        buffer = io.BytesIO()
        # Kokoro's default rate is 24000
        sf.write(buffer, full_audio.cpu().numpy() if isinstance(full_audio, torch.Tensor) else full_audio, 24000, format='WAV')
        buffer.seek(0)
        logger.info("Audio generation complete.")
        return buffer
    except Exception as e:
        logger.error(f"Error during audio generation: {e}", exc_info=True)
        raise

@app.post("/tts")
async def text_to_speech(request_data: TTSRequest):
    """
    Generates TTS audio and returns the full WAV file.
    Handles requests concurrently using asyncio.to_thread.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")

    try:
        # Run the synchronous generation function in a separate thread
        audio_buffer = await asyncio.to_thread(
            generate_audio_sync,
            request_data.text,
            request_data.voice,
            request_data.speed,
            request_data.split_pattern
        )
        if audio_buffer is None:
             raise HTTPException(status_code=500, detail="TTS generation failed: No audio produced.")

        return StreamingResponse(audio_buffer, media_type="audio/wav")
    except RuntimeError as e: # Catch specific runtime error from generate_audio_sync
         raise HTTPException(status_code=503, detail=f"TTS Service Unavailable: {e}")
    except Exception as e:
        logger.error(f"Unhandled exception in /tts endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


async def stream_audio_generator(text: str, voice: str, speed: float, split_pattern: str):
    """
    Asynchronous generator that yields audio chunks as they are produced by kokoro
    running in a separate thread.
    """
    if pipeline is None:
        raise RuntimeError("Kokoro pipeline is not initialized.")

    # Use an asyncio Queue to communicate between the thread and the async generator
    queue = asyncio.Queue()

    async def producer():
        """Runs the blocking kokoro generator in a thread and puts chunks in the queue."""
        try:
            logger.info(f"Streaming audio for voice: {voice}, speed: {speed}")
            # This is the blocking part
            generator = pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
            for i, (gs, ps, audio_chunk) in enumerate(generator):
                logger.debug(f"Streaming chunk {i}")
                # Convert chunk to bytes (WAV format) before putting in queue
                buffer = io.BytesIO()
                sf.write(buffer, audio_chunk.cpu().numpy() if isinstance(audio_chunk, torch.Tensor) else audio_chunk, 24000, format='WAV')
                await queue.put(buffer.getvalue())
            await queue.put(None) # Signal end of stream
            logger.info("Streaming generation complete.")
        except Exception as e:
            logger.error(f"Error during streaming audio generation: {e}", exc_info=True)
            await queue.put(e) # Put exception in queue to signal error

    # Start the producer thread
    producer_task = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(producer())))

    # Consume from the queue
    while True:
        item = await queue.get()
        if item is None:
            break # End of stream
        if isinstance(item, Exception):
            raise item # Propagate exception
        yield item
        queue.task_done()

    await producer_task # Ensure producer finishes


@app.post("/tts-stream")
async def text_to_speech_stream(request_data: TTSRequest):
    """
    Generates TTS audio and streams the WAV audio chunks as they become available.
    Handles requests concurrently.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")

    try:
        # Note: StreamingResponse expects an async generator or iterator
        audio_stream = stream_audio_generator(
            request_data.text,
            request_data.voice,
            request_data.speed,
            request_data.split_pattern
        )
        # Need to figure out how to stream WAV chunks properly.
        # The current stream_audio_generator yields complete mini-WAV files.
        # A better approach might be to yield raw audio samples and handle WAV header separately,
        # but that's more complex. For now, streaming mini-WAVs might work for some clients.
        # Alternatively, could stream raw PCM and require client to know format.
        # Let's stick with streaming mini-WAV chunks for simplicity first.
        return StreamingResponse(audio_stream, media_type="audio/wav")
        # Consider media_type="application/octet-stream" if clients handle raw chunks

    except RuntimeError as e:
         raise HTTPException(status_code=503, detail=f"TTS Service Unavailable: {e}")
    except Exception as e:
        logger.error(f"Unhandled exception in /tts-stream endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    import uvicorn
    # For local testing, might need to adjust host/port/reload
    uvicorn.run(app, host="0.0.0.0", port=8080)

# Add numpy import if needed for concatenation, check kokoro output type
# import numpy
