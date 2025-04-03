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
        pipeline = None
    yield
    # Clean up resources if needed during shutdown
    logger.info("Shutting down...")
    pipeline = None  # Release memory

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_heart'  # Default voice
    speed: float = 1.0
    split_pattern: str = r'\n+'

def generate_audio_sync(text: str, voice: str, speed: float, split_pattern: str):
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

        # Concatenate chunks
        if isinstance(all_audio_chunks[0], torch.Tensor):
            full_audio = torch.cat(all_audio_chunks)
            full_audio_np = full_audio.cpu().numpy()
        else:
            full_audio_np = np.concatenate(all_audio_chunks)

        # Save to an in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, full_audio_np, 24000, format='WAV')
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

        # Return the audio stream
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"TTS Service Unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unhandled exception in /tts endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/tts-stream")
async def text_to_speech_stream(request_data: TTSRequest):
    """
    Generates TTS audio and streams the WAV audio chunks as they become available.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")

    async def audio_stream_generator():
        try:
            # Use async thread to run the blocking kokoro generator
            async def process_chunks():
                generator = pipeline(
                    request_data.text, 
                    voice=request_data.voice, 
                    speed=request_data.speed, 
                    split_pattern=request_data.split_pattern
                )
                
                # Send WAV header first
                yield b'RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\\\0\0\0\x01\x18\0data\0\0\0\0'
                
                # Process each chunk
                for i, (gs, ps, audio_chunk) in enumerate(generator):
                    logger.debug(f"Streaming chunk {i}")
                    # Convert tensor to numpy if needed
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy()
                    else:
                        audio_np = audio_chunk
                        
                    # Convert to bytes and yield
                    audio_bytes = audio_np.tobytes()
                    yield audio_bytes
                    
                logger.info("Streaming generation complete.")
            
            # Start the generator in a separate thread and stream results
            chunks_generator = process_chunks()
            async for chunk in chunks_generator:
                yield chunk
                
        except Exception as e:
            logger.error(f"Error during streaming audio generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Streaming Error: {str(e)}")

    try:
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
        )
    except Exception as e:
        logger.error(f"Unhandled exception in /tts-stream endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)