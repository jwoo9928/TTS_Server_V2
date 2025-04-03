import asyncio
import io
import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model during startup
    global model
    logger.info("Loading Zonos model...")
    try:
        # Initialize Zonos model
        # Default to hybrid model, can be changed via environment variable
        model_type = "hybrid"  # or "transformer"
        model_path = f"Zyphra/Zonos-v0.1-{model_type}"
        model = Zonos.from_pretrained(model_path, device=device)
        logger.info(f"Zonos model ({model_type}) loaded successfully on {device}.")
    except Exception as e:
        logger.error(f"Failed to load Zonos model: {e}", exc_info=True)
        model = None
    yield
    # Clean up resources during shutdown
    logger.info("Shutting down...")
    model = None  # Release memory

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    speaker_path: str  # Path to the reference audio file
    language: str = "en-us"  # Default language
    speed: float = 1.0

async def process_audio_reference(speaker_path):
    """Load and process speaker reference audio."""
    try:
        wav, sampling_rate = await asyncio.to_thread(torchaudio.load, speaker_path)
        return wav, sampling_rate
    except Exception as e:
        logger.error(f"Error loading reference audio: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load reference audio: {str(e)}")

def generate_audio_sync(text, speaker_wav, sampling_rate, language, speed):
    """Synchronous function to generate audio using Zonos."""
    if model is None:
        raise RuntimeError("Zonos model is not initialized.")
    
    try:
        logger.info(f"Generating audio for language: {language}, speed: {speed}")
        
        # Create speaker embedding from reference audio
        speaker = model.make_speaker_embedding(speaker_wav, sampling_rate)
        
        # Prepare conditioning
        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
        conditioning = model.prepare_conditioning(cond_dict)
        
        # Apply speed factor if not 1.0
        if speed != 1.0:
            conditioning = model.adjust_speed(conditioning, speed_factor=speed)
        
        # Generate audio codes
        codes = model.generate(conditioning)
        
        # Decode to audio waveform
        wavs = model.autoencoder.decode(codes)
        audio_np = wavs[0].cpu().numpy()
        
        # Save to an in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, model.autoencoder.sampling_rate, format='WAV')
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
    """
    if model is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")

    try:
        # Load speaker reference audio
        speaker_wav, sampling_rate = await process_audio_reference(request_data.speaker_path)
        
        # Run the synchronous generation function in a separate thread
        audio_buffer = await asyncio.to_thread(
            generate_audio_sync,
            request_data.text,
            speaker_wav,
            sampling_rate,
            request_data.language,
            request_data.speed
        )
        
        if audio_buffer is None:
            raise HTTPException(status_code=500, detail="TTS generation failed: No audio produced.")

        # Return the audio stream
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

@app.post("/tts-batch")
async def text_to_speech_batch(request_data: list[TTSRequest]):
    """
    Processes multiple TTS requests in parallel.
    Returns a ZIP file containing all generated audio files.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")
    
    import zipfile
    
    async def process_single_request(idx, req):
        try:
            speaker_wav, sampling_rate = await process_audio_reference(req.speaker_path)
            audio_buffer = await asyncio.to_thread(
                generate_audio_sync,
                req.text,
                speaker_wav,
                sampling_rate,
                req.language,
                req.speed
            )
            return idx, audio_buffer
        except Exception as e:
            logger.error(f"Error processing request {idx}: {e}", exc_info=True)
            return idx, None
    
    # Process all requests concurrently
    tasks = [process_single_request(i, req) for i, req in enumerate(request_data)]
    results = await asyncio.gather(*tasks)
    
    # Create a ZIP file with all results
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for idx, buffer in results:
            if buffer:
                buffer.seek(0)
                zip_file.writestr(f"audio_{idx}.wav", buffer.read())
    
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=audio_batch.zip"}
    )

# Optional: Create a class for streaming functionality
class AudioChunkGenerator:
    def __init__(self, model, text, speaker_wav, sampling_rate, language, speed, chunk_size=1024):
        self.model = model
        self.text = text
        self.speaker_wav = speaker_wav
        self.sampling_rate = sampling_rate
        self.language = language
        self.speed = speed
        self.chunk_size = chunk_size
    
    async def generate_chunks(self):
        try:
            # Create speaker embedding
            speaker = self.model.make_speaker_embedding(self.speaker_wav, self.sampling_rate)
            
            # Prepare conditioning
            cond_dict = make_cond_dict(text=self.text, speaker=speaker, language=self.language)
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Apply speed factor if needed
            if self.speed != 1.0:
                conditioning = self.model.adjust_speed(conditioning, speed_factor=self.speed)
            
            # Generate audio codes
            codes = await asyncio.to_thread(self.model.generate, conditioning)
            
            # Decode to audio waveform (this could be chunked for larger outputs)
            wavs = await asyncio.to_thread(self.model.autoencoder.decode, codes)
            audio_np = wavs[0].cpu().numpy()
            
            # Send WAV header first
            yield b'RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\\\0\0\0\x01\x18\0data\0\0\0\0'
            
            # Stream audio in chunks
            total_bytes = 0
            for i in range(0, len(audio_np), self.chunk_size):
                chunk = audio_np[i:i+self.chunk_size]
                chunk_bytes = chunk.tobytes()
                total_bytes += len(chunk_bytes)
                yield chunk_bytes
                
            logger.info("Streaming generation complete.")
            
        except Exception as e:
            logger.error(f"Error during streaming audio generation: {e}", exc_info=True)
            raise

@app.post("/tts-stream")
async def text_to_speech_stream(request_data: TTSRequest):
    """
    Generates TTS audio and streams the WAV audio chunks as they become available.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")

    try:
        # Load speaker reference audio
        speaker_wav, sampling_rate = await process_audio_reference(request_data.speaker_path)
        
        # Create a chunk generator
        chunk_generator = AudioChunkGenerator(
            model, 
            request_data.text, 
            speaker_wav, 
            sampling_rate, 
            request_data.language, 
            request_data.speed
        )
        
        # Return streaming response
        return StreamingResponse(
            chunk_generator.generate_chunks(),
            media_type="audio/wav",
        )
    except Exception as e:
        logger.error(f"Unhandled exception in /tts-stream endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)