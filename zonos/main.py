import asyncio
import asyncio
import io
import numpy as np
import soundfile as sf
import torch
import torchaudio
import tempfile # Added tempfile
import os # Added os
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form # Added Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import logging
from zonos.utils import DEFAULT_DEVICE as device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
logger.info(f"Using device: {device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model during startup
    global model
    logger.info("Loading Zonos model...")
    try:
        # Initialize Zonos model
        # Default to hybrid model, can be changed via environment variable
        model_type = "transformer"  # or "transformer"
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
    speaker_path: Optional[str] = None # Optional path to reference audio file
    language: str = "en-us"  # Default language
    speed: float = 1.0

async def process_audio_reference(speaker_path: Optional[str] = None, speaker_file: Optional[UploadFile] = None):
    """Load and process speaker reference audio from path or uploaded file."""
    wav = None
    sampling_rate = None

    if speaker_path:
        try:
            wav, sampling_rate = await asyncio.to_thread(torchaudio.load, speaker_path)
        except Exception as e:
            logger.error(f"Error loading reference audio from path: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load reference audio from path: {str(e)}")
    elif speaker_file:
        temp_file_path = None # Initialize path variable
        try:
            # Log file details
            logger.info(f"Processing uploaded file: filename='{speaker_file.filename}', content_type='{speaker_file.content_type}'")
            
            # Create a temporary file to save the upload
            # Use delete=False on Windows if needed, but True is safer generally
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(speaker_file.filename or ".wav")[1]) as temp_file_obj:
                contents = await speaker_file.read()
                temp_file_obj.write(contents)
                temp_file_path = temp_file_obj.name # Get the path
            
            logger.info(f"Uploaded file saved temporarily to: {temp_file_path}")

            # Load from the temporary file path using soundfile
            data, sampling_rate = await asyncio.to_thread(
                sf.read, temp_file_path, dtype='float32'
            )
            # Convert numpy array to torch tensor
            wav = torch.from_numpy(data).t() # Transpose to get [channels, samples]
            # Ensure it has a channel dimension if mono
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
        except Exception as e:
            # Log the specific error during loading with soundfile
            logger.error(f"Soundfile failed to load uploaded reference audio from temp file: {e}", exc_info=True)
            # Fallback attempt with torchaudio on the same temp file
            try:
                logger.warning(f"Attempting fallback load with torchaudio for temp file: {temp_file_path}")
                wav, sampling_rate = await asyncio.to_thread(torchaudio.load, temp_file_path)
                logger.info("Fallback load with torchaudio succeeded.")
            except Exception as e_torch:
                 logger.error(f"Fallback torchaudio load also failed for temp file: {e_torch}", exc_info=True)
                 raise RuntimeError(f"Failed to load uploaded reference audio (tried soundfile and torchaudio): {str(e)}")
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file deleted: {temp_file_path}")
                except OSError as e_os:
                    logger.error(f"Error deleting temporary file {temp_file_path}: {e_os}")
    else:
        raise ValueError("Either speaker_path or a speaker file must be provided.")
    
    return wav, sampling_rate

def generate_audio_sync(text, speaker_wav, sampling_rate, language, speed, emotion_list=None):
    """Synchronous function to generate audio using Zonos."""
    if model is None:
        raise RuntimeError("Zonos model is not initialized.")
    
    try:
        logger.info(f"Generating audio for language: {language}, speed: {speed}")
        
        # Create speaker embedding from reference audio
        speaker = model.make_speaker_embedding(speaker_wav, sampling_rate)

        if emotion_list is None:
           emotion_tensor = None

        else:
            emotion_tensor = torch.tensor(
                list(map(float, emotion_list)), 
                device=device
            )
        
        # Prepare conditioning
        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, emotion=emotion_tensor)
        conditioning = model.prepare_conditioning(cond_dict)
        
        # Apply speed factor if not 1.0
        if speed != 1.0:
            conditioning = model.adjust_speed(conditioning, speed_factor=speed)
        
        # Generate audio codes
        codes = model.generate(conditioning)
        
        # Decode to audio waveform
        wavs = model.autoencoder.decode(codes)
        audio_tensor = wavs.cpu()

        torchaudio.save("sample.wav", audio_tensor[0], model.autoencoder.sampling_rate)

        # Read the file back into memory
        buffer = io.BytesIO()
        with open("sample.wav", 'rb') as f:
            buffer.write(f.read())
        
        # Clean up temporary file
        try:
            os.unlink("sample.wav")
        except OSError as e:
            logger.warning(f"Failed to delete temporary file {"sample.wav"}: {e}")
            
        buffer.seek(0)
        logger.info("Audio generation complete.")
        return buffer
    except Exception as e:
        logger.error(f"Error during audio generation: {e}", exc_info=True)
        raise

@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form("en-us"),
    speed: float = Form(1.0),
    speaker_path: Optional[str] = Form(None),
    emotion_list: Optional[list[str]] = Form(None),  # Fixed type annotation for emotion_list
    speaker_file: Optional[UploadFile] = File(None, description="Optional speaker reference audio file")
):
    """
    Generates TTS audio using multipart/form-data. 
    Accepts text, language, speed, optional speaker_path (form fields), 
    and optional speaker_file (file upload). Returns the full WAV file.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="TTS Service Unavailable: Model not loaded.")

    speaker_wav = None
    sampling_rate = None
    try:
        # Load speaker reference audio from path or file
        speaker_wav, sampling_rate = await process_audio_reference(speaker_path, speaker_file)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"TTS Service Unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing audio reference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


    try:        
        # Run the synchronous generation function in a separate thread
        audio_buffer = await asyncio.to_thread(
            generate_audio_sync,
            text, # Use the form parameter
            speaker_wav,
            sampling_rate,
            language, # Use the form parameter
            speed, # Use the form parameter
            emotion_list, # Use the form parameter
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



# e1 = 0.1  # anger
# e2 = 0.4  # anticipation
# e3 = 0.0  # disgust
# e4 = 0.2  # fear
# e5 = 0.9  # joy
# e6 = 0.1  # sadness
# e7 = 0.3  # surprise
# e8 = 0.6  # trust


# 완전 서버리스처럼 최소비용 유지하고 싶으면 auto-scaling 기반 서버리스 아키텍처(Lambda+S3+TorchScript)로 가는 것도 고려해볼 수 있어.

