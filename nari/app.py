import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import soundfile as sf
import io
import asyncio
import numpy as np # Assuming output is numpy array

# --- Model Loading ---
# Attempt to import and load the actual model
try:
    from dia.model import Dia
    # Load the model globally on startup
    # This might take time; consider FastAPI lifespan events for complex setups
    model = Dia.from_pretrained("nari-labs/Dia-1.6B")
    print("Dia model loaded successfully.")
except ImportError:
    print("Warning: 'dia.model' not found. Using a placeholder mock model.")
    # Placeholder Mock Dia class if actual import fails
    class MockDia:
        def generate(self, text: str):
            print(f"Mock generating audio for: {text}")
            # Simulate audio generation - replace with actual model call logic if using mock
            sample_rate = 44100
            duration_seconds = len(text) / 10 # Simple duration based on text length
            frequency = 440 # A4 note
            t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds))
            amplitude = np.iinfo(np.int16).max * 0.5
            data = amplitude * np.sin(2. * np.pi * frequency * t)
            # Simulate the output structure: audio data (numpy array) and sample rate
            return data.astype(np.int16), sample_rate
    model = MockDia()
except Exception as e:
    print(f"Error loading Dia model: {e}")
    # Set model to None or raise to prevent app start if model is critical
    model = None

# --- FastAPI App ---
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/generate/")
async def generate_audio_endpoint(input_data: TextInput):
    """
    Accepts text input and returns generated dialogue audio.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    text = input_data.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # --- Potentially CPU-bound task ---
        # Run the synchronous model.generate function in a thread pool
        # to avoid blocking the asyncio event loop.
        loop = asyncio.get_running_loop()

        # Assuming model.generate returns a tuple: (audio_data_np_array, sample_rate)
        # If generate is already async, you can just 'await model.generate(text)'
        output_data, sample_rate = await loop.run_in_executor(None, model.generate, text)

        # Ensure output_data is a NumPy array suitable for soundfile
        if not isinstance(output_data, np.ndarray):
             # Attempt conversion or raise error if format is unexpected
             try:
                 output_data = np.array(output_data)
                 # Add further checks/conversions if needed based on model output type
             except Exception as conversion_error:
                 print(f"Error converting model output to NumPy array: {conversion_error}")
                 raise HTTPException(status_code=500, detail="Unexpected model output format.")

        # Write the NumPy array to an in-memory WAV file buffer
        buffer = io.BytesIO()
        # Using WAV format as it's widely supported by soundfile without extra deps
        # Ensure the dtype is appropriate (e.g., int16, float32)
        sf.write(buffer, output_data, sample_rate, format='WAV', subtype='PCM_16') # Adjust subtype if needed
        buffer.seek(0)

        # Return as a streaming response
        return StreamingResponse(buffer, media_type="audio/wav")

    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        # Log the full error for debugging purposes
        print(f"Error during audio generation for text '{text}': {e}")
        # Return a generic server error to the client
        raise HTTPException(status_code=500, detail="Internal server error during audio generation.")

if __name__ == "__main__":
    # Run with Uvicorn for development purposes
    # Host '0.0.0.0' makes it accessible on the network
    # For production, run using:
    # uvicorn nari.app:app --host 0.0.0.0 --port 8080 --workers 4
    print("Starting Uvicorn server for development...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
