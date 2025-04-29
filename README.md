# Multi-Service TTS Server

This project provides a Text-to-Speech (TTS) service powered by multiple backend TTS engines, orchestrated using Docker Compose and Nginx.

## Project Overview

The system consists of an Nginx reverse proxy that routes requests to two distinct TTS backend services: `kokoro` and `zonos`. This allows leveraging different TTS models and features through a single entry point.

## Architecture

```
+-----------------+      +---------------------+      +---------------------+
|      User       | ---> | Nginx Reverse Proxy | ---> |   Kokoro Service    |
| (HTTP Requests) |      |    (Port 8080)      |      | (TTS - Kokoro Lib)  |
+-----------------+      +---------------------+      +---------------------+
                             |
                             |
                             v
                           +---------------------+
                           |    Zonos Service    |
                           | (TTS - Zonos Model) |
                           +---------------------+
```

-   **Nginx:** Acts as the entry point, listening on port 8080. It routes requests based on the URL path:
    -   Requests to `/kokoro/` are forwarded to the `kokoro` service.
    -   Requests to `/zonos/` are forwarded to the `zonos` service.
-   **Kokoro Service:** A FastAPI application using the `kokoro` Python library for TTS generation.
-   **Zonos Service:** A FastAPI application using the `Zonos` model for TTS generation, including voice cloning capabilities.

## Services

### 1. Nginx (`nginx`)

-   **Image:** `nginx:latest`
-   **Configuration:** Uses `./nginx.conf` for routing rules.
-   **Port:** Exposes port `8080` on the host.
-   **Role:** Routes incoming traffic to the appropriate backend service based on the URL prefix (`/kokoro/` or `/zonos/`).

### 2. Kokoro TTS Service (`kokoro`)

-   **Build Context:** `./kokoro`
-   **Dockerfile:** Defines a Python environment, installs `kokoro`, `fastapi`, `uvicorn`, and other dependencies. Uses a multi-stage build supporting CPU and GPU (defaults to CPU in `docker-compose.yml`).
-   **Application:** `kokoro/main.py` (FastAPI)
-   **Functionality:** Provides TTS using the `kokoro` library.
-   **Endpoints (relative to `/kokoro/`):**
    -   `POST /tts`: Generates a single WAV file from text. Accepts JSON body: `{ "text": "...", "voice": "...", "speed": ... }`.
    -   `POST /tts-batch`: Generates multiple WAV files from a batch of requests and returns a ZIP archive. Accepts JSON body: `{ "items": [ { "text": ..., "voice": ..., "speed": ... }, ... ] }`.
    -   `POST /tts-stream`: Streams generated WAV audio chunks. Accepts JSON body: `{ "text": "...", "voice": "...", "speed": ... }`.
-   **Internal Port:** `8080`

### 3. Zonos TTS Service (`zonos`)

-   **Build Context:** `./zonos`
-   **Dockerfile:** Uses a PyTorch base image, clones the `Zonos` repository, installs dependencies using `uv`, and runs the FastAPI app.
-   **Application:** `zonos/main.py` (FastAPI)
-   **Functionality:** Provides TTS using the `Zonos` model, supporting voice cloning from reference audio.
-   **Endpoints (relative to `/zonos/`):**
    -   `POST /tts`: Generates a single WAV file. Accepts `multipart/form-data` with fields:
        -   `text` (required): The text to synthesize.
        -   `language` (optional, default: "en-us"): Language code.
        -   `speed` (optional, default: 1.0): Speech speed.
        -   `speaker_path` (optional): Path to a reference audio file *on the server*.
        -   `speaker_file` (optional): Uploaded reference audio file for voice cloning.
        -   `emotion_list` (optional): List of emotion values.
-   **Internal Port:** `8080`

## Setup and Running

1.  **Prerequisites:**
    -   Docker
    -   Docker Compose

2.  **Build and Run:**
    ```bash
    docker-compose up -d --build
    ```
    This command will build the images for `kokoro` (CPU target) and `zonos`, and start all services (nginx, kokoro, zonos) in detached mode.

3.  **Stopping:**
    ```bash
    docker-compose down
    ```

## API Usage Examples

Replace `localhost:8080` with your server's address if running remotely.

### Kokoro TTS

**Generate single audio:**

```bash
curl -X POST http://localhost:8080/kokoro/tts \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hello from the Kokoro service.",
           "voice": "af_heart",
           "speed": 1.1
         }' \
     --output kokoro_output.wav
```

**Generate batch audio:**

```bash
curl -X POST http://localhost:8080/kokoro/tts-batch \
     -H "Content-Type: application/json" \
     -d '{
           "items": [
             { "text": "First sentence.", "voice": "en_female_1" },
             { "text": "Second sentence.", "voice": "en_male_1", "speed": 0.9 }
           ]
         }' \
     --output kokoro_batch_output.zip
```

### Zonos TTS

**Generate audio with default voice:**

```bash
curl -X POST http://localhost:8080/zonos/tts \
     -F "text=Hello from the Zonos service." \
     --output zonos_output.wav
```

**Generate audio with voice cloning (uploading reference):**

```bash
curl -X POST http://localhost:8080/zonos/tts \
     -F "text=This audio should sound like the reference file." \
     -F "speaker_file=@/path/to/your/reference_audio.wav" \
     --output zonos_cloned_output.wav
```

**Generate audio with voice cloning (using server path):**

*Note: This requires the reference audio file to exist at the specified path *inside* the `zonos` container.*

```bash
curl -X POST http://localhost:8080/zonos/tts \
     -F "text=This audio should sound like the reference file on the server." \
     -F "speaker_path=/path/inside/container/reference.wav" \
     --output zonos_cloned_server_path_output.wav
