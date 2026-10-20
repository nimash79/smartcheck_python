# SmartCheck — Python Recognition Engine

The Python component of **SmartCheck**, a camera-based monitoring system that combines computer vision, face recognition, and audio processing to detect people and communicate recognition events to the backend.

## Overview

SmartCheck uses a Python-based processing layer to analyze camera input and support person recognition. The component communicates with the central backend through HTTP requests and WebSocket connections, allowing recognition events to be integrated into the wider application.

In addition to face and human detection, the repository contains modules for speaker identification, live speech processing, and audio-stream handling.

## Features

- **Face recognition:** Uses the `face_recognition` library to identify faces.
- **Human detection:** Includes YOLO-related and human-detection modules.
- **Camera processing:** Uses OpenCV to process visual input.
- **Audio processing:** Supports audio recording and processing workflows.
- **Speaker identification:** Uses voice embeddings and the Resemblyzer library.
- **Speech recognition:** Includes live-captioning and speech-processing code using Vosk and other recognition workflows.
- **Backend integration:** Sends data to the Node.js backend using configurable HTTP and WebSocket endpoints.
- **Event cooldown:** Includes a cooldown mechanism to reduce repeated recognition events.
- **Modular processing:** Separates recognition, audio, and supporting functionality into individual Python modules.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core processing logic |
| OpenCV | Image and video processing |
| `face_recognition` | Face detection and recognition |
| YOLO | Object/person detection workflows |
| NumPy | Numerical operations |
| `requests` | HTTP communication |
| `websockets` | WebSocket communication |
| Flask | HTTP service in `server.py` |
| Resemblyzer | Voice embeddings and speaker identification |
| Librosa | Audio loading and processing |
| Vosk | Offline speech-recognition workflow |
| `python-dotenv` | Environment configuration |

Some modules represent separate experiments or supporting utilities. Their dependencies may differ from those required by the primary runtime.

## Repository Structure

```text
smartcheck_python/
├── main.py                  # Recognition and backend communication
├── server.py                # Flask service and audio-related processing
├── human_detection.py       # Human-detection workflow
├── full_body_detection.py   # Full-body and landmark experiments
├── speaker_identification.py
├── audio_server.py
├── live_voice.py
├── live_captions_vosk.py
├── google_model.py
├── vosk_model.py
├── yolov8n.pt               # YOLO model weights
├── .env.example
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10 or another Python version compatible with the selected dependencies.
- A camera or accessible camera stream.
- FFmpeg for workflows that record or process audio streams.
- A configured SmartCheck backend.
- Compatible model weights and speech-recognition resources.

The `face_recognition` library may require additional system dependencies, and some models may need separate downloads.

### 1. Clone the repository

```bash
git clone https://github.com/nimash79/smartcheck_python.git
cd smartcheck_python
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

The repository does not provide a verified, complete dependency lockfile for all modules. Install the dependencies required by the workflow you intend to run.

Core libraries include:

```bash
pip install opencv-python numpy requests python-dotenv websockets
pip install Flask flask-cors face-recognition
pip install resemblyzer librosa
```

Additional dependencies may be required for YOLO, Vosk, speech recognition, and the selected model versions. Check each module before running it.

### 4. Configure environment variables

Create a local `.env` file based on `.env.example`.

```dotenv
API_URL=http://localhost:3000
WS_URL=ws://localhost:3000
PORT=5000
```

These are illustrative local-development values. Replace them with the actual addresses and ports used by your deployment. The Python service may also require a `VOICES_FOLDER` configuration for audio-related workflows.

Never commit credentials or sensitive deployment settings.

### 5. Start the recognition process

The main entry point is:

```bash
python main.py
```

The repository also includes a Flask service:

```bash
python server.py
```

These entry points serve different purposes. Start the process required by your deployment and configure the backend connection before running them together.

## Integration with SmartCheck

The Python component communicates with the Node.js backend using the configured API and WebSocket endpoints.

```text
Camera / Audio Input
        |
        v
Python Recognition Engine
        |
        +---- Face / Person Processing
        |
        +---- Audio / Speaker Processing
        |
        v
HTTP API and WebSocket Events
        |
        v
SmartCheck Backend
        |
        v
SmartCheck Web Interface
```

The exact event payloads and message contracts should be taken from the implementation in `main.py`, `server.py`, and the backend's route and WebSocket handlers.

## Operational Considerations

- Ensure the camera stream is accessible from the machine running the recognition engine.
- Confirm that model weights and speech-recognition resources are available.
- Configure the API and WebSocket addresses consistently with the backend.
- Avoid deploying camera streams or audio services without appropriate access controls.
- Test recognition accuracy under different lighting, camera angles, and audio conditions.
- Treat face and voice information as sensitive personal data and restrict access accordingly.

## Related Repositories

- [SmartCheck Backend](https://github.com/nimash79/smartcheck_backend) — API, application logic, and event integration.
- [SmartCheck Frontend](https://github.com/nimash79/smartcheck_frontend) — Web interface.

## Author

**Nima Sharifinia** · [GitHub](https://github.com/nimash79)
