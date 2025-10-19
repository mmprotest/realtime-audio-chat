# Realtime Audio Chat

This project wires together a Groq LLM, a local Whisper transcription service, and a local F5-TTS streaming server into a realtime voice assistant. The FastAPI backend hosts a Gradio UI via FastRTC so the browser can capture microphone audio, hand it to Whisper for speech-to-text, generate a response with Groq, and immediately stream synthesized speech from F5-TTS back to the user.

## Features
- 🎤 **Local Whisper STT** – send microphone audio to your own Whisper REST API.
- 🗣️ **Local F5-TTS streaming** – stream raw PCM audio from a self-hosted F5-TTS server.
- 🤖 **Groq-powered responses** – low-latency text generation via the Groq Cloud API.
- ⚡ **FastRTC integration** – bidirectional audio streaming with pause detection and WebRTC transport.

## Getting started

### Requirements
- Python 3.10+
- Running REST endpoints for Whisper (`/transcribe`) and F5-TTS (`/tts`).
- A Groq API key (`GROQ_API_KEY`).

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file (optional) or export the following environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | – | Required API key for Groq. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Chat completion model. |
| `GROQ_MAX_TOKENS` | `512` | Maximum tokens to generate per turn. |
| `F5_TTS_URL` | `http://localhost:9880` | Base URL of the local F5-TTS server. |
| `F5_TTS_VOICE` | `default` | Voice identifier understood by the TTS service. |
| `F5_TTS_OUTPUT_FORMAT` | `pcm_s16le` | Output format requested from F5-TTS. |
| `WHISPER_URL` | `http://localhost:9000` | Base URL of the Whisper REST server. |
| `WHISPER_LANGUAGE` | – | Optional hint passed to Whisper (`language` payload field). |
| `OUTPUT_SAMPLE_RATE` | `24000` | Sample rate expected from F5-TTS audio. |
| `INPUT_SAMPLE_RATE` | `16000` | Microphone sample rate the Stream handler should expect. |
| `HTTP_TIMEOUT` | `30` | Timeout (seconds) for local STT/TTS HTTP requests. |

### Run the app
```bash
python -m src.app
```

Set `MODE=UI` to force a standard Gradio launch, or `MODE=PHONE` to start the FastRTC FastPhone demo mode.

## Local service expectations

### Whisper `/transcribe`
- Method: `POST`
- JSON body: `{ "audio": "<base64 wav>", "sample_rate": 16000, "language": "en" }`
- Response JSON: `{ "text": "..." }`

### F5-TTS `/tts`
- Method: `POST`
- JSON body: `{ "text": "...", "voice_id": "...", "output_format": "..." }`
- Response body: raw PCM16 bytes streamed with chunked transfer encoding. The client reshapes the stream into `(1, N)` numpy arrays using the configured `OUTPUT_SAMPLE_RATE`.

## Project layout
```
realtime-audio-chat/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ src/
│  ├─ app.py
│  ├─ config.py
│  └─ local_clients.py
└─ tests/
   ├─ conftest.py
   ├─ test_config.py
   └─ test_local_clients.py
```

## Testing
```bash
pytest
```
