# Realtime Audio Chat

This project wires together an OpenAI-compatible LLM, a local Whisper transcription service, and a local F5-TTS streaming server into a realtime voice assistant. The FastAPI backend hosts a Gradio UI via FastRTC so the browser can capture microphone audio, hand it to Whisper for speech-to-text, generate a response with your configured OpenAI-compatible model, and immediately stream synthesized speech from F5-TTS back to the user.

## Features
- 🎤 **Local Whisper STT** – send microphone audio to your own Whisper REST API.
- 🗣️ **Local F5-TTS streaming** – stream raw PCM audio from a self-hosted F5-TTS server.
- 🤖 **OpenAI-compatible responses** – point the app at the OpenAI API or any server that implements the Chat Completions interface.
- ⚡ **FastRTC integration** – bidirectional audio streaming with pause detection and WebRTC transport.

## Getting started

### Requirements
- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`) or credentials for your compatible server.
- For local audio services you can either supply your own deployments or run the bundled Whisper and F5-TTS FastAPI apps.

### Installation

#### Windows (PowerShell)
Run the bundled setup script to automate every prerequisite, including FFmpeg:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./scripts/setup_windows.ps1
```

The script will:

- Validate that Python 3.10+ is available (and tell you how to install it if not).
- Create or reuse a `.venv` virtual environment in the repository.
- Upgrade `pip`, install the packages from `requirements.txt`, and add the optional `stt_server`/`tts_server` extras so the bundled Whisper and F5-TTS services are ready to run.
- Download the latest FFmpeg “essentials” build for Windows into `tools/ffmpeg` and add its `bin` directory to your user `PATH` (you may need to reopen your terminal for the new PATH entry to take effect).

After the script finishes, activate the environment and launch the app:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.app
```

#### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: install inference dependencies for the bundled servers
pip install '.[stt_server]' '.[tts_server]'
```

### Configuration
Create a `.env` file (optional) or export the following environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | – | API key for OpenAI or a compatible server (optional if provided elsewhere). |
| `OPENAI_BASE_URL` | – | Base URL for OpenAI-compatible deployments (omit for api.openai.com). |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat completion model. |
| `OPENAI_MAX_TOKENS` | `512` | Maximum tokens to generate per turn. |
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

### Run the bundled audio services

Two FastAPI apps live under `src/servers/` so you can host Whisper and F5-TTS locally without additional glue code.

```bash
# Whisper speech-to-text on http://localhost:9000
python -m src.servers.whisper

# F5-TTS streaming synthesis on http://localhost:9880
python -m src.servers.f5
```

By default the F5-TTS server loads the included `resources/voices/voices.yaml`, which maps the `ljspeech` voice to a
public-domain LJSpeech sample. You can point the server at your own voices file via `F5_TTS_VOICES_FILE=/path/to/voices.yaml`.

## Local service expectations

### Whisper `/transcribe`
- Method: `POST`
- JSON body: `{ "audio": "<base64 wav>", "sample_rate": 16000, "language": "en" }`
- Response JSON: `{ "text": "...", "duration": 1.23 }`

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
│  ├─ local_clients.py
│  ├─ resources/
│  │  └─ voices/
│  │     ├─ README.md
│  │     ├─ ljspeech-LJ001-0001.txt
│  │     ├─ ljspeech-LJ001-0001.wav
│  │     └─ voices.yaml
│  └─ servers/
│     ├─ __init__.py
│     ├─ f5.py
│     └─ whisper.py
└─ tests/
   ├─ conftest.py
   ├─ test_app_history.py
   ├─ test_config.py
   ├─ test_f5_server.py
   ├─ test_local_clients.py
   └─ test_whisper_server.py
```

## Testing
```bash
pytest
```
