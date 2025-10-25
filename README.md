# Realtime Audio Chat

Realtime Audio Chat is a local-first voice assistant that streams audio to an OpenAI-compatible
LLM, speaks replies with Fish-Speech, and now relies on a dedicated Faster-Whisper speech-to-text
(STT) microservice. The two environments keep heavy dependencies isolated so the transcription stack
never pollutes the TTS/LLM runtime.

## Architecture Overview

```
┌──────────────────────┐        HTTP        ┌──────────────────────────┐
│ venv-stt (FastAPI)   │ <────────────────> │ venv-app (FastAPI/Gradio)│
│ faster-whisper model │                    │ Fish-Speech + LLM client │
└──────────────────────┘                    └──────────────────────────┘
```

* `stt_service` exposes `/health` and `/v1/transcribe` endpoints using Faster-Whisper and CTranslate2.
* `app.py` hosts the FastRTC/Gradio interface, calls the LLM, and streams Fish-Speech audio back.
* Communication between the two environments happens over HTTP via `src/stt_client.py`.

## Repository Layout

```
app.py                       # FastAPI + Gradio entrypoint and FastRTC stream handler
fish_speech_adapter.py       # Fish-Speech OpenAudio S1 Mini adapter for FastRTC
requirements.txt             # Application dependencies (no STT libraries)
constraints-app.txt          # Minimal pins shared by the app environment
samples/hello.wav            # Demo audio for the smoke test
scripts/
  setup-stt.ps1              # Create .venv-stt and install Faster-Whisper dependencies
  run-stt.ps1                # Launch the STT service
  setup-app.ps1              # Create .venv-app and install app requirements
  run-app.ps1                # Launch the realtime assistant
  smoke-test.ps1             # One-shot STT service verification
src/stt_client.py            # HTTP client used by the app to reach the STT service
stt_service/                 # FastAPI microservice wrapping Faster-Whisper
```

## Prerequisites

* Windows 11 with PowerShell (preferred workflow) or any platform with Python 3.12.
* A GPU with CUDA 12 + cuDNN 8 for the best STT experience. The service automatically falls back to
  CPU int8 inference when CUDA is not available.
* FFmpeg installed and available on `PATH` for Gradio/FastRTC audio handling.

## Environment Variables

Copy `.env.example` to `.env` and update values as needed:

```
STT_URL=http://127.0.0.1:5007
STT_PORT=5007
STT_MODEL=small
STT_DEVICE=auto
STT_COMPUTE_TYPE=
LOCAL_OPENAI_API_KEY=
LOCAL_OPENAI_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_OPENAI_MODEL=llama-3.1-8b-instruct
```

`run-app.ps1` loads the `.env` file automatically so the assistant can locate the STT service and
LLM endpoint.

## Option B (Two Virtual Environments)

The repository now ships solely with the dual-venv workflow. Follow these steps in PowerShell from
the project root.

### 1. Prepare the STT Service

```powershell
pwsh scripts/setup-stt.ps1
pwsh scripts/run-stt.ps1     # serves FastAPI on http://127.0.0.1:5007
```

Leave the STT process running in its PowerShell window. The health endpoint is available at
`http://127.0.0.1:5007/health`.

### 2. Prepare the Application Environment

```powershell
pwsh scripts/setup-app.ps1
$env:STT_URL = "http://127.0.0.1:5007"
pwsh scripts/run-app.ps1
```

`setup-app.ps1` installs the core Gradio/FastRTC/Fish-Speech stack without any STT dependencies. The
app connects to the STT microservice via the `STT_URL` environment variable (defaulting to
`http://127.0.0.1:5007`).

### 3. Smoke Test the STT Service

Run the automated check after the STT service is running to make sure transcription works end-to-end:

```powershell
pwsh scripts/smoke-test.ps1
```

The script uploads `samples/hello.wav` and prints the JSON response from the STT endpoint.

## Running the Assistant

With both PowerShell windows active (`run-stt.ps1` + `run-app.ps1`), open
<http://127.0.0.1:7860> in a browser. Click **Connect** to start streaming audio through FastRTC and
converse with the assistant. Responses stream back in real time using Fish-Speech.

## Troubleshooting

* **“Unable to reach the speech-to-text service”** – make sure `scripts/run-stt.ps1` is running and
  that `STT_URL` points to it. Restart the app window after the STT service is ready.
* **Slow transcription on CPU** – consider downloading a smaller Faster-Whisper model by setting
  `STT_MODEL` to `base` or `tiny`, or run on a CUDA-enabled GPU.
* **Fish-Speech downloads take a while** – the first launch of `run-app.ps1` triggers checkpoint
  downloads. Subsequent runs reuse the cached models.

## Additional Commands

* Update STT model or device at runtime by exporting `STT_MODEL`, `STT_DEVICE`, or
  `STT_COMPUTE_TYPE` before launching `run-stt.ps1`.
* Stop the STT service started by `smoke-test.ps1` with `Stop-Process` if it remains running after
  manual interruption.

## License

This project inherits the original licensing terms of the upstream realtime-audio-chat repository.
