# STT Microservice

This FastAPI application exposes Faster-Whisper speech-to-text over HTTP so the main realtime chat
app can offload transcription to its own Python environment.

## Quickstart (Windows PowerShell)

```powershell
pwsh scripts/setup-stt.ps1
pwsh scripts/run-stt.ps1
```

The service listens on `http://127.0.0.1:5007` by default. Adjust `STT_PORT` before launching to
change the binding.

## Manual Setup

```powershell
python -m venv .venv-stt
. .\.venv-stt\Scripts\Activate.ps1
pip install --upgrade pip wheel
pip install -r stt_service/requirements-stt.txt
```

## Configuration

Environment variables influence how the model loads:

- `STT_MODEL` (default `small`) – Faster-Whisper model identifier.
- `STT_DEVICE` (default `auto`) – `auto`, `cuda`, or `cpu`.
- `STT_COMPUTE_TYPE` – Optional override of the Faster-Whisper compute type.
- `STT_PORT` (default `5007`) – HTTP listen port.

When `STT_DEVICE=auto`, the service picks CUDA if available, otherwise CPU (int8 inference).

## Example Request

```bash
curl -F "file=@samples/hello.wav" http://127.0.0.1:5007/v1/transcribe
```
