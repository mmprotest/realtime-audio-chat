# Realtime Audio Chat

A reference implementation of a local realtime voice assistant that combines [FastRTC](https://github.com/answerdotai/fastrtc) for low-latency audio streaming, Whisper-based speech-to-text, OpenAI-compatible text generation, and [Fish-Speech OpenAudio S1 Mini](https://huggingface.co/fishaudio/openaudio-s1-mini) voice cloning for speech synthesis. The project exposes a Gradio-powered browser UI and can also proxy the experience over a phone call.

## Features

- **Bidirectional audio streaming** powered by FastRTC with optional TURN credentials for hosted deployments.
- **Local speech-to-text** transcription via `fastrtc-whisper-cpp`.
- **Configurable LLM backend** that targets any OpenAI-compatible API endpoint.
- **Streaming text-to-speech** responses driven by the Fish-Speech OpenAudio S1 Mini model, returning audio as sentences complete.
- **Gradio UI or telephony mode** based on the `MODE` environment variable.

## Repository Layout

```text
app.py                       # FastAPI + Gradio entrypoint and FastRTC stream handler
fish_speech_adapter.py       # Fish-Speech OpenAudio S1 Mini adapter for FastRTC
requirements.txt             # Core Python dependencies for the application runtime
requirements-fish-speech.txt # Fish-Speech package (install with --no-deps)
scripts/
  setup_windows.ps1          # Turnkey Windows PowerShell bootstrap script (CUDA-enabled)
```

## Prerequisites

| Component | Version / Notes |
|-----------|-----------------|
| Python    | 3.10.x (script defaults to 3.10.14) |
| GPU       | NVIDIA GPU with CUDA 13-compatible drivers (script validates with `nvcc`) |
| OS        | Windows 10/11 (scripted install) or Linux/macOS (manual install) |

The Windows automation script installs CUDA-enabled PyTorch wheels that are forward-compatible with CUDA 13 drivers by sourcing the `cu121` distribution channel from PyTorch.

## Quick Start (Windows + CUDA)

1. **Open PowerShell.** You can start from a standard session; the setup script will prompt for elevation if required.
2. **Clone this repository** and move into the project directory:
   ```powershell
   git clone https://github.com/your-org/realtime-audio-chat.git
   Set-Location realtime-audio-chat
   ```
3. **Run the setup script:**
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   .\scripts\setup_windows.ps1
   ```
   The script will:
   - elevate automatically when administrative privileges are required and check for CUDA 13.0 via `nvcc`
   - install Python 3.10.14, the MSVC runtime, and FFmpeg by downloading their official installers when missing
   - create `.venv`, install CUDA-enabled PyTorch + the core Python dependencies, and then install the Fish-Speech package with its upstream dependency pins disabled (to avoid NumPy/Gradio conflicts)
   - report the commands required to activate the environment and launch the app
4. **Activate the virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
5. **Launch the realtime UI:**
   ```powershell
   python app.py
   ```

### Script Options

The PowerShell script exposes a handful of parameters should you need to customize the install:

```powershell
# Use Python 3.11 and a custom virtual environment directory
.\scripts\setup_windows.ps1 -PythonVersion 3.11 -VenvDir .\.venv-311
```

- `-PythonVersion` (default `3.10.14`) downloads the specified CPython release (major.minor.patch) directly from python.org.
- `-CudaVersion` (default `13.0`) is used for validation only and will emit a warning if the detected toolkit differs.
- `-VenvDir` controls where the virtual environment is created.

## Manual Installation (Linux/macOS)

1. Install system dependencies (FFmpeg, PortAudio) using your package manager.
2. Ensure Python 3.10 is available and create a virtual environment:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. Upgrade pip tooling and install the Python dependencies:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install --no-deps -r requirements-fish-speech.txt
   ```
4. Install PyTorch manually. Choose the wheel that matches your CUDA runtime or CPU-only needs from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
5. Configure environment variables (see below) and run the application with `python app.py`.

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `LOCAL_OPENAI_API_KEY` | API key for your OpenAI-compatible endpoint. | `"dummy"` |
| `LOCAL_OPENAI_BASE_URL` | Base URL for the OpenAI-compatible endpoint. | `http://127.0.0.1:1234/v1` |
| `LOCAL_OPENAI_MODEL` | Chat completion model identifier to request. | `llama-3.1-8b-instruct` |
| `F5_REFERENCE_WAV` | Reference audio file used by F5-TTS for cloning. | `morgan.mp3` |
| `F5_REFERENCE_TEXT` | Optional transcript of the reference audio. | auto-generated |
| `F5_MODEL_NAME` | Alternate F5-TTS checkpoint to load. | `F5TTS_v1_Base` |
| `F5_TARGET_SAMPLE_RATE` | Ensure synthesized audio is emitted at a particular sample rate. | unset |
| `MODE` | `UI` (Gradio web), `PHONE` (FastRTC phone bridge), or unset (UI default). | unset |

Create a `.env` file in the repository root to persist these settings across runs. The Windows setup script leaves `.env` untouched so you can supply credentials later.

## Running the Application

With the virtual environment active and the necessary variables configured:

```bash
python app.py
```

- **Web UI:** Visit <http://127.0.0.1:7860> to interact with the assistant. The Gradio interface streams microphone input and plays back responses.
- **Phone bridge:** Set `MODE=PHONE` (and supply FastRTC-compatible TURN credentials if required) to run the telephony entrypoint.

## Troubleshooting

- **CUDA not detected:** The script warns if `nvcc` is missing. Install the CUDA 13 toolkit and matching NVIDIA drivers from the [official download page](https://developer.nvidia.com/cuda-downloads).
- **PyTorch wheel mismatch:** If you maintain a different CUDA minor version, adjust the `-CudaVersion` parameter and update the `$torchIndex` / `$torchPackages` variables in `setup_windows.ps1` accordingly.
- **F5-TTS sample-rate errors:** Set `F5_TARGET_SAMPLE_RATE` to the rate expected by your playback pipeline, or clear the variable to accept the model default.
- **Phone mode connectivity:** Populate TURN credentials via environment variables exposed by FastRTC when deploying remotely.

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes and ensure formatting/tests pass.
3. Submit a pull request summarizing the modifications.

## License

This project inherits the licensing terms of its upstream dependencies. Please consult their repositories for details.
