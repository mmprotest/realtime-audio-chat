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

## Before You Start

1. **Check your hardware.** You need an NVIDIA GPU with drivers that support CUDA 13.0 or newer. If you do not have a GPU, you can still run the project by installing CPU-only PyTorch, but the scripted Windows install focuses on CUDA.
2. **Free up disk space.** You will download multiple gigabytes of model checkpoints and Python wheels. Leave at least 15 GB available on the drive where you clone the repo.
3. **Decide on your install path.** The instructions below assume you clone the repository into `C:\realtime-audio-chat` on Windows or `~/realtime-audio-chat` on Linux/macOS.
4. **Have administrator access on Windows.** The setup script elevates itself to install Python, the Microsoft Visual C++ runtime, and FFmpeg. Keep the PowerShell window open until it finishes.

## Windows: Foolproof One-Time Setup

Follow every step in order. Do not skip the verification prompts.

1. **Open PowerShell as Administrator.**
   - Press the Windows key, type `PowerShell`, right-click "Windows PowerShell", and select **Run as administrator**.
   - When asked "Do you want to allow this app to make changes to your device?", choose **Yes**.
2. **Enable script execution for this session.** Run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   ```
   The change only lasts for the current PowerShell window.
3. **Clone the repository and enter it.** Replace the URL if you are using your own fork.
   ```powershell
   git clone https://github.com/your-org/realtime-audio-chat.git C:\realtime-audio-chat
   Set-Location C:\realtime-audio-chat
   ```
4. **Run the automated setup.**
   ```powershell
   .\scripts\setup_windows.ps1
   ```
   The script will:
   - confirm administrative rights (it re-launches itself elevated if required)
   - check that an NVIDIA GPU and CUDA 13.0 toolkit are visible via `nvcc`
   - download Python 3.10.14 from python.org if it is missing and install it silently
   - install the Microsoft Visual C++ runtime and FFmpeg if they are absent
   - create a virtual environment at `.venv` and upgrade `pip`, `setuptools`, and `wheel`
   - install the CUDA-enabled PyTorch wheels and the pinned runtime packages from `requirements.txt` (including `numpy<2` which torchaudio requires)
   - install the Fish-Speech package from `requirements-fish-speech.txt` using `--no-deps`
   - print the exact command to activate the environment when it finishes
5. **Verify that Python works inside the virtual environment.** When the script finishes, follow its final prompt (normally `.\.venv\Scripts\Activate.ps1`). After activation, run:
   ```powershell
   python --version
   ```
   You should see `Python 3.10.14` (or another version if you passed `-PythonVersion`).
6. **Download the Fish-Speech checkpoints (first run only).** Start the app once to trigger the download and let it finish completely:
   ```powershell
   python app.py
   ```
   Keep the window open until the download progress bars disappear and the server prints `Running on local URL:  http://127.0.0.1:7860`.
7. **Test your microphone and speakers.**
   - Visit <http://127.0.0.1:7860> in a browser on the same machine.
   - Click **Connect**, grant microphone permission, speak a short sentence, and wait for the assistant to respond.
   - You should hear the voice cloned from `morgan.mp3` unless you have configured a custom reference file.

### Reusing the Installation Later

For every new PowerShell session:
```powershell
Set-Location C:\realtime-audio-chat
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python app.py
```

## Linux or macOS: Manual Setup

1. **Install system-level dependencies.**
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg portaudio19-dev python3.10-venv`
   - macOS (Homebrew): `brew install ffmpeg portaudio python@3.10`
2. **Create and enter a Python 3.10 virtual environment.**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
   Confirm the interpreter version with `python --version`.
3. **Upgrade packaging tools.**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```
4. **Install the core requirements and Fish-Speech package.** This step pins `numpy<2` (torchaudio currently breaks on NumPy 2.x) and adds helper libraries that Fish-Speech expects such as `flatten-dict`.
   ```bash
   pip install -r requirements.txt
   pip install --no-deps -r requirements-fish-speech.txt
   ```
5. **Install a compatible PyTorch build.** Visit <https://pytorch.org/get-started/locally/> and copy the command that matches your platform (CUDA or CPU). Run it inside the virtual environment.
6. **Launch the app to download model weights.**
   ```bash
   python app.py
   ```
   Leave the process running until the Fish-Speech checkpoints finish downloading and the server announces the Gradio URL.

## Configuration

Create a `.env` file in the repository root to persist environment variables. The application loads it automatically via `python-dotenv`.

| Variable | Purpose | Default |
|----------|---------|---------|
| `LOCAL_OPENAI_API_KEY` | API key for your OpenAI-compatible endpoint. | `"dummy"` |
| `LOCAL_OPENAI_BASE_URL` | Base URL for the OpenAI-compatible endpoint. | `http://127.0.0.1:1234/v1` |
| `LOCAL_OPENAI_MODEL` | Chat completion model identifier to request. | `llama-3.1-8b-instruct` |
| `F5_REFERENCE_WAV` | Path to the voice-clone reference audio (WAV/MP3). | `morgan.mp3` |
| `F5_REFERENCE_TEXT` | Optional transcript of the reference audio for better alignment. | auto-generated |
| `F5_TARGET_SAMPLE_RATE` | Resample synthesized audio to this rate before streaming. | unset |
| `FISH_SPEECH_CHECKPOINT_DIR` | Directory where Fish-Speech downloads and caches models. | auto inside `~/.cache/fish-speech` |
| `FISH_SPEECH_AUTO_DOWNLOAD` | Set to `0` to require manual checkpoint placement. | `1` |
| `FISH_SPEECH_DEVICE` | Torch device string such as `cuda:0` or `cpu`. | auto-detected |
| `FISH_SPEECH_PRECISION` | Mixed-precision mode (e.g. `bfloat16`, `float16`). | Fish-Speech default |
| `FISH_SPEECH_COMPILE` | Set to `1` to enable `torch.compile` for the model. | `0` |
| `FISH_SPEECH_CHUNK_LENGTH` | Override the number of tokens per audio chunk. | unset |
| `FISH_SPEECH_MAX_NEW_TOKENS` | Limit output length per response. | unset |
| `FISH_SPEECH_TOP_P` | Top-p nucleus sampling value. | unset |
| `FISH_SPEECH_REPETITION_PENALTY` | Penalize repeated tokens. | unset |
| `FISH_SPEECH_TEMPERATURE` | Softmax temperature. | unset |
| `FISH_SPEECH_SEED` | Force deterministic output when supported. | unset |
| `FISH_SPEECH_USE_MEMORY_CACHE` | Opt in to the Fish-Speech memory cache (`1`/`0`). | library default |
| `FISH_SPEECH_NORMALIZE` | Toggle output normalization (`1`/`0`). | `1` |
| `MODE` | `UI` (Gradio web), `PHONE` (FastRTC phone bridge), or unset (UI default). | unset |

The Windows setup script does not modify `.env`. Edit it manually with the values you need after the first successful run.

## Running the Application

With the virtual environment active and your `.env` configured:

```bash
python app.py
```

- **Web UI:** Visit <http://127.0.0.1:7860> to interact with the assistant. Click **Connect** to start streaming audio.
- **Phone bridge:** Set `MODE=PHONE` and supply the TURN credentials required by FastRTC to enable the telephony handler.

## Troubleshooting Checklist

- **The setup script stops with a 404 when downloading Python.** Ensure the machine has internet access. If python.org is blocked, pass a specific version that you know exists, for example `-PythonVersion 3.10.13`.
- **`nvcc` was not found.** Install the CUDA 13 toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads) and reboot so the PATH updates.
- **Packages fail to build wheels on Windows.** Verify that the Microsoft Visual C++ Redistributable installed correctly. Re-run the script; it will re-download the installer if needed.
- **`pip` reports conflicts when installing Fish-Speech.** Make sure you ran the `pip install --no-deps -r requirements-fish-speech.txt` command exactly. The `--no-deps` flag prevents incompatible versions from being pulled automatically.
- **You see "A module compiled against NumPy 1.x cannot be run in NumPy 2.1" when launching the app.** Something upgraded NumPy after setup. Inside the virtual environment run `pip install "numpy<2" --force-reinstall` to roll back to a compatible version, then restart the app.
- **The app keeps re-downloading Fish-Speech checkpoints.** Set `FISH_SPEECH_CHECKPOINT_DIR` to a location that persists between runs, such as `C:\Models\fish-speech`, and run the app once to populate it.
- **You do not hear audio in the browser.** Confirm that your system default playback device is working and that your browser allows autoplay for the local site. Chrome/Edge will prompt in the address bar the first time audio plays.

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes and ensure formatting/tests pass.
3. Submit a pull request summarizing the modifications.

## License

This project inherits the licensing terms of its upstream dependencies. Please consult their repositories for details.
