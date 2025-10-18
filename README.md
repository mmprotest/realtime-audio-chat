# Realtime Voice Chat

Low-latency realtime audio assistant that combines Moonshine STT, OpenAI-compatible LLM replies, and F5-TTS one-shot cloning for natural voice responses.

## Features
- 🎤 Moonshine STT via FastRTC for responsive speech recognition.
- 🤖 OpenAI-compatible chat responses (supports custom `OPENAI_BASE_URL`).
- 🗣️ F5-TTS one-shot cloning with streaming 200 ms WAV chunks.
- 🎚️ Persona side panel to upload a reference voice and describe conversational style.
- 🔁 FastRTC “send-receive” chat interface for live audio conversations.

## Quickstart

### Prerequisites
- Python 3.10+
- Optional: CUDA-capable GPU for best latency
- Recommended: `ffmpeg` for broader audio format support

### Installation

#### Windows (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
scripts\setup_windows.ps1
```

#### macOS / Linux (bash)
```bash
bash scripts/setup_unix.sh
```

Both scripts create a virtual environment, install dependencies (including Torch, FastRTC, and F5-TTS), and run the unit tests.

### Configure environment
Copy the template and edit values:

```bash
cp .env.example .env
```

Set at minimum:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini        # change if desired
OPENAI_BASE_URL=http://localhost:8000/v1   # optional for local server
```

### Run the app
```bash
python -m src.app
```

Command-line flag `--cpu` forces CPU inference even when CUDA is available:

```bash
python -m src.app --cpu
```

### Using the realtime chat
1. Open the FastRTC UI (terminal prints the URL) and allow microphone access.
2. Visit the side panel at [http://localhost:7862](http://localhost:7862).
3. Upload a 2–10 s clean reference voice clip. The assistant will mimic its tone.
4. Describe conversational persona cues (tone, pacing, fillers, etc.) and save.
5. Start speaking—Moonshine STT transcribes, the LLM responds, and F5-TTS streams cloned speech back within ~0.5–1.5 s on GPU.

### Latency tips
- Prefer GPUs (`DEVICE=cuda`).
- Reduce LLM `max_tokens` or persona verbosity.
- Increase `CHUNK_MS` for longer but fewer audio chunks.
- Keep voice samples short (under 10 s) and clean.

### Privacy & passphrase check
For sensitive deployments, optionally require the uploaded voice sample to contain a passphrase:
1. Add an environment variable `VOICE_PASSPHRASE="<your phrase>"`.
2. Extend `set_voice_from_path` to run Moonshine STT on the clip and verify the phrase before accepting.
3. Reject uploads that fail the check to prevent unintended cloning.

### Troubleshooting
- **CUDA not detected**: the app falls back to CPU automatically; ensure compatible Torch build is installed.
- **Audio device errors**: close other audio apps or check system permissions.
- **Windows script blocked**: use the `Set-ExecutionPolicy` command above, then rerun the setup script.

## Project structure
```
realtime-voice-chat/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .env.example
├─ src/
│  ├─ app.py
│  ├─ config.py
│  ├─ llm_client.py
│  ├─ persona.py
│  ├─ stt_wrapper.py
│  ├─ tts/
│  │  └─ f5_wrapper.py
│  └─ ui/
│     ├─ __init__.py
│     └─ side_panel.py
├─ tests/
│  ├─ test_persona.py
│  ├─ test_f5_wrapper.py
│  └─ test_config.py
└─ scripts/
   ├─ setup_windows.ps1
   └─ setup_unix.sh
```
