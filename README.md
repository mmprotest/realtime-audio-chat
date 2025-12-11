# Realtime Audio Chat

This repository contains a collection of small Gradio/FastAPI apps that demonstrate low-latency, bidirectional voice chat. Each app combines automatic speech recognition (ASR), a large language model (LLM), and text-to-speech (TTS) synthesis to enable “hear a user, respond with speech” workflows.

## Features
- **FastRTC streaming**: Streams microphone audio from the browser, returning synthesized speech in near real time.
- **Pluggable ASR**: Uses either local Faster-Whisper or the built-in FastRTC ASR model.
- **Pluggable LLMs**: Works with OpenAI-compatible models or Groq-hosted models (Llama 3.1).
- **Flexible TTS**: Swap between locally hosted F5-TTS or ElevenLabs streaming TTS.
- **Multiple entry points**: Choose the stack that matches your environment:
  - `realtime_audio_chat.py`: FastRTC + local Faster-Whisper + F5-TTS + OpenAI-compatible LLM.
  - `fastrtc_ui.py`: FastRTC + FastRTC ASR + Groq LLM + ElevenLabs TTS.
  - `whisper_stt_gradio_ui.py`: Simple Faster-Whisper transcription UI (no TTS or LLM).

## Prerequisites
- Python 3.10+
- `ffmpeg` (required by `soundfile`/`gradio` for audio handling)
- A virtual environment is recommended.

## Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is unavailable, install directly:
   ```bash
   pip install gradio fastapi fastrtc python-dotenv requests numpy soundfile openai groq elevenlabs
   ```

## Environment Variables
Set the following variables in a `.env` file or your shell before running the apps.

### Shared
- `MODE`: `UI` (default) to launch the web UI or `PHONE` to start the FastRTC phone gateway.
- `GRADIO_SSR_MODE`: Set to `false` to disable SSR (recommended for local use).

### `realtime_audio_chat.py`
- **LLM**
  - `OPENAI_API_KEY`: API key for an OpenAI-compatible server (required).
  - `OPENAI_BASE_URL`: Base URL for the API (e.g., `http://127.0.0.1:1234/v1`).
  - `OPENAI_MODEL`: Model name (default: `gpt-4o-mini`).
  - `OPENAI_MAX_TOKENS`: Maximum tokens in the response (default: `512`).
  - `OPENAI_TEMPERATURE`: Sampling temperature (default: `0.7`).
- **ASR (Faster-Whisper API)**
  - `FASTER_WHISPER_API_URL`: Base URL (default: `http://localhost:8080`).
  - `FASTER_WHISPER_MODEL`: Model name (default: `medium`).
  - `FASTER_WHISPER_TASK`: `transcribe` or `translate` (default: `transcribe`).
  - `FASTER_WHISPER_LANGUAGE`: Optional language hint.
  - `FASTER_WHISPER_BEAM_SIZE`: Beam search width (default: `5`).
  - `FASTER_WHISPER_TIMEOUT`: Request timeout in seconds (default: `120`).
- **TTS (F5-TTS API)**
  - `F5_TTS_API_URL`: Base URL (default: `http://127.0.0.1:8000`).
  - `F5_REFERENCE_AUDIO`: Path to a reference voice sample (default: `morgan.mp3`).
  - `F5_REFERENCE_TEXT`: Reference transcript that matches the reference audio (required).
  - `F5_REMOVE_SILENCE`: `true`/`false` to trim silence (default: `false`).
  - `F5_SEED`: Optional deterministic seed.
  - `F5_REQUEST_TIMEOUT`: Request timeout in seconds (default: `300`).
  - `F5_CHUNK_DURATION`: Seconds per streamed chunk (default: `0.5`).

### `fastrtc_ui.py`
- `ELEVENLABS_API_KEY`: API key for ElevenLabs streaming TTS.
- `GROQ_API_KEY`: API key for Groq-hosted Llama models (read by the `groq` client).

### `whisper_stt_gradio_ui.py`
- `FASTER_WHISPER_API_URL`: Base URL of your Faster-Whisper service (default: `http://localhost:8000`).

## Running the Apps
All apps bind to port `7860` by default. Disable SSR for Gradio by setting `GRADIO_SSR_MODE=false`.

### Realtime Audio Chat (Faster-Whisper + F5-TTS + OpenAI-compatible LLM)
```bash
GRADIO_SSR_MODE=false MODE=UI python realtime_audio_chat.py
# or, to expose a phone endpoint via FastRTC
GRADIO_SSR_MODE=false MODE=PHONE python realtime_audio_chat.py
```
Navigate to `http://localhost:7860` and allow microphone access. Speak into the mic; the app transcribes with Faster-Whisper, sends the text to the LLM, and streams synthesized speech from F5-TTS.

### FastRTC Groq/ElevenLabs Demo
```bash
GRADIO_SSR_MODE=false MODE=UI python fastrtc_ui.py
```
This version uses the FastRTC ASR helper, Groq’s Llama 3.1 for responses, and ElevenLabs streaming TTS.

### Faster-Whisper Transcription UI
```bash
python whisper_stt_gradio_ui.py
```
Opens a simple Gradio interface for recording microphone audio and submitting it to your Faster-Whisper API. Configure the model, language hint, task, and beam size directly in the UI.

## Project Structure
- `realtime_audio_chat.py` — Full-stack FastRTC demo with local Faster-Whisper, OpenAI-compatible LLM, and F5-TTS streaming.
- `fastrtc_ui.py` — Alternative FastRTC demo using Groq + ElevenLabs.
- `whisper_stt_gradio_ui.py` — Standalone Faster-Whisper transcription client.
- `morgan.mp3` — Default reference audio used by F5-TTS.

## How It Works (high level)
1. **Capture**: FastRTC streams microphone audio from the browser to the server.
2. **Transcribe**: Audio is sent to Faster-Whisper (`transcribe_audio`) or the built-in FastRTC ASR.
3. **Chat**: The transcription and prior chat history are sent to an LLM (`generate_response`).
4. **Synthesize**: The response text is synthesized with F5-TTS or ElevenLabs (`synthesize_speech`).
5. **Stream back**: Audio chunks are streamed back to the browser, with chat history updates handled via `AdditionalOutputs`.

## Troubleshooting
- **No audio output**: Ensure your TTS service is reachable (`F5_TTS_API_URL` or ElevenLabs credentials) and the reference audio/text are set.
- **Transcription failures**: Verify the Faster-Whisper endpoint and that `FASTER_WHISPER_MODEL` exists on the server.
- **LLM errors**: Confirm `OPENAI_API_KEY` / `GROQ_API_KEY` and the correct base URL/model name.
- **Browser microphone issues**: Clear site permissions or try another browser; FastRTC requires mic access.

## License
No explicit license is provided in this repository. Please reach out to the maintainers before reuse or redistribution.
