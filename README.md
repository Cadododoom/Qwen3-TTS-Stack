# Qwen3 TTS Stack

A Docker Compose stack for running and deploying `Qwen3-TTS` (specifically the 0.6B Base model) locally on both CPU and GPU. It provides both an interactive Gradio web interface and a high-performance streaming endpoint optimized for Twilio/Hermes systems.

## Features

- **Dedicated Streaming API**: Python server (`app_streaming.py`) utilizing PyTorch on GPU (`privateuseone:1` / DirectML WSL2 GPU acceleration) for low-latency real-time text-to-speech.
- **Interactive Web Interface**: Gradio-based multi-model hub (`app.py`) for manual testing and voice exploration on CPU.
- **Llama.cpp Companion**: Integrates the standard `llama.cpp` container for downstream local LLM routing.
- **WDDM/WSL2 GPU Passthrough**: Preconfigured volume mapping `/usr/lib/wsl` and device mounting `/dev/dxg` for GPU compute under WSL2 on Windows.

## Project Structure

- `docker-compose.yml`: Orchestrates the streaming API, Graduation web client, and llama.cpp companion services.
- `qwen3-tts/`:
  - `Dockerfile`: Multi-stage build context for setup.
  - `app.py`: Gradio web client.
  - `app_streaming.py`: FastAPI server for streaming voice generation.
  - `default_voice.pt`: Default voice embedding tensor.

## Getting Started

1. Ensure Docker Desktop is running with WSL2 backend.
2. Launch the stack:
   ```bash
   docker compose up -d
   ```
3. Access the endpoints:
   - **Gradio Web Interface**: `http://localhost:7860`
   - **Streaming API**: `http://localhost:7861`
   - **Llama.cpp**: `http://localhost:8080`
