FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install system dependencies (portaudio19-dev needed for PyAudio/RealtimeTTS build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox git build-essential portaudio19-dev nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and torch-directml
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 torch-directml

# Install bleeding-edge RealtimeTTS and faster-qwen3-tts
RUN pip install --no-cache-dir "realtimetts[qwen] @ git+https://github.com/KoljaB/RealtimeTTS.git" \
    && pip install --no-cache-dir git+https://github.com/andimarafioti/faster-qwen3-tts.git \
    && pip install --no-cache-dir fastapi uvicorn websockets gradio qwen-tts

# Force downgrade/reinstall torch, torchaudio, and torchvision to 2.3.1 / 0.18.1 without dependencies
RUN pip install --no-cache-dir --force-reinstall --no-deps torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1


WORKDIR /app
COPY app_streaming.py /app/app_streaming.py
COPY app.py /app/app.py
COPY demo.py /app/demo.py

EXPOSE 7860

# Default model name
ENV MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-Base

# Start the FastAPI streaming server
CMD ["python", "app_streaming.py"]
