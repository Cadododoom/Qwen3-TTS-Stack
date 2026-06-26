#!/bin/bash

# Terminate background processes on exit
trap 'kill $(jobs -p)' EXIT

# Start the GPU-driven streaming server (Port 8090 inside the container)
echo "Starting GPU-driven streaming server on port 8090..."
PORT=8090 python app_streaming.py &

# Start the CPU-driven Gradio web server (Port 7860 inside the container)
echo "Starting CPU-driven web Gradio server on port 7860..."
python app.py Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 7860 --device cpu --dtype float32
