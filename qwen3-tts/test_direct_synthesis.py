import os
import sys
import torch
import time

# Disable engine warmup before importing RealtimeTTS
from RealtimeTTS.engines.faster_qwen_engine import FasterQwenEngine
FasterQwenEngine._warmup = lambda self: None
print("[Patch] FasterQwenEngine warmup disabled.")

from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines.faster_qwen_engine import FasterQwenVoice

print("Python version:", sys.version)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

print("Initializing FasterQwenEngine...")
sys.stdout.flush()
start_time = time.time()
engine = FasterQwenEngine(
    model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda",
)
print(f"Engine initialized in {time.time() - start_time:.2f} seconds.")
sys.stdout.flush()

default_pt_path = "/app/default_voice.pt"
if os.path.exists(default_pt_path):
    print(f"Loading voice profile from {default_pt_path}")
    voice = FasterQwenVoice(
        name="default",
        speaker_pt=default_pt_path,
        language="English"
    )
    engine.set_voice(voice)
else:
    print("Voice profile not found!")
    sys.exit(1)

stream = TextToAudioStream(engine, muted=True)

test_text = "This is a test of direct synthesis on AMD GPU."
print(f"Feeding text: '{test_text}'")
stream.feed(test_text)

print("Starting stream.play...")
sys.stdout.flush()
play_start = time.time()

chunks = []
def on_audio_chunk(chunk):
    chunks.append(chunk)
    print(f"Received audio chunk of size: {len(chunk)} bytes")
    sys.stdout.flush()

stream.play(
    muted=True,
    on_audio_chunk=on_audio_chunk,
    fast_sentence_fragment=True
)

print(f"stream.play finished in {time.time() - play_start:.2f} seconds.")
print(f"Total audio chunks received: {len(chunks)}")
sys.stdout.flush()
