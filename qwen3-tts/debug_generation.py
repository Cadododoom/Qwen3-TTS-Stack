import sys
import torch
import time
import os

print("PyTorch Version:", torch.__version__)
print("ROCm/CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

from qwen_tts import Qwen3TTSModel

device = "cuda"
dtype = torch.float16

print("Loading model on device:", device)
start = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map=device,
    torch_dtype=dtype,
    attn_implementation="sdpa"
)
print(f"Model loaded in {time.time() - start:.2f} seconds.")

# Let's inspect model type
print("Model type:", getattr(model.model, "tts_model_type", None))

text = "This is a direct test of speech synthesis."
speaker = "Serena"
instruct = "measured, normal voice"

print(f"Generating voice for text='{text}', speaker='{speaker}', instruct='{instruct}'...")
sys.stdout.flush()

start_gen = time.time()
try:
    # We will run generator step by step to see where it hangs
    # generate_custom_voice returns (wavs, sr)
    # Let's print before calling it
    print("Calling generate_custom_voice...")
    sys.stdout.flush()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="English",
        speaker=speaker,
        instruct=instruct
    )
    print(f"Generation successful! Time: {time.time() - start_gen:.2f}s")
    print(f"WAV shape: {wavs[0].shape}, Sample Rate: {sr}")
    
    import soundfile as sf
    sf.write("/app/debug_output.wav", wavs[0], sr)
    print("Saved audio to /app/debug_output.wav")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
