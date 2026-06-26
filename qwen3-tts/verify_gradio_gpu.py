import sys
import torch
import time
import os

# Import demo functions
sys.path.append("/app")
import demo

print("PyTorch Version:", torch.__version__)
print("CUDA/ROCm Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

# Set the active settings
demo.active_device = "cuda"
demo.active_dtype = torch.float16
demo.active_attn_impl = "sdpa"

print("Loading model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice on CUDA GPU...")
start_load = time.time()
status = demo.load_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
print(status)
print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

if "Failed" in status:
    sys.exit(1)

# Now set up the language and speaker mapping
lang_choices, lang_mapping, spk_choices, spk_mapping = demo.get_choices_for_active_model()
demo.active_lang_map = lang_mapping
demo.active_spk_map = spk_mapping

print("Supported languages:", lang_choices)
print("Supported speakers:", spk_choices)

text = "This is a direct test of the Gradio backend running on the GPU."
language = "English"
speaker = spk_choices[0] if spk_choices else "default"
print(f"Running run_instruct with text: '{text}', language: '{language}', speaker: '{speaker}'...")

start_gen = time.time()
audio_out, status_msg = demo.run_instruct(text, "English", speaker, "measured, normal voice")
print("Status message:", status_msg)
print(f"Synthesis finished in {time.time() - start_gen:.2f} seconds.")

if audio_out:
    sr, wav = audio_out
    print(f"Generated WAV audio: sample rate = {sr}, length = {len(wav)} samples.")
    import soundfile as sf
    sf.write("/app/gradio_gpu_test.wav", wav, sr)
    print("Saved audio to /app/gradio_gpu_test.wav")
else:
    print("Failed to generate audio!")
    sys.exit(1)
