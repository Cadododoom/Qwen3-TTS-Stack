import os
import sys
import asyncio
import torch

# Mock torch.cuda.synchronize as a no-op on CPU to prevent CUDA drivers crash
torch.cuda.synchronize = lambda *args, **kwargs: None

# --- DYNAMIC MONKEY-PATCHING FOR CPU COMPATIBILITY ---
# faster-qwen3-tts is designed for CUDA Graphs (strictly GPU). We dynamically patch it
# on import to bypass CUDA checks, skip graph compilation, and redirect streaming to 
# the dynamic-cache CPU-compatible generator (parity_generate_streaming).

try:
    import faster_qwen3_tts
    import faster_qwen3_tts.model
    import faster_qwen3_tts.streaming
    from qwen_tts import Qwen3TTSModel

    # 1. Override from_pretrained to skip CUDA Graph compilation on CPU & DirectML
    def custom_from_pretrained(
        cls,
        model_name: str,
        device: str = "cpu",
        dtype = torch.float32,
        attn_implementation: str = "sdpa",
        max_seq_len: int = 2048,
    ):
        device_str = str(device).lower()
        is_dml = "privateuseone" in device_str or "dml" in device_str
        
        if is_dml:
            import torch_directml
            # Parse index if present
            idx = 0
            if ":" in device_str:
                try:
                    idx = int(device_str.split(":")[-1])
                except:
                    pass
            dml_device = torch_directml.device(idx)
            print(f"[Patch] Loading base Qwen3TTSModel: {model_name} on DirectML GPU {dml_device}...")
            
            # Use float16 for DirectML
            dtype = torch.float16
            
            # Load on CPU first, then transfer to DirectML device
            base_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            base_model.model = base_model.model.to(dml_device)
            
            instance = cls.__new__(cls)
            instance.model = base_model
            instance.device = dml_device
            instance.dtype = dtype
            instance.max_seq_len = max_seq_len
            instance.sample_rate = cls._infer_sample_rate(base_model)
            instance._warmed_up = True  # Skip warmup/capture steps
            instance._voice_prompt_cache = {}
            instance.predictor_graph = None
            instance.talker_graph = None
            return instance
            
        else:
            # Force float32 and SDPA for optimal CPU execution
            dtype = torch.float32
            print(f"[Patch] Loading base Qwen3TTSModel: {model_name} on CPU...")
            base_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            
            instance = cls.__new__(cls)
            instance.model = base_model
            instance.device = "cpu"
            instance.dtype = dtype
            instance.max_seq_len = max_seq_len
            instance.sample_rate = cls._infer_sample_rate(base_model)
            instance._warmed_up = True  # Skip warmup/capture steps
            instance._voice_prompt_cache = {}
            instance.predictor_graph = None
            instance.talker_graph = None
            return instance

    faster_qwen3_tts.FasterQwen3TTS.from_pretrained = classmethod(custom_from_pretrained)

    # 2. Wrap parity_generate_streaming to consume and ignore graph arguments
    original_parity_generate_streaming = faster_qwen3_tts.streaming.parity_generate_streaming

    def wrapped_parity_generate_streaming(*args, **kwargs):
        kwargs.pop("predictor_graph", None)
        kwargs.pop("talker_graph", None)
        return original_parity_generate_streaming(*args, **kwargs)

    # 3. Redirect all streaming generation to our wrapped CPU generator
    faster_qwen3_tts.streaming.fast_generate_streaming = wrapped_parity_generate_streaming
    faster_qwen3_tts.streaming.parity_generate_streaming = wrapped_parity_generate_streaming
    faster_qwen3_tts.model.fast_generate_streaming = wrapped_parity_generate_streaming

    print("[Patch] faster-qwen3-tts successfully patched for execution.")

    # 4. Disable engine warmup (only needed for CUDA Graphs)
    from RealtimeTTS.engines.faster_qwen_engine import FasterQwenEngine
    FasterQwenEngine._warmup = lambda self: None
    print("[Patch] FasterQwenEngine warmup disabled.")

except Exception as e:
    print(f"[Patch] Error applying patches: {e}", file=sys.stderr)


# Now we can safely import RealtimeTTS
from fastapi import FastAPI, WebSocket, Response
from pydantic import BaseModel
from RealtimeTTS import TextToAudioStream, FasterQwenEngine
from RealtimeTTS.engines.faster_qwen_engine import FasterQwenVoice

app = FastAPI()

# Retrieve model and device from environment
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
tts_device = os.getenv("TTS_DEVICE", "cpu")

try:
    print(f"Initializing FasterQwenEngine with model: {model_name} on device: {tts_device}...")
    engine = FasterQwenEngine(
        model_name=model_name,
        device=tts_device,
    )
except Exception as e:
    print(f"[Warning] Failed to initialize FasterQwenEngine on device {tts_device}: {e}")
    if tts_device != "cpu":
        print("Falling back to CPU...")
        engine = FasterQwenEngine(
            model_name=model_name,
            device="cpu",
        )
    else:
        raise e

stream = TextToAudioStream(engine, muted=True)

# 4. Initialize and set Default voice
default_pt_path = "/app/default_voice.pt"
default_voice_path = "/app/default_voice.wav"
default_text_path = "/app/default_voice.txt"

if os.path.exists(default_pt_path):
    print(f"Setting default voice profile from precomputed embedding: {default_pt_path}")
    default_voice = FasterQwenVoice(
        name="default",
        speaker_pt=default_pt_path,
        language="English"
    )
    engine.set_voice(default_voice)
elif os.path.exists(default_voice_path) and os.path.exists(default_text_path):
    with open(default_text_path, "r", encoding="utf-8") as f:
        default_text = f.read().strip()
    
    print(f"Setting default voice profile from audio: {default_voice_path}")
    default_voice = FasterQwenVoice(
        name="default",
        ref_audio=default_voice_path,
        ref_text=default_text,
        language="English",
        speaker_pt=default_pt_path
    )
    engine.set_voice(default_voice)
else:
    print("[Warning] Default voice profile files not found.")

# Ensure /app/voices directory exists
voices_dir = "/app/voices"
os.makedirs(voices_dir, exist_ok=True)

print("FasterQwenEngine initialized successfully.")

@app.websocket("/stream-tts")
async def stream_tts(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    
    # Callback to stream generated PCM chunks back to WebSocket
    def on_audio_chunk(chunk):
        print(f"[WS] Sending audio chunk ({len(chunk)} bytes) to client.")
        asyncio.run_coroutine_threadsafe(
            websocket.send_bytes(chunk),
            loop
        )

    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await websocket.receive_json()
            text_chunk = data.get("text", "")
            emotion = data.get("emotion", "")
            selected_voice = data.get("voice", "default")
            
            if not text_chunk:
                continue
                
            # If the client requested a custom voice profile
            if selected_voice != "default":
                voice_pt = os.path.join(voices_dir, f"{selected_voice}.pt")
                voice_wav = os.path.join(voices_dir, f"{selected_voice}.wav")
                voice_txt = os.path.join(voices_dir, f"{selected_voice}.txt")
                
                if os.path.exists(voice_pt):
                    print(f"Switching voice to precomputed: '{selected_voice}'")
                    custom_voice = FasterQwenVoice(
                        name=selected_voice,
                        speaker_pt=voice_pt,
                        language=data.get("language", "English")
                    )
                    await asyncio.to_thread(engine.set_voice, custom_voice)
                elif os.path.exists(voice_wav) and os.path.exists(voice_txt):
                    with open(voice_txt, "r", encoding="utf-8") as f:
                        v_text = f.read().strip()
                    
                    print(f"Switching voice to: '{selected_voice}' (extracting and caching)")
                    custom_voice = FasterQwenVoice(
                        name=selected_voice,
                        ref_audio=voice_wav,
                        ref_text=v_text,
                        language=data.get("language", "English"),
                        speaker_pt=voice_pt
                    )
                    await asyncio.to_thread(engine.set_voice, custom_voice)
                else:
                    print(f"[Warning] Custom voice '{selected_voice}' files not found in {voices_dir}. Using active voice.")

            print(f"Received text chunk: '{text_chunk}' with emotion: '{emotion}'")
            
            if emotion:
                stream.feed(f"[{emotion}] {text_chunk}")
            else:
                stream.feed(text_chunk)
                
            # Run blocking play in a separate thread so event loop stays responsive
            await asyncio.to_thread(
                stream.play,
                muted=True,
                on_audio_chunk=on_audio_chunk,
                fast_sentence_fragment=True
            )
    except Exception as e:
        print(f"Connection closed/error: {e}")

class VapiTTSRequest(BaseModel):
    text: str

    class Config:
        extra = "allow"

tts_lock = asyncio.Lock()

@app.post("/vapi-tts")
async def vapi_tts(request: VapiTTSRequest):
    async with tts_lock:
        chunks = []
        def on_audio_chunk(chunk):
            chunks.append(chunk)

        # Feed text to the stream
        stream.feed(request.text)
        
        # Synthesize audio blocking in worker thread
        await asyncio.to_thread(
            stream.play,
            muted=True,
            on_audio_chunk=on_audio_chunk,
            fast_sentence_fragment=True
        )
        
        raw_bytes = b"".join(chunks)
        if not raw_bytes:
            return Response(content=b"", media_type="audio/l16")
            
        try:
            import numpy as np
            import torchaudio.functional as F
            
            # Convert bytes to tensor
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_tensor = torch.from_numpy(audio_data.copy()).float()
            
            # Resample to 16kHz
            orig_sr = getattr(engine, "sample_rate", 24000)
            if orig_sr != 16000:
                resampled_tensor = F.resample(audio_tensor, orig_freq=orig_sr, new_freq=16000)
            else:
                resampled_tensor = audio_tensor
                
            # Clamp and convert back to int16 bytes
            resampled_tensor = torch.clamp(resampled_tensor, min=-32768, max=32767)
            resampled_data = resampled_tensor.to(torch.int16).numpy()
            resampled_bytes = resampled_data.tobytes()
            
            return Response(content=resampled_bytes, media_type="audio/l16")
        except Exception as e:
            print(f"[Error] Failed to resample/process audio: {e}")
            # Fallback to returning raw bytes as-is
            return Response(content=raw_bytes, media_type="audio/l16")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
