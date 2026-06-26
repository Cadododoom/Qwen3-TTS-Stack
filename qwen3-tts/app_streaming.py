import os
import sys
import asyncio
import torch
import types
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

# Mock torch.cuda.synchronize as a no-op on CPU to prevent CUDA drivers crash
torch.cuda.synchronize = lambda *args, **kwargs: None

# Monkey patch torch.cat to handle DirectML 0-size tensor concatenation crash
original_cat = torch.cat

def patched_cat(tensors, dim=0, *args, **kwargs):
    if len(tensors) > 0:
        first_tensor = tensors[0]
        actual_dim = dim if dim >= 0 else (first_tensor.dim() + dim)
        filtered_tensors = [t for t in tensors if t.shape[actual_dim] > 0]
        if len(filtered_tensors) == 0:
            return original_cat(tensors, dim, *args, **kwargs)
        if len(filtered_tensors) == 1:
            return filtered_tensors[0]
        return original_cat(filtered_tensors, dim, *args, **kwargs)
    return original_cat(tensors, dim, *args, **kwargs)

torch.cat = patched_cat
print("[Patch] torch.cat patched successfully.")

# Monkey patch RepetitionPenaltyLogitsProcessor to avoid out-of-bounds gather on DirectML
original_repetition_penalty_call = RepetitionPenaltyLogitsProcessor.__call__

def patched_repetition_penalty_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    if self.prompt_ignore_length:
        input_ids = input_ids[:, self.prompt_ignore_length :]

    if scores.dim() == 3:
        if self.logits_indices is not None and self.cu_seq_lens_q is not None:
            last_positions = self.logits_indices
            last_scores = scores[0, last_positions, :]

            # Prepare token mask
            token_mask = torch.zeros_like(last_scores, dtype=torch.bool)
            cu_seq_lens = self.cu_seq_lens_q
            lengths = cu_seq_lens[1:] - cu_seq_lens[:-1]
            seq_indices = torch.repeat_interleave(torch.arange(len(lengths), device=input_ids.device), lengths)
            
            valid_mask = (input_ids >= 0) & (input_ids < last_scores.shape[1])
            clamped_input_ids = torch.where(valid_mask, input_ids, 0)
            token_mask[seq_indices, clamped_input_ids] = valid_mask

            # Apply penalty
            penalty_scores = torch.where(last_scores < 0, last_scores * self.penalty, last_scores / self.penalty)
            scores[0, last_positions, :] = torch.where(token_mask, penalty_scores, last_scores)
        else:
            batch_size, seq_len, vocab_size = scores.shape
            last_scores = scores[:, -1, :]
            token_mask = torch.zeros_like(last_scores, dtype=torch.bool)
            
            valid_mask = (input_ids >= 0) & (input_ids < last_scores.shape[1])
            clamped_input_ids = torch.where(valid_mask, input_ids, 0)
            
            if input_ids.dim() == 1:
                valid_unique = input_ids[(input_ids >= 0) & (input_ids < last_scores.shape[1])]
                if len(valid_unique) > 0:
                    token_mask.scatter_(1, torch.unique(valid_unique).unsqueeze(0), True)
            else:
                token_mask.scatter_(1, clamped_input_ids, valid_mask)
                
            penalty_scores = torch.where(last_scores < 0, last_scores * self.penalty, last_scores / self.penalty)
            scores[:, -1, :] = torch.where(token_mask, penalty_scores, last_scores)
        return scores

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(1)

    vocab_size = scores.shape[1]
    valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
    clamped_input_ids = torch.where(valid_mask, input_ids, 0)
    
    token_mask = torch.zeros_like(scores, dtype=torch.bool)
    token_mask.scatter_(1, clamped_input_ids, valid_mask)
    
    penalized_scores = torch.where(scores < 0, scores * self.penalty, scores / self.penalty)
    scores_processed = torch.where(token_mask, penalized_scores, scores)
    return scores_processed

RepetitionPenaltyLogitsProcessor.__call__ = patched_repetition_penalty_call
print("[Patch] RepetitionPenaltyLogitsProcessor patched successfully.")


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
        
        is_cuda = "cuda" in device_str
        
        if is_cuda:
            # Load on GPU (ROCm)
            dtype = torch.float16
            print(f"[Patch] Loading base Qwen3TTSModel: {model_name} on CUDA/ROCm GPU {device_str}...")
            sys.stdout.flush()
            
            base_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device_str,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            
            instance = cls.__new__(cls)
            instance.model = base_model
            instance.device = torch.device(device_str)
            instance.dtype = dtype
            instance.max_seq_len = max_seq_len
            instance.sample_rate = cls._infer_sample_rate(base_model)
            instance._warmed_up = True  # Skip warmup/capture steps
            instance._voice_prompt_cache = {}
            instance.predictor_graph = None
            instance.talker_graph = None
            return instance
            
        elif is_dml:
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
            dtype = torch.float32
            
            # Load on CPU first, then transfer to DirectML device
            base_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            
            # Patch extract_speaker_embedding to force CPU execution for speaker_encoder
            def patched_extract_speaker_embedding(self, audio, sr):
                from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
                assert sr == 24000, "Only support 24kHz audio"
                mels = mel_spectrogram(
                    torch.from_numpy(audio).unsqueeze(0), 
                    n_fft=1024, 
                    num_mels=128, 
                    sampling_rate=24000,
                    hop_size=256, 
                    win_size=1024, 
                    fmin=0, 
                    fmax=12000
                ).transpose(1, 2)
                
                # Run on CPU
                mels = mels.to("cpu").to(self.dtype)
                self.speaker_encoder.to("cpu")
                
                print("[Patch] Running speaker_encoder on CPU...")
                sys.stdout.flush()
                speaker_embedding = self.speaker_encoder(mels)[0]
                
                # Return moved to target device
                return speaker_embedding.to(self.device)

            base_model.model.extract_speaker_embedding = types.MethodType(patched_extract_speaker_embedding, base_model.model)
            print("[Patch] extract_speaker_embedding patched successfully.")

            # Move base_model.model to DirectML device layer-by-layer to avoid WDDM TDR timeout
            print("[Patch] Transferring base_model.model to DirectML device layer-by-layer...")
            for name, child in base_model.model.named_children():
                if name != "layers" and name != "speaker_encoder":
                    print(f"[Patch] Moving child module: {name} ({type(child)}) to DirectML {dml_device}...")
                    sys.stdout.flush()
                    try:
                        child.to(dml_device)
                        print(f"[Patch] Child module: {name} successfully moved.")
                    except Exception as ex:
                        print(f"[Patch] Failed to move child {name}: {ex}")
                        import traceback; traceback.print_exc()
                    sys.stdout.flush()
            if hasattr(base_model.model, "layers"):
                for idx, layer in enumerate(base_model.model.layers):
                    print(f"[Patch] Moving transformer layer: {idx} to DirectML...")
                    sys.stdout.flush()
                    base_model.model.layers[idx] = layer.to(dml_device)
                    print(f"[Patch] Transformer layer: {idx} successfully moved.")
                    sys.stdout.flush()
            
            # Move base_model.model to DirectML but keep speaker_encoder on CPU
            base_model.model = base_model.model.to(dml_device)
            base_model.model.speaker_encoder.to("cpu")
            
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
    import traceback
    traceback.print_exc()
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

class OpenAITTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0

    class Config:
        extra = "allow"

@app.post("/v1/audio/speech")
async def openai_tts(request: OpenAITTSRequest):
    async with tts_lock:
        chunks = []
        def on_audio_chunk(chunk):
            chunks.append(chunk)

        # Handle voice selection
        selected_voice = request.voice
        if selected_voice != "default":
            voice_pt = os.path.join(voices_dir, f"{selected_voice}.pt")
            voice_wav = os.path.join(voices_dir, f"{selected_voice}.wav")
            voice_txt = os.path.join(voices_dir, f"{selected_voice}.txt")
            
            if os.path.exists(voice_pt):
                print(f"[OpenAI-TTS] Switching voice to precomputed: '{selected_voice}'")
                custom_voice = FasterQwenVoice(
                    name=selected_voice,
                    speaker_pt=voice_pt,
                    language="English"
                )
                await asyncio.to_thread(engine.set_voice, custom_voice)
            elif os.path.exists(voice_wav) and os.path.exists(voice_txt):
                with open(voice_txt, "r", encoding="utf-8") as f:
                    v_text = f.read().strip()
                
                print(f"[OpenAI-TTS] Switching voice to: '{selected_voice}' (extracting and caching)")
                custom_voice = FasterQwenVoice(
                    name=selected_voice,
                    ref_audio=voice_wav,
                    ref_text=v_text,
                    language="English",
                    speaker_pt=voice_pt
                )
                await asyncio.to_thread(engine.set_voice, custom_voice)
            else:
                print(f"[Warning] Custom voice '{selected_voice}' files not found in {voices_dir}. Using default.")
                if 'default_voice' in globals():
                    await asyncio.to_thread(engine.set_voice, default_voice)

        print(f"[OpenAI-TTS] Received text chunk: '{request.input}'")
        stream.feed(request.input)
        
        # Synthesize audio blocking in worker thread
        await asyncio.to_thread(
            stream.play,
            muted=True,
            on_audio_chunk=on_audio_chunk,
            fast_sentence_fragment=True
        )
        
        raw_bytes = b"".join(chunks)
        if not raw_bytes:
            return Response(content=b"", media_type="audio/wav")
            
        try:
            import struct
            sample_rate = getattr(engine, "sample_rate", 24000)
            channels = 1
            bits_per_sample = 16
            
            num_samples = len(raw_bytes) // (bits_per_sample // 8)
            data_size = num_samples * channels * (bits_per_sample // 8)
            file_size = data_size + 36
            
            # WAV Header structure (44 bytes)
            header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF',
                file_size,
                b'WAVE',
                b'fmt ',
                16, # Subchunk1Size
                1,  # AudioFormat (1 = PCM)
                channels,
                sample_rate,
                sample_rate * channels * (bits_per_sample // 8), # ByteRate
                channels * (bits_per_sample // 8),               # BlockAlign
                bits_per_sample,
                b'data',
                data_size
            )
            
            wav_bytes = header + raw_bytes
            
            media_type = "audio/wav" if request.response_format == "wav" else "audio/mpeg"
            return Response(content=wav_bytes, media_type=media_type)
        except Exception as e:
            print(f"[Error] Failed to create WAV container: {e}")
            return Response(content=raw_bytes, media_type="audio/l16")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
