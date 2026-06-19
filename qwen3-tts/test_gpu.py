import sys
import types
import torch
import torch_directml
import numpy as np

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

from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

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


# Ensure DirectML device
dml_device = torch_directml.device(2)
print(f"Targeting DML device: {dml_device}")

from qwen_tts import Qwen3TTSModel

print("Loading model on CPU first...")
base_model = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
    device_map='cpu',
    torch_dtype=torch.float32,
    attn_implementation='sdpa'
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

print("Moving model to DML...")
base_model.model.to(dml_device)
base_model.device = dml_device

# Also make sure speaker_encoder is on CPU (since base_model.model.to(dml_device) moves it to DML)
base_model.model.speaker_encoder.to("cpu")

print("Reading voice profile...")
ref_text = open('/app/default_voice.txt').read().strip()

print("Generating voice clone...")
sys.stdout.flush()
wavs, fs = base_model.generate_voice_clone(
    text="This is a test of the Qwen3 TTS model running on the AMD GPU.",
    language="English",
    ref_audio="/app/default_voice.wav",
    ref_text=ref_text,
    non_streaming_mode=True
)

print(f"Success! Generated {len(wavs)} wavs, sample rate: {fs}")
