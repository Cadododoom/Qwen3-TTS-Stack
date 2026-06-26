import os
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    print("Loading model on CPU...")
    # Force sdpa attention implementation for CPU compatibility
    model = Qwen3TTSModel.from_pretrained(
        'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
        device_map='cpu',
        torch_dtype=torch.float32,
        attn_implementation='sdpa'
    )
    print("Model loaded successfully.")

    # Speaker A reference (default_voice.wav)
    ref_a = "/app/default_voice.wav"
    ref_a_text = open("/app/default_voice.txt").read().strip()
    
    # Speaker B reference (readme_clone_input.wav)
    ref_b = "/app/readme_clone_input.wav"

    # Define the conversation script
    dialogue = [
        {"speaker": "A", "text": "Hello! Welcome to the Qwen3 Text-to-Speech system demonstration. This conversation is running entirely inside a Docker container.", "ref": ref_a, "ref_text": ref_a_text, "x_vector_only": False},
        {"speaker": "B", "text": "Hi there! This is amazing. The audio is incredibly clear, and the voice cloning preserves the acoustic details perfectly.", "ref": ref_b, "ref_text": None, "x_vector_only": True},
        {"speaker": "A", "text": "Exactly. We can design consistent personas and lock them in for a seamless user experience. How does this sound to you?", "ref": ref_a, "ref_text": ref_a_text, "x_vector_only": False},
        {"speaker": "B", "text": "It sounds fantastic! Truly professional and natural.", "ref": ref_b, "ref_text": None, "x_vector_only": True}
    ]

    fs = 24000
    all_segments = []

    for i, turn in enumerate(dialogue):
        spk = turn["speaker"]
        txt = turn["text"]
        ref = turn["ref"]
        ref_txt = turn["ref_text"]
        x_only = turn["x_vector_only"]

        print(f"Synthesizing turn {i+1}/4 (Speaker {spk}): \"{txt}\"")
        wavs, sample_rate = model.generate_voice_clone(
            text=txt,
            language="English",
            ref_audio=ref,
            ref_text=ref_txt,
            x_vector_only_mode=x_only,
            non_streaming_mode=True
        )
        # wavs is a list, get first item
        segment = wavs[0]
        # Normalize just in case
        if segment.ndim > 1:
            segment = np.mean(segment, axis=-1)
        
        all_segments.append(segment)
        
        # Add 0.5s silence after turn (except last one)
        if i < len(dialogue) - 1:
            silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
            all_segments.append(silence)
            
        fs = sample_rate

    print("Concatenating segments...")
    final_audio = np.concatenate(all_segments)
    
    # Save the output file
    output_path = "/app/conversation_test.wav"
    sf.write(output_path, final_audio, fs)
    print(f"Successfully saved final conversation to {output_path}")

if __name__ == "__main__":
    main()
