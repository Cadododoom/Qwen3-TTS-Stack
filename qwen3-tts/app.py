import sys
import torch
from qwen_tts import Qwen3TTSModel

# Save original from_pretrained
original_from_pretrained = Qwen3TTSModel.from_pretrained

# Wrap it to force attn_implementation="sdpa" on CPU
def custom_from_pretrained(*args, **kwargs):
    # Force sdpa attention implementation for CPU compatibility
    kwargs["attn_implementation"] = "sdpa"
    return original_from_pretrained(*args, **kwargs)

Qwen3TTSModel.from_pretrained = custom_from_pretrained

# Now run our local custom demo main
from demo import main
if __name__ == "__main__":
    sys.exit(main())
