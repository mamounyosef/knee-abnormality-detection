"""vlm_speed_bench.py - how fast is a full Qwen3.5-2B (vision encoder + language model) forward per knee study?

Runs in the isolated env (transformers 5): C:/temp1/vlm_env/Scripts/python.exe runs/others/vlm_speed_bench.py
Per study the model would see N grey MRI images at 384 px (3 slices as RGB, as our 2.5D models) plus a short prompt;
the classifier would read the language model's last hidden states (no text generation). Times fp16 inference at
N = 8, 16, 24, reports tokens, s/study and peak VRAM, and the projected hours for ~1300 hidden studies on Kaggle's
T4 (assumed 2.0x slower than this RTX 4060 for fp16 transformer inference; a rough factor, to be measured on Kaggle).
"""
import os
import time
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MID = "Qwen/Qwen3.5-2B"
proc = AutoProcessor.from_pretrained(MID)
model = Qwen3_5ForConditionalGeneration.from_pretrained(MID, dtype=torch.float16).cuda().eval()
print("params", sum(p.numel() for p in model.parameters()) / 1e9, "B", flush=True)
rng = np.random.default_rng(0)
for n in (8, 16, 24):
    imgs = [Image.fromarray(rng.integers(0, 255, (384, 384, 3), dtype=np.uint8)) for _ in range(n)]
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": "Knee MRI. Findings?"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].half()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(4):
            if i == 1:
                torch.cuda.synchronize(); t0 = time.time()
            out = model(**inp, output_hidden_states=True, logits_to_keep=1)
        torch.cuda.synchronize()
    s = (time.time() - t0) / 3
    print(f"N={n:2d} images: {inp['input_ids'].shape[1]} tokens | {s:.3f} s/study on this GPU | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
          f"| T4 est. {s * 2.0:.2f} s/study -> {s * 2.0 * 1300 / 3600:.2f} h for 1300 studies on ONE T4 ({s * 2.0 * 1300 / 3600 / 2:.2f} h on two)", flush=True)
