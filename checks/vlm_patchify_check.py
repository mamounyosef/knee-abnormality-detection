"""vlm_patchify_check.py - does qwen35_2b_vlm's in-model patching equal Qwen's own image processor, byte for byte?

CPU only (no model weights): random 384 px images go through (a) the real Qwen3.5 processor and (b) the run's
_patchify on the engine's normalisation ((x/255 - 0.5) / 0.5); pixel_values and grid_thw must match. Also prints the
prompt length for the run's config (model.vlm.n_tokens) and checks mm_token_type_ids marks exactly the image tokens.

    C:/temp1/vlm_env/Scripts/python.exe checks/vlm_patchify_check.py
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vlm_train", ROOT / "runs" / "qwen35_2b_vlm" / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.cfg
proc = AutoProcessor.from_pretrained(mod.VLM_ID)
S = int(cfg["input"]["size"])
n = len(cfg["input"]["slots"]) * int(cfg["input"]["slices_as_channels"])
rng = np.random.default_rng(0)
arr = rng.integers(0, 256, (n, S, S, 3), dtype=np.uint8)
msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": cfg["model"]["vlm"]["prompt"]}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
enc = proc(text=[text], images=[Image.fromarray(a) for a in arr], return_tensors="pt")
x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
mean = torch.tensor(cfg["input"]["norm_mean"]).view(1, 3, 1, 1)
std = torch.tensor(cfg["input"]["norm_std"]).view(1, 3, 1, 1)
pv, grid = mod._patchify((x - mean) / std)
ok = True
d = (pv - enc["pixel_values"].float()).abs().max().item()
print(f"pixel_values {tuple(pv.shape)} vs processor {tuple(enc['pixel_values'].shape)}: max diff {d:.2e}")
ok &= pv.shape == enc["pixel_values"].shape and d < 1e-5
ok &= torch.equal(grid.long(), enc["image_grid_thw"].long())
print("grid_thw equal:", torch.equal(grid.long(), enc["image_grid_thw"].long()), grid[0].tolist())
ids, mtt = enc["input_ids"], enc["mm_token_type_ids"]
img_tok = int(proc.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
print(f"prompt tokens {ids.shape[1]} (config n_tokens {cfg['model']['vlm']['n_tokens']}); image tokens {(ids == img_tok).sum().item()} "
      f"= {n} x {(ids == img_tok).sum().item() // n}; mm_token_type_ids marks {int((mtt > 0).sum())}")
ok &= ids.shape[1] == int(cfg["model"]["vlm"]["n_tokens"])
ok &= torch.equal((mtt > 0), (ids == img_tok))
print("ALL OK" if ok else "FAILED")
