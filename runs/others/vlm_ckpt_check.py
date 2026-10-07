"""vlm_ckpt_check.py - do qwen35_2b_vlm_sys's slim checkpoints (trained tensors only) round-trip exactly, and does the
old full 10.9 GB last.pt still resume? CPU container on the workspace that holds the run's volume. (7 Oct)

  1. engine save_checkpoint of a fresh model: file size + key count (frozen LM must be absent)
  2. a second model (different random head) loads it strict: every saved tensor equal
  4. (first) gradients with the run's grad checkpointing: one study forward/backward; every trainable group (vision
     encoder blocks, LoRA, head) must get non-zero grads (transformers warns the vision model exposes no input
     embeddings, which would silently zero the vision grads under reentrant checkpointing)
  3. the volume's last.pt (written before the slim state_dict) loads strict into a fresh model; its trained tensors
     equal the file's; reports its step

    python -m modal run runs/others/vlm_ckpt_check.py
"""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "qwen35_2b_vlm_sys"
cfg0 = sc.load_config(ROOT / "runs" / RUN, verbose=False)
app = modal.App("knee-vlm-ckpt-check")
image = sc.modal_image(extra_pip=("transformers>=5.5", "peft>=0.18", "flash-linear-attention", "kernels>=0.16,<0.17"))
runs = modal.Volume.from_name(cfg0["modal"]["runs_volume"])


@app.function(image=image, gpu="L4", cpu=8.0, memory=80 * 1024, timeout=3600, volumes={cfg0["modal"]["runs_mount"]: runs})
def check() -> str:
    import importlib.util
    import os
    import tempfile

    import torch
    spec = importlib.util.spec_from_file_location("vlm_train", f"/root/runs/{RUN}/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out, ok = [], True

    def say(s):
        print(s, flush=True)
        out.append(s)

    torch.manual_seed(0)
    a = mod.build_model(mod.cfg)
    a.cuda().train()
    S = int(mod.cfg["input"]["size"])
    x = torch.rand(a.n_img, 3, S, S, device="cuda") * 2 - 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = a(x)
    lo.float().sum().backward()
    groups = {"vision first block": "backbone.blocks.0.", "vision last block": f"backbone.blocks.{len(a.backbone.blocks) - 1}.",
              "vision merger": "backbone.merger", "LoRA first layer": "lm.base_model.model.layers.0.",
              "LoRA last layer": "lm.base_model.model.layers.23.", "head": "head."}
    for gname, pre in groups.items():
        # lora_A is exempt: a fresh LoRA has B = 0, so dL/dA = B^T(...) = 0 exactly at step 0 (B moves first)
        ps = [(n, q) for n, q in a.named_parameters() if n.startswith(pre) and q.requires_grad and "lora_A" not in n]
        nz = sum(int(q.grad is not None and q.grad.abs().sum().item() > 0) for _, q in ps)
        say(f"4. grads {gname:20s}: {nz}/{len(ps)} trainable tensors non-zero")
        ok &= len(ps) > 0 and nz == len(ps)
    a.zero_grad(set_to_none=True)
    a.cpu()
    torch.cuda.empty_cache()
    sd = a.state_dict()
    n_full = len(torch.nn.Module.state_dict(a))
    lm_frozen = [k for k in sd if k.startswith("lm.") and "lora_" not in k]
    say(f"1. state_dict: {len(sd)} keys (full module {n_full}); frozen LM keys saved: {len(lm_frozen)}")
    ok &= not lm_frozen
    p = Path(tempfile.mkdtemp()) / "t.pt"
    sc.save_checkpoint(p, mod.cfg, a, epoch=1, step=1, score=0.5)
    say(f"   engine save_checkpoint: {p.stat().st_size / 2**30:.2f} GB on disk")
    torch.manual_seed(1)
    b = mod.build_model(mod.cfg)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    b.load_state_dict(ck["model"], strict=True)
    sb = b.state_dict()
    diff = max((sd[k].float() - sb[k].float()).abs().max().item() for k in sd)
    say(f"2. round trip into a fresh model: same keys {set(sd) == set(sb)}, max diff {diff:.1e}")
    ok &= set(sd) == set(sb) and diff == 0
    last = Path(mod.cfg["modal"]["runs_mount"]) / RUN / "results" / "checkpoints" / "last.pt"
    if last.exists():
        say(f"3. volume last.pt: {last.stat().st_size / 2**30:.1f} GB")
        old = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
        torch.manual_seed(2)
        c = mod.build_model(mod.cfg)
        c.load_state_dict(old["model"], strict=True)
        sc_ = c.state_dict()
        d3 = max((old["model"][k].float() - sc_[k].float()).abs().max().item() for k in sc_)
        say(f"   loads strict; trained tensors vs file max diff {d3:.1e}; step {old.get('step') or old.get('global_step')}, "
            f"epoch {old.get('epoch')}, score {old.get('score')}; optimizer state present {'optimizer' in old}")
        ok &= d3 == 0
    else:
        say("3. no last.pt on this volume yet")
        ok = False
    say("ALL OK" if ok else "FAILED")
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(check.remote())
