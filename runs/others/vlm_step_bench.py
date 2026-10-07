"""vlm_step_bench.py - how fast is ONE training update of qwen35_2b_vlm_sys under each attention kernel / checkpointing
setting, on the run's own H100? (7 Oct, the live run shows ~12% MFU)

Builds the run's real model (its train.py, its config) and times full updates on random images: bf16 autocast, BCE on
the 12 logits, backward, AdamW step, 8 studies per update (batch x accum). Per variant: s/update, peak GB, and the max
logit diff vs the first variant on one fixed eval batch (the kernels must agree). Then a CUDA profile (top ops) of
the first variant, so the time sinks are named, not guessed.

    python -m modal run runs/others/vlm_step_bench.py
"""
import sys
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")   # in the container the script sits in /root
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "qwen35_2b_vlm_sys"
app = modal.App("knee-vlm-step-bench")
image = sc.modal_image(extra_pip=("transformers>=5.5", "peft>=0.18", "flash-linear-attention", "kernels>=0.16,<0.17"))

# (name, model.vlm overrides, batch, accum). Round 1 (7 Oct): live 4.21 s/update, 30.8 GB; no checkpointing OOMs even
# at batch 4 x 2 on 80 GB; attention is ~6% of GPU time (cuDNN flash kernels already); fp32->bf16 casts dominate.
F3 = "kernels-community/flash-attn3"
VARIANTS = [
    ("live (sdpa, ckpt, b8)", {}, 8, 1),
    ("bf16 frozen", {"frozen_linear_bf16": True}, 8, 1),
    ("bf16 frozen + no LoRA dropout", {"frozen_linear_bf16": True, "lora_dropout": 0.0}, 8, 1),
    ("bf16 frozen + fa3", {"frozen_linear_bf16": True, "attn_implementation": F3}, 8, 1),
    ("bf16 frozen, no ckpt, b2x4", {"frozen_linear_bf16": True, "grad_checkpointing": False}, 2, 4),
]


@app.function(image=image, gpu="H100", cpu=8.0, memory=64 * 1024, timeout=3600)
def bench() -> str:
    import copy
    import gc
    import importlib.util
    import traceback

    import torch
    spec = importlib.util.spec_from_file_location("vlm_train", f"/root/runs/{RUN}/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    S = int(mod.cfg["input"]["size"])
    g = torch.Generator().manual_seed(0)
    n_img = len(mod.cfg["input"]["slots"]) * int(mod.cfg["input"]["slices_as_channels"])
    xs = torch.rand(8 * n_img, 3, S, S, generator=g) * 2 - 1
    ys = (torch.rand(8, 12, generator=g) > 0.7).float()
    xe = xs[: 2 * n_img].cuda()
    ref = None
    for name, over, bs, acc in VARIANTS:
        cfg = copy.deepcopy(mod.cfg)
        cfg["model"]["vlm"].update(over)
        try:
            torch.manual_seed(0)
            m = mod.build_model(cfg).cuda()
            ai = lambda o: getattr(getattr(o, "config", None), "_attn_implementation", "?")
            say(f"[{name}] attn: text {ai(m.lm)} | vision {ai(m.backbone)}")
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                le = m(xe).float().cpu()
            m.train()
            opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-6)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for it in range(6):
                torch.cuda.synchronize(); t0 = time.time()
                for a in range(acc):
                    x = xs[a * bs * n_img:(a + 1) * bs * n_img].cuda(non_blocking=True)
                    y = ys[a * bs:(a + 1) * bs].cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = m(x)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y) / acc
                    loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); times.append(time.time() - t0)
            t = sum(times[2:]) / len(times[2:])
            d = "" if ref is None else f" | eval logit max diff vs live {(le - ref).abs().max().item():.2e}"
            ref = le if ref is None else ref
            say(f"[{name}] {t:.2f} s/update (8 studies) = {8 * n_img / t:.0f} img/s | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB{d}")
            if name == VARIANTS[1][0]:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = m(xs[: bs * n_img].cuda())
                    torch.nn.functional.binary_cross_entropy_with_logits(out.float(), ys[:bs].cuda()).backward()
                    torch.cuda.synchronize()
                say(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
        except Exception as e:
            say(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            say(traceback.format_exc()[-1500:])
        finally:
            for k in ("m", "opt", "out", "loss"):
                locals().pop(k, None)
            m = opt = out = loss = None
            gc.collect(); torch.cuda.empty_cache()
    return "\n".join(log)


@app.local_entrypoint()
def main():
    out = bench.remote()
    p = Path("C:/temp1/logs/vlm_step_bench.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(out, encoding="utf-8")
    print("\n==== saved", p)
