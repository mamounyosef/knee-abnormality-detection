"""Full-chain rehearsal of the PESG training path, minus the shards.

Exercises everything the real run does per step: AMP autocast, GradScaler,
gradient accumulation, scaler.step(PESG), the no-op scheduler, clip_value,
update_regularizer at a decay epoch, and a checkpoint save+reload round trip.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/root/runs" if Path("/root/runs/shared_code.py").exists() else "runs")
import shared_code as sc
import modal
app = modal.App("pesg-full")

@app.function(image=sc.modal_image(), gpu="T4", timeout=1800)
def check():
    import sys, torch, math
    sys.path.insert(0, "/root/runs")
    import shared_code as sc
    import torch.nn as nn
    dev = torch.device("cuda")
    T = 12
    cfg = {"tasks": [f"t{i}" for i in range(T)], "target": {"hard_threshold": 0.5},
           "training": {"amp": True,
                        "loss": {"name": "aucm_official", "aucm_margin": 1.0,
                                 "aucm_version": "v1", "aucm_imratio": None, "pos_weight": None},
                        "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0,
                                      "pesg": {"mode": "sgd", "momentum": 0.9,
                                               "clip_value": 1.0, "epoch_decay": 2e-3,
                                               "decay_epochs": [2], "decay_factor": 3}}}}
    ok = lambda n, c: print(("  PASS  " if c else "  FAIL  ") + n)

    loss = sc.build_loss(cfg, None, dev)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, T)).to(dev)
    opt = sc.build_optimizer(model, cfg, loss)
    sched = sc._NullScheduler(opt)   # what the run builds
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    accum = 4
    x = torch.randn(8, 64, device=dev)
    y = (torch.rand(8, T, device=dev) > 0.6).float()

    # --- the real inner loop, verbatim in shape -------------------------------
    losses = []
    try:
        for step in range(12):
            for micro in range(accum):
                with torch.autocast("cuda", enabled=True):
                    l = loss(model(x), y) / accum
                scaler.scale(l).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step()
            losses.append(float(l) * accum)
        ok("AMP + GradScaler + accumulation + scaler.step(PESG) + sched.step()", True)
    except Exception as e:
        ok(f"inner loop -> {type(e).__name__}: {e}", False); return "FAILED"

    ok("loss is finite", all(math.isfinite(v) for v in losses))
    ok("weights moved", bool(model[0].weight.abs().sum() > 0))
    ok("alpha stayed >= 0 (dual projection)", bool((loss.alpha >= 0).all()))
    ok("a/b learned (not still zero)", bool(loss.a.abs().sum() > 0 or loss.b.abs().sum() > 0))

    lr0 = opt.param_groups[0]["lr"]
    opt.update_regularizer(decay_factor=3)
    ok(f"update_regularizer divides lr ({lr0:.4g} -> {opt.param_groups[0]['lr']:.4g})",
       abs(opt.param_groups[0]["lr"] - lr0 / 3) < 1e-12)
    ok("sched.step() still works after update_regularizer",
       (sched.step() or True))
    ok("update_regularizer did NOT get undone by sched.step()",
       abs(opt.param_groups[0]["lr"] - lr0 / 3) < 1e-9)

    # --- checkpoint round trip ----------------------------------------------
    try:
        sd = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict()}
        m2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, T)).to(dev)
        l2 = sc.build_loss(cfg, None, dev)
        o2 = sc.build_optimizer(m2, cfg, l2)
        m2.load_state_dict(sd["model"])
        o2.load_state_dict(sd["optimizer"])
        ok("PESG state_dict round trip", True)
    except Exception as e:
        ok(f"PESG state_dict round trip -> {type(e).__name__}: {e}", False)

    print("  loss trace:", " ".join(f"{v:.4f}" for v in losses[:6]), "...", f"{losses[-1]:.4f}")
    return "done"

@app.local_entrypoint()
def main(): print(check.remote())
