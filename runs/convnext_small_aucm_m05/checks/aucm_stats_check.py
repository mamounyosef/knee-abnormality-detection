"""Does AUCMOfficial.stats() actually report the duals, and do they move?"""
import sys
from pathlib import Path
sys.path.insert(0, "/root/runs" if Path("/root/runs/shared_code.py").exists() else "runs")
import shared_code as sc
import modal
app = modal.App("aucm-stats")


@app.function(image=sc.modal_image(), timeout=900)
def check():
    import sys, torch
    sys.path.insert(0, "/root/runs")
    import shared_code as sc
    dev = torch.device("cpu")
    tasks = [f"t{i}" for i in range(12)]
    cfg = {"tasks": tasks, "target": {"hard_threshold": 0.5},
           "training": {"loss": {"name": "aucm_official", "aucm_margin": 1.0,
                                 "aucm_version": "v1", "aucm_imratio": None,
                                 "pos_weight": None},
                        "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0,
                                      "pesg": {"mode": "sgd", "momentum": 0.9,
                                               "clip_value": 1.0, "epoch_decay": 2e-3}}}}
    torch.manual_seed(0)
    loss = sc.build_loss(cfg, None, dev)
    model = torch.nn.Linear(32, 12)
    opt = sc.build_optimizer(model, cfg, loss)
    first = loss.stats(tasks)
    print("  keys:", len(first), "| sample:",
          {k: round(v, 4) for k, v in list(first.items())[:4]})
    for _ in range(120):
        x = torch.randn(16, 32)
        y = (torch.rand(16, 12) > 0.6).float()
        loss(model(x), y).backward(); opt.step(); opt.zero_grad()
    last = loss.stats(tasks)
    print(f"  alpha mean {first['aucm_alpha_mean']:.5f} -> {last['aucm_alpha_mean']:.5f}")
    print(f"  gap   mean {first['aucm_gap_mean']:.5f} -> {last['aucm_gap_mean']:.5f}")
    per = [k for k in last if k.startswith("aucm_alpha__")]
    ok = lambda n, c: print(("  PASS  " if c else "  FAIL  ") + n)
    ok(f"per-task columns present ({len(per)} of 12)", len(per) == 12)
    ok("alpha moved off its init", last["aucm_alpha_mean"] != first["aucm_alpha_mean"])
    ok("alpha stayed non-negative", last["aucm_alpha_min"] >= 0)
    ok("a/b separated (gap non-zero)", abs(last["aucm_gap_mean"]) > 1e-6)
    return "done"


@app.local_entrypoint()
def main():
    print(check.remote())
