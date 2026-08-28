"""Rehearsal of a WORKSPACE HAND-OFF mid AUC-M run.

The dangerous case: the run is at epoch 10 with reinit_head still true in the config,
credits run out, and it restarts elsewhere from its own last.pt. If the engine treats
that checkpoint like the stage-1 hand-off it re-randomises the head and resets the
epoch counter - silently throwing 10 epochs away. This drives the real save/load path
and asserts the opposite.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/root/runs" if Path("/root/runs/shared_code.py").exists() else "runs")
import shared_code as sc
import modal
app = modal.App("handoff-check")


@app.function(image=sc.modal_image(), timeout=1800)
def check():
    import sys, torch
    sys.path.insert(0, "/root/runs")
    import shared_code as sc
    import torch.nn as nn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP = dev.type == "cuda"
    T = 12
    ok = lambda n, c: print(("  PASS  " if c else "  FAIL  ") + n)
    fails = []
    def want(n, c):
        ok(n, c)
        if not c: fails.append(n)

    def mkcfg(loss_name):
        return {"tasks": [f"t{i}" for i in range(T)], "target": {"hard_threshold": 0.5},
                "training": {"amp": True, "reinit_head": True,
                             "loss": {"name": loss_name, "aucm_margin": 1.0,
                                      "aucm_version": "v1", "aucm_imratio": None,
                                      "pos_weight": None},
                             "optimizer": {"name": "adamw", "lr": 1e-2,
                                           "weight_decay": 0.0,
                                           "pesg": {"mode": "sgd", "momentum": 0.9,
                                                    "clip_value": 1.0,
                                                    "epoch_decay": 2e-3,
                                                    "decay_epochs": [2],
                                                    "decay_factor": 3}}}}

    cfg = mkcfg("aucm_official")
    loss = sc.build_loss(cfg, None, dev)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.backbone = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
            s.head = nn.Linear(64, T)
        def forward(s, x): return s.head(s.backbone(x))

    model = Net().to(dev)
    opt = sc.build_optimizer(model, cfg, loss)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    x = torch.randn(8, 64, device=dev)
    y = (torch.rand(8, T, device=dev) > 0.6).float()

    # train a bit so head weights, PESG momentum and the duals are all non-trivial
    for _ in range(8):
        with torch.autocast(dev.type, enabled=AMP):
            l = loss(model(x), y)
        scaler.scale(l).backward(); scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)

    head_before = model.head.weight.detach().clone()
    duals_before = {k: v.clone() for k, v in loss.state().items()}
    want("duals are non-zero before the save",
         any(float(v.abs().sum()) > 0 for v in duals_before.values()))

    d = Path("/tmp/hoff"); d.mkdir(exist_ok=True)
    sc.save_checkpoint(d / "last.pt", cfg, model, opt, None, scaler, epoch=10, step=4400,
                       score=0.8364, track={"best": 0.8364, "best_epoch": 10, "bad": 0},
                       device=dev, loss_fn=loss)
    ck_raw = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    want("checkpoint carries the AUC-M duals", "loss" in ck_raw and bool(ck_raw["loss"]))

    # ---- the hand-off: brand new process, same config, resume from our own last.pt
    loss2 = sc.build_loss(cfg, None, dev)
    model2 = Net().to(dev)
    opt2 = sc.build_optimizer(model2, cfg, loss2)
    ck = sc.load_checkpoint("last", model2, d, None, None, None, dev, cfg_now=cfg)

    same = (((ck.get("cfg") or {}).get("training") or {}).get("loss") or {}).get("name") \
        == cfg["training"]["loss"]["name"]
    want("hand-off is detected as SAME stage (head must survive)", same)

    if same and ck.get("optimizer"):
        opt2.load_state_dict(ck["optimizer"])
    if same and ck.get("loss"):
        loss2.load_state(ck["loss"])

    want("head weights survived the hand-off byte-for-byte",
         torch.equal(model2.head.weight.detach().cpu(), head_before.cpu()))
    want("duals restored exactly",
         all(torch.equal(loss2.state()[k], v) for k, v in duals_before.items()))
    want("epoch/step preserved", ck.get("epoch") == 10 and ck.get("step") == 4400)
    want("best score preserved", abs((ck.get("track") or {}).get("best", 0) - 0.8364) < 1e-6)

    # PESG momentum must have come back, not restarted at zero
    st = opt2.state_dict().get("state", {})
    want("PESG optimizer state restored", bool(st))

    # ---- and the stage-1 case must STILL reinit
    ck1 = dict(ck); ck1["cfg"] = mkcfg("bce")
    same1 = (((ck1.get("cfg") or {}).get("training") or {}).get("loss") or {}).get("name") \
        == cfg["training"]["loss"]["name"]
    want("stage-1 checkpoint is still detected as a DIFFERENT stage (head reinit fires)",
         not same1)

    # training continues after the hand-off
    try:
        for _ in range(3):
            with torch.autocast(dev.type, enabled=AMP):
                l2 = loss2(model2(x), y)
            scaler.scale(l2).backward(); scaler.step(opt2); scaler.update()
            opt2.zero_grad(set_to_none=True)
        want("training continues after the hand-off", bool(torch.isfinite(l2)))
    except Exception as e:
        want(f"training continues after the hand-off -> {type(e).__name__}: {e}", False)

    print("\n" + ("ALL PASS" if not fails else "FAILURES: " + "; ".join(fails)))
    return "OK" if not fails else "FAILED"


@app.local_entrypoint()
def main():
    print(check.remote())
