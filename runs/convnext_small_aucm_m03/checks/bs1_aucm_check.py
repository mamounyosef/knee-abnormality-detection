"""Does MultiLabelAUCMLoss still mean anything at micro-batch 1?

aucm_imratio is null, so libauc estimates the class prior from EACH mini-batch. With
one study per micro-batch a task sees exactly one label, so it is either all-positive
or all-negative and the prior is 0 or 1. accum_steps does not rescue this - the loss is
computed per micro-batch, before accumulation. This drives the real wrapper at batch
1, 2 and 8 and reports what comes out.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/root/runs" if Path("/root/runs/shared_code.py").exists() else "runs")
import shared_code as sc
import modal
app = modal.App("bs1-aucm")


@app.function(image=sc.modal_image(), timeout=900)
def check():
    import sys, torch, math
    sys.path.insert(0, "/root/runs")
    import shared_code as sc
    dev = torch.device("cpu")
    T = 12
    cfg = {"tasks": [f"t{i}" for i in range(T)], "target": {"hard_threshold": 0.5},
           "training": {"loss": {"name": "aucm_official", "aucm_margin": 1.0,
                                 "aucm_version": "v1", "aucm_imratio": None,
                                 "pos_weight": None}}}
    torch.manual_seed(0)
    for bs in (1, 2, 8):
        loss = sc.build_loss(cfg, None, dev)
        vals, grads = [], []
        for trial in range(60):
            logits = torch.randn(bs, T, requires_grad=True)
            y = (torch.rand(bs, T) > 0.6).float()
            l = loss(logits, y)
            l.backward()
            v = float(l)
            vals.append(v)
            g = logits.grad
            grads.append(float(g.abs().sum()) if g is not None else 0.0)
        nan = sum(1 for v in vals if not math.isfinite(v))
        zerog = sum(1 for g in grads if g == 0.0)
        finite = [v for v in vals if math.isfinite(v)]
        mean = sum(finite) / max(len(finite), 1)
        print(f"  batch {bs}:  non-finite loss {nan}/60   zero-gradient steps "
              f"{zerog}/60   mean loss {mean:.4f}")
    return "done"


@app.local_entrypoint()
def main():
    print(check.remote())
