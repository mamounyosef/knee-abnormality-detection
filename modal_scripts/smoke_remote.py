r"""smoke_remote.py - run a run's real training loop for a few steps ON MODAL.

checks/external_run_smoke.py does this locally, but this PC has an 8 GB card and 16 GB of
RAM, so anything at 480 px or 360M parameters either runs out of memory or has to be
shrunk with SMOKE_TINY - which changes the geometry and therefore proves the wiring rather
than the run. This does the same checks at the run's REAL geometry, on the GPU the run will
actually use, for a handful of steps.

What it reports that the local version cannot:
  - the true memory peak with the trunk UNFROZEN, which is the number to size a batch from.
    Epoch 1 is a frozen probe and its peak says nothing; reading it as steady state is what
    made medimageinsight_davit_huge_336 run out of memory at batch 8.
  - throughput at the real window count.

    python -m modal run modal_scripts/smoke_remote.py --run medimageinsight_davit_huge_perslice_480 --steps 4
"""

import sys
from pathlib import Path

# The container re-imports this file as /root/smoke.py, where __file__.parent.parent/"runs"
# is /runs and does not exist - the runs tree is mounted at /root/runs. Try that first, the
# way every train.py does, or the module-level import dies before anything runs.
RUNS = next((p for p in (Path("/root/runs"),
                         Path(__file__).resolve().parent.parent / "runs")
             if (p / "shared_code.py").exists()),
            Path(__file__).resolve().parent.parent / "runs")
sys.path.insert(0, str(RUNS))
import shared_code as sc  # noqa: E402

import modal  # noqa: E402

RUN = "medimageinsight_davit_huge_perslice_480"
cfg = sc.load_config(RUNS / RUN, verbose=False)

app = modal.App("smoke-" + RUN.replace("_", "-"))
_image = sc.modal_image()
_shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
_runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)


# modal_resources already supplies gpu, timeout, scaledown_window, cpu and memory, so do
# NOT pass timeout again here - modal reports it as "got multiple values for keyword
# argument 'timeout'" at import time, before anything runs.
@app.function(image=_image,
              volumes={cfg["modal"]["shards_mount"]: _shards,
                       cfg["modal"]["runs_mount"]: _runs},
              **sc.modal_resources(cfg))
def smoke(run_name: str, steps: int = 4):
    import importlib
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root/runs")
    sys.path.insert(0, "/root/runs/" + run_name)
    import shared_code as s  # noqa: F811

    c = s.remote_cfg(s.load_config(Path("/root/runs") / run_name, verbose=False))
    ok, bad = [], []

    def ck(name, cond, detail=""):
        (ok if cond else bad).append(name)
        print("  %s %-58s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)

    dev = torch.device("cuda")
    print("\n  run %s on %s" % (run_name, torch.cuda.get_device_name(0)))
    i = c["input"]
    print("  geometry: %s px, %s slices, slices_as_channels=%r, %d windows/study, "
          "laterality=%r" % (i["size"], i["n_slices"], i.get("slices_as_channels"),
                             s.images_per_study(c), i.get("laterality_flip")))
    print("  batch %d x accum %d, norm mean %s std %s\n"
          % (c["dataloader"]["batch_size"], c["dataloader"]["accum_steps"],
             i["norm_mean"], i["norm_std"]), flush=True)

    loaders = s.make_loaders(c)
    tm = importlib.import_module("train")
    model = tm.build_model(c).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    ck("model built", n_par > 0, "%.1fM parameters" % (n_par / 1e6))

    batch = next(iter(loaders["train"]))
    xb = s.prepare_batch(batch[0], c, dev)
    yb = batch[1].to(dev).float()
    per = s.images_per_study(c)
    B = c["dataloader"]["batch_size"]
    ck("prepare_batch shape", tuple(xb.shape) == (B * per, i["channels"], i["size"], i["size"]),
       str(tuple(xb.shape)))
    ck("normalised, not raw 0-1", float(xb.min()) < -0.5, "min %.3f max %.3f mean %.3f"
       % (xb.min(), xb.max(), xb.mean()))
    # per-slice: the three channels must hold the same grey plane once normalisation is undone
    if not i.get("slices_as_channels"):
        mean = torch.tensor(i["norm_mean"], device=dev).view(1, -1, 1, 1)
        std = torch.tensor(i["norm_std"], device=dev).view(1, -1, 1, 1)
        raw = xb * std + mean
        ck("per-slice: 3 channels hold the SAME grey plane",
           torch.allclose(raw[0, 0], raw[0, 1], atol=1e-4)
           and torch.allclose(raw[0, 1], raw[0, 2], atol=1e-4))
        ck("consecutive windows hold DIFFERENT slices",
           not torch.allclose(raw[0, 0], raw[1, 0], atol=1e-3))

    # THE POINT OF RUNNING THIS REMOTELY: memory with the trunk UNFROZEN.
    s.set_lp_ft_phase(c, model, epoch=2)
    opt = s.build_optimizer(model, c)
    lossf = torch.nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    watch = [n for n, p in model.named_parameters()
             if n.startswith("backbone.") and p.ndim >= 2][:1]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n in watch}
    losses, t0 = [], time.time()
    it = iter(loaders["train"])
    for st in range(steps):
        b = next(it)
        x = s.prepare_batch(b[0], c, dev)
        y = b[1].to(dev).float()
        with torch.autocast("cuda", enabled=c["training"]["amp"]):
            loss = lossf(model(x), y)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c["training"]["grad_clip"])
        opt.step()
        losses.append(float(loss.detach()))
        print("      step %d  loss %.4f  grad_norm %.3f  peak %.2f GB"
              % (st, losses[-1], gn, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    el = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    ck("no NaN or inf in the loss", all(np.isfinite(losses)))
    ck("the loss moved", abs(losses[-1] - losses[0]) > 1e-4,
       "%.4f -> %.4f" % (losses[0], losses[-1]))
    for n in watch:
        rel = ((dict(model.named_parameters())[n].detach() - before[n]).norm()
               / before[n].norm().clamp_min(1e-12)).item() / max(len(losses), 1)
        ck("the trunk moved (%s)" % n.replace("backbone.", "")[:30], rel > 1e-9,
           "log10(update/weight) per step %.2f" % np.log10(max(rel, 1e-30)))
    print("\n  UNFROZEN PEAK %.2f GB of %.0f GB (%.0f%%) - THIS is the number to size a "
          "batch from" % (peak, total, 100 * peak / total))
    print("  %.1f s for %d steps = %.1f s/step, %.0f images/s"
          % (el, steps, el / steps, steps * B * per / el))
    print("\n  %d passed, %d failed" % (len(ok), len(bad)))
    return {"peak_gb": peak, "total_gb": total, "s_per_step": el / steps,
            "losses": losses, "failed": bad}


@app.local_entrypoint()
def main(run: str = RUN, steps: int = 4):
    r = smoke.remote(run, steps)
    print("\n  %s" % ("SMOKE PASSED" if not r["failed"] else "FAILED: %s" % r["failed"]))
    print("  peak %.2f GB of %.0f GB, %.1f s/step"
          % (r["peak_gb"], r["total_gb"], r["s_per_step"]))
