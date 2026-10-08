"""make_local_channel_runs.py - the three LOCAL (RTX 4060) input-representation runs of 8 Oct, copies of
convnext_base_25d_18slices_v5_aug_base (v4 labels, v5 split, flip; val 0.9174 / audit 0.9209 / LB 0.930) with ONE change
each, on the local recipe of convnext_base_25d_18slices_v4_flip_f0 (batch 2 x accum 4 = the same 8 studies per update,
grad checkpointing, 1 worker, keep 1 checkpoint, checkpoints on C: via a junction).

  convnext_base_25d_v5_multiwindow   each image = ONE slice in 3 windows [1-80] / [1-99] / [60-99.8] percentile
  convnext_base_25d_v5_slab_mmm      each image = [MinIP, mean, MIP] over every stored slice of one bin
  convnext_base_25d_v5_randconv      augmentation.randconv p 0.25 per slot, kernels 1/3/5, alpha 0.1-0.5

    python runs/others/make_local_channel_runs.py
"""
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs"
DONOR = "convnext_base_25d_18slices_v5_aug_base"
CK = Path(r"C:\temp1\local_ckpts")

LOCAL = {
    "  batch_size: 8 ": "  batch_size: 2                     # LOCAL: 2 x accum_steps 4 = the same 8 studies per update\n",
    "  accum_steps: 1 ": "  accum_steps: 4                    # effective 8 studies per update\n",
    "  val_batch_size: 16 ": "  val_batch_size: 4                 # local\n",
    "  num_workers: 26 ": "  num_workers: 1                    # local PC: 16 GB RAM ran out with 2 (30 Sep)\n",
    "  val_num_workers: null ": "  val_num_workers: 1\n",
    "  prefetch_factor: 6 ": "  prefetch_factor: 2                # batches queued per worker\n",
    "  pin_memory: true ": "  pin_memory: false                 # local: no page-locked memory\n",
}
OUTPUT = ("\noutput:\n  keep_last_n: 1                    # local disk: last.pt covers resume\n  keep_top_k: 1\n"
          "  checkpoint_every_steps: 100       # a crash loses at most 100 steps\n")

RUNS = {
    "convnext_base_25d_v5_multiwindow": (
        "MULTI-WINDOW CHANNELS: each image is ONE slice shown through 3 intensity windows (dark tissue p1-p80, today's\n"
        "# p1-p99, fluid p60-p99.8), as CT members stack HU windows. 6 slices per slot instead of 18 (same 24 images): the\n"
        "# 2.5D neighbour channels measured ~0 (convnext_small_25d -0.0005), so the channels go to contrast instead.",
        {"input": '  channel_mode: "multiwindow"\n  multiwindow_pct: [[1.0, 80.0], [1.0, 99.0], [60.0, 99.8]]\n'}),
    "convnext_base_25d_v5_slab_mmm": (
        "SLAB MIN/MEAN/MAX CHANNELS: each image covers one bin of the stack (6 bins per slot) and holds [MinIP, mean,\n"
        "# MIP] over EVERY stored slice in it. Today 35-43% of stored slices are never seen; MinIP keeps the dark ACL /\n"
        "# menisci continuous, MIP keeps fluid / edema bright. Same 24 images per study.",
        {"input": '  channel_mode: "slab_mmm"\n'}),
    "convnext_base_25d_v5_randconv": (
        "RANDCONV AUGMENTATION: with p 0.25 per slot the slot passes a random 1/3/5 px filter, rescaled to its own\n"
        "# mean/std, mixed in at alpha 0.1-0.5 (Xu 2021): scanner texture scrambled, anatomy kept. Input unchanged.",
        {"augmentation": "  randconv: {p: 0.25, kernels: [1, 3, 5], alpha: [0.1, 0.5]}\n"}),
}

for name, (why, add) in RUNS.items():
    d = R / name
    d.mkdir(exist_ok=True)
    for f in ("config.yaml", "train.py", "evaluate.py", "plot_results.py", "submit.py"):
        shutil.copy(R / DONOR / f, d / f)
    for f in ("train.py", "evaluate.py", "plot_results.py", "submit.py"):
        p = d / f
        s = p.read_text(encoding="utf-8").replace(DONOR, name).replace("cb25d18_v5_aug_base", name.replace(
            "convnext_base_25d_v5_", "cb25d_v5_")[:50])
        p.write_text(s, encoding="utf-8")
    p = d / "config.yaml"
    s = p.read_text(encoding="utf-8")
    hdr = (f"# {name}\n#\n# {why}\n# Copy of {DONOR} (val 0.9174 / audit 0.9209 / LB 0.930): ONE change, judged against\n"
           f"# it on the same v5 val + audit-58. LOCAL RTX 4060 (user, 8 Oct: free), recipe of v4_flip_f0.\n#\n"
           "# ---- the donor's notes follow ----\n")
    s = hdr + s.split("\n", 1)[1]
    s = s.replace(f'name: "{DONOR}"', f'name: "{name}"')
    lines = s.split("\n")
    out = []
    for ln in lines:
        key = next((k for k in LOCAL if (ln + " ").startswith(k)), None)
        out.append(LOCAL[key].rstrip("\n") if key else ln)
    s = "\n".join(out)
    for k, v in LOCAL.items():
        assert v.split("#")[0].strip() in s, (name, k)
    s = s.replace("training:\n", "training:\n  grad_checkpointing: true          # local 8 GB GPU: recompute activations (exact)\n", 1)
    for sec, txt in add.items():
        if re.search(rf"^{sec}:\n", s, re.M):
            s = re.sub(rf"^{sec}:\n", f"{sec}:\n{txt}", s, count=1, flags=re.M)
        else:
            s = s.rstrip("\n") + f"\n\n{sec}:\n{txt}"
    s = s.rstrip("\n") + "\n" + OUTPUT
    p.write_text(s, encoding="utf-8")
    # checkpoints on C: (D: is nearly full), as for the round-2 local folds
    res = d / "results"
    res.mkdir(exist_ok=True)
    ck = res / "checkpoints"
    tgt = CK / name
    tgt.mkdir(parents=True, exist_ok=True)
    if not ck.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(ck), str(tgt)], check=True, capture_output=True)
    print("made", name, "->", tgt)
