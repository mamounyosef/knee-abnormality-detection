"""local_recipe.py - turn a copied Modal run's config.yaml into its LOCAL (RTX 4060) version.

Only the machine settings change: batch 2 x 4 accumulation (same 8 studies per update),
gradient checkpointing (exact), 2 loader workers, no pinned memory. Every replacement must hit
exactly once, or it stops. Optional --stop N adds training.stop_after_steps.

    python data_code/local_recipe.py <run> [--stop 2400] [--header "one line of purpose"]
"""
import argparse
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stop", type=int, default=0)
ap.add_argument("--header", default="")
a = ap.parse_args()
p = Path(__file__).resolve().parent.parent / "runs" / a.run / "config.yaml"
s = p.read_text(encoding="utf-8")


def sub(pat, rep):
    global s
    s, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, f"{pat!r} matched {n} times"


if a.header:
    s = f"# {a.run}\n#\n# {a.header}\n#\n# ---- the donor's notes follow ----\n" + s
extra = f"  stop_after_steps: {a.stop}            # short check run; lr schedule stays the full run's\n" if a.stop else ""
sub(r"^training:\n", "training:\n  grad_checkpointing: true          # local 8 GB GPU: recompute activations (exact)\n" + extra)
sub(r"^  batch_size: 8 .*$", "  batch_size: 2                     # LOCAL: 2 x accum_steps 4 = the same 8 studies per update")
sub(r"^  accum_steps: 1 .*$", "  accum_steps: 4                    # effective 8 studies per update")
sub(r"^  val_batch_size: 16 .*$", "  val_batch_size: 4                 # local")
sub(r"^  num_workers: 26 .*$", "  num_workers: 2                    # local PC: 16 GB RAM, keep it usable")
sub(r"^  val_num_workers: null .*$", "  val_num_workers: 2")
sub(r"^  prefetch_factor: 6 .*$", "  prefetch_factor: 2                # batches queued per worker")
sub(r"^  pin_memory: true .*$", "  pin_memory: false                 # local: no page-locked memory")
p.write_text(s, encoding="utf-8")
print("ok", p)
