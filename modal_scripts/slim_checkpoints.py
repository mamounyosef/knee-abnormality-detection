"""slim_checkpoints.py - free disk space in runs/ without losing any model.

1. deletes every runs/**/kaggle/ build folder (submit.py and ensemble.py rebuild them from the
   run checkpoints; checked bit-identical first, 26 Sep)
2. deletes step*.pt / last.pt / top_step*.pt: only best.pt (and avg_*.pt) are ever evaluated,
   ensembled or submitted
3. strips best.pt / avg_*.pt down to what inference needs (model, cfg, epoch, step, score,
   track, note): the optimizer / scaler / rng state only serves RESUMING a run, and every run
   here is finished. Each stripped file is re-loaded and compared tensor by tensor before it
   replaces the original.
4. deletes a weights-only duplicate (best_weights.pt) once it equals the stripped best.pt

    python modal_scripts/slim_checkpoints.py            # dry run: prints what would happen
    python modal_scripts/slim_checkpoints.py --apply
"""
import os
import shutil
import sys
from pathlib import Path

import torch

RUNS = Path(__file__).resolve().parent.parent / "runs"
APPLY = "--apply" in sys.argv
KEEP = ("model", "cfg", "epoch", "step", "score", "track", "note")
freed = 0


def gb(n):
    return f"{n / 1e9:.2f} GB"


def size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) if Path(p).is_dir() else Path(p).stat().st_size


for d in sorted(p for p in RUNS.rglob("kaggle") if p.is_dir()):
    s = size(d)
    freed += s
    print(f"[kaggle dir] delete {d.relative_to(RUNS)} ({gb(s)})")
    if APPLY:
        shutil.rmtree(d)

for p in sorted(RUNS.rglob("*.pt")):
    if "kaggle" in p.parts:
        continue
    if p.name.startswith(("step", "last", "top_step")):
        freed += p.stat().st_size
        print(f"[extra ckpt] delete {p.relative_to(RUNS)} ({gb(p.stat().st_size)})")
        if APPLY:
            p.unlink()

for p in sorted(RUNS.rglob("*.pt")):
    if "kaggle" in p.parts or p.name.startswith(("step", "last", "top_step")) or p.name == "best_weights.pt":
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        print(f"[skip] {p.relative_to(RUNS)}: no 'model' key")
        continue
    extra = [k for k in ck if k not in KEEP]
    if not extra:
        continue
    before = p.stat().st_size
    slim = {k: ck[k] for k in KEEP if k in ck}
    tmp = p.with_suffix(".slim.tmp")
    torch.save(slim, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)
    assert back["model"].keys() == ck["model"].keys()
    assert all(torch.equal(back["model"][k], ck["model"][k]) for k in ck["model"]), p
    assert all(back.get(k) == ck.get(k) for k in ("epoch", "step", "score")), p
    after = tmp.stat().st_size
    freed += before - after
    print(f"[strip] {p.relative_to(RUNS)}: dropped {extra}, {gb(before)} -> {gb(after)}, weights identical")
    if APPLY:
        os.replace(tmp, p)
    else:
        tmp.unlink()

for p in sorted(RUNS.rglob("best_weights.pt")):
    b = p.with_name("best.pt")
    if not b.exists():
        continue
    m1 = torch.load(p, map_location="cpu", weights_only=False)["model"]
    m2 = torch.load(b, map_location="cpu", weights_only=False)["model"]
    if m1.keys() == m2.keys() and all(torch.equal(m1[k], m2[k]) for k in m1):
        freed += p.stat().st_size
        print(f"[duplicate] delete {p.relative_to(RUNS)} (equals best.pt)")
        if APPLY:
            p.unlink()
    else:
        print(f"[keep] {p.relative_to(RUNS)} differs from best.pt")

print(f"\n{'FREED' if APPLY else 'WOULD FREE'} {gb(freed)}")
