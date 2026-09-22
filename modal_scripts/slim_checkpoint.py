"""slim_checkpoint.py - strip a checkpoint down to the weights a warm start needs.

A training checkpoint carries the optimizer state, which for AdamW is two moment tensors per
parameter: for MedImageInsight's 360.7M parameters that is 4.3 GB on disk against 1.4 GB of
actual weights. model.init_from reads the weights ONLY (the optimizer and scheduler stay
fresh by design), so shipping the full file to another Modal workspace uploads three times
what is used.

    python modal_scripts/slim_checkpoint.py medimageinsight_davit_huge_336 best

writes best_weights.pt beside it, keeping the provenance fields so the warm start can still
print which checkpoint it came from.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("epoch", "step", "global_step", "score", "cfg")


def main(run: str, name: str = "best") -> int:
    ck_dir = ROOT / "runs" / run / "results" / "checkpoints"
    src = ck_dir / ("%s.pt" % name)
    if not src.exists():
        raise SystemExit("no such checkpoint: %s" % src)
    dst = ck_dir / ("%s_weights.pt" % name)

    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit("%s has no 'model' key, so it is not a training checkpoint" % src)
    slim = {"model": ck["model"]}
    for k in KEEP:
        if k in ck:
            slim[k] = ck[k]
    dropped = sorted(set(ck) - set(slim))
    torch.save(slim, dst)

    a, b = src.stat().st_size / 1e9, dst.stat().st_size / 1e9
    print("%s" % src.relative_to(ROOT))
    print("  kept    : model (%d tensors, %.1fM params) + %s"
          % (len(ck["model"]), sum(v.numel() for v in ck["model"].values()) / 1e6,
             ", ".join(k for k in KEEP if k in slim)))
    print("  dropped : %s" % (", ".join(dropped) or "nothing"))
    print("  %.2f GB -> %.2f GB  (%.1fx smaller)" % (a, b, a / max(b, 1e-9)))
    print("  wrote %s" % dst.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "best"))
