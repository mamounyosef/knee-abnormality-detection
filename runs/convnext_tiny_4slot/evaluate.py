"""
evaluate.py — score a saved checkpoint, outside any training run.

Calibrates the per-task F1 thresholds automatically if results/thresholds.json
is missing, then reuses that file on later runs (--recalibrate forces a refresh).
AUROC never depends on a threshold; F1/precision/recall do.

    python runs/convnext_tiny_4slot/evaluate.py
    python runs/convnext_tiny_4slot/evaluate.py --checkpoint top1 --sets val audit
    python runs/convnext_tiny_4slot/evaluate.py --sets ../../data/05_val.csv
    python runs/convnext_tiny_4slot/evaluate.py --recalibrate

Writes into results/: <set>_predictions.csv, <set>_results.json, <set>_results.txt
"""

import argparse
import sys
from pathlib import Path

import torch

EXP_NAME = "convnext_tiny_4slot"


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(EXP_DIR))
import shared_code as sc  # noqa: E402
from train import build_model  # noqa: E402  same architecture the run was trained with


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best",
                    help="best | last | top1..topK | <step> | path to a .pt")
    ap.add_argument("--sets", nargs="*", default=["val", "audit"],
                    help="'val', 'audit', or a path to any split CSV")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refresh thresholds.json even if it already exists")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    args = ap.parse_args()

    cfg = sc.load_config(EXP_DIR, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)

    if args.recalibrate:
        thr_path = (EXP_DIR / cfg["output"]["run_dir"]
                    / cfg["output"].get("thresholds_json", "thresholds.json"))
        if thr_path.exists():
            thr_path.unlink()
            print(f"  🔄 removed {thr_path.name}, recalibrating")

    sc.evaluate_official(cfg, model, EXP_DIR, device=device,
                         checkpoint=args.checkpoint, eval_sets=tuple(args.sets),
                         calibrate_if_missing=True, batch_size=args.batch_size,
                         num_workers=args.num_workers, amp=args.amp)


if __name__ == "__main__":
    main()
