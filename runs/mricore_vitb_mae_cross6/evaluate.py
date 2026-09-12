"""
evaluate.py — score a saved checkpoint, outside any training run.

Calibrates the per-task F1 thresholds automatically if results/thresholds.json
is missing, then reuses that file on later runs (--recalibrate forces a refresh).
AUROC never depends on a threshold; F1/precision/recall do.

    python runs/mricore_vitb_mae_cross6/evaluate.py
    python runs/mricore_vitb_mae_cross6/evaluate.py --checkpoint top1 --sets val audit
    python runs/mricore_vitb_mae_cross6/evaluate.py --sets ../../data/05_val.csv
    python runs/mricore_vitb_mae_cross6/evaluate.py --recalibrate

Writes into results/: <set>_predictions.csv, <set>_results.json, <set>_results.txt
"""

import argparse
import sys
from pathlib import Path

import torch

EXP_NAME = "mricore_vitb_mae_cross6"


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


def _run(cfg, exp_dir, args):
    """Shared by the local and the Modal path."""
    import torch as _t
    device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    if args.recalibrate:
        thr = (Path(exp_dir) / cfg["output"]["run_dir"]
               / cfg["output"].get("thresholds_json", "thresholds.json"))
        if thr.exists():
            thr.unlink()
            print(f"  🔄 removed {thr.name}, recalibrating")
    # --checkpoint takes a comma-separated list: one container, one data load, several
    # checkpoints. Scoring top1..top5 as five invocations would pay container startup
    # and shard warm-up five times over.
    model = build_model(cfg)
    out = {}
    cks = [c for c in str(args.checkpoint).split(",") if c]
    for i, ck in enumerate(cks, 1):
        if len(cks) > 1:
            print("#" * 70)
            print(f"# checkpoint {i}/{len(cks)}: {ck}")
        out[ck] = sc.evaluate_official(cfg, model, exp_dir, device=device,
                                       checkpoint=ck, eval_sets=tuple(args.sets),
                                       calibrate_if_missing=True, batch_size=args.batch_size,
                                       num_workers=args.num_workers, amp=args.amp)
    return out if len(cks) > 1 else out[cks[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best",
                    help="best | last | top1..topK | <step> | path, or a comma-separated list")
    ap.add_argument("--sets", nargs="*", default=["val", "audit"],
                    help="'val', 'audit', or a path to any split CSV")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refresh thresholds.json even if it already exists")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    args = ap.parse_args()
    _run(sc.load_config(EXP_DIR, verbose=False), EXP_DIR, args)


class _Args:
    """Plain holder so the remote side reuses _run without argparse."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


try:
    # skipped when modal isn't installed, so the local path still works
    import modal

    cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App(f"{EXP_NAME}-eval")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(cfg))
    def eval_remote(checkpoint: str, sets: list, recalibrate: bool,
                    batch_size, num_workers, amp):
        """Scores on the GPU, beside the shards volume — the reports come back via sync."""
        rcfg = sc.remote_cfg(cfg)
        out_dir = Path(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        try:
            _run(rcfg, out_dir, _Args(checkpoint=checkpoint, sets=sets,
                                      recalibrate=recalibrate, batch_size=batch_size,
                                      num_workers=num_workers, amp=amp))
        finally:
            _runs.commit()

    @app.local_entrypoint()          # `modal run evaluate.py`
    def modal_main(checkpoint: str = "best", sets: str = "val,audit",
                   recalibrate: bool = False, batch_size: int = 0,
                   num_workers: int = -1, no_amp: bool = False):
        try:
            eval_remote.remote(checkpoint, [s for s in sets.split(",") if s], recalibrate,
                               batch_size or None, None if num_workers < 0 else num_workers,
                               False if no_amp else None)
        finally:
            sc.sync_back(cfg, "evaluated")

except ImportError:
    pass


if __name__ == "__main__":
    main()
