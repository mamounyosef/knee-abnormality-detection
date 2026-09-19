r"""laterality_flip_eval.py - does mirroring one knee side change what the model scores?

THE QUESTION. "Medial" means the inner side of the leg. On a scan that is the LEFT half of
the image for a right knee and the RIGHT half for a left knee, so the four side-specific
findings (Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA) sit on opposite halves
of the frame depending on which leg was scanned. Every run we have trained on the mixed
set, so each of those four findings had to be learned twice, once per orientation. The
public 0.941 notebook normalises this away. Is that costing us?

WHAT THIS MEASURES, EXACTLY. One checkpoint, three validation passes over the same 704
studies, differing ONLY in the mirroring:

    none    every study as stored                       (the baseline, what we ship)
    right   right knees mirrored -> all face like a left knee
    left    left knees mirrored  -> all face like a right knee

Nothing is retrained. The weights are identical in all three passes.

HOW TO READ THE RESULT. The model was trained on un-normalised data, so a flip is an input
it has never seen and some drop is expected everywhere. What matters is WHERE the drop
lands, not that there is one:

    big drop concentrated on the four side-specific findings
        -> the model IS using orientation to tell medial from lateral. It already solved
           the problem on its own. Normalising is redundant, and a retrain buys nothing.

    drop spread evenly across all twelve
        -> generic distribution shift from an unfamiliar input, no laterality signal.

    no drop anywhere, side-specific findings included
        -> the model is orientation-BLIND. It cannot be distinguishing medial from lateral
           by position at all, which is the one result that argues a normalised retrain
           could actually add something.

This cannot prove normalising helps (only a retrain does that). It can show whether the
model currently depends on orientation, which is the premise the whole idea rests on.

    modal run runs/others/laterality_flip_eval.py
    modal run runs/others/laterality_flip_eval.py --run dinov3_convnext_base --checkpoint best
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_NAME = "dinov3_convnext_base"
CHECKPOINT = "best"
SPLIT = "val"
MODES = ("none", "right", "left")

# the four whose MEANING flips with the knee. Everything else names a structure that is
# where it is regardless of which leg it is, so orientation should not matter to it.
SIDE_SPECIFIC = ("Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA")


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def _build_model(cfg, run_name):
    """Import the run's OWN train.py from the IMAGE, never from the runs volume.

    The volume holds results/, not code. Importing `train` off it is the
    ModuleNotFoundError this project has hit before.
    """
    import copy
    import importlib.util
    code = PKG_ROOT / run_name / "train.py"
    spec = importlib.util.spec_from_file_location("_lat_train_%s" % run_name, code)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(PKG_ROOT / run_name))
    spec.loader.exec_module(mod)
    # the checkpoint overwrites every weight a moment later, so downloading the pretrained
    # trunk first would be a few hundred MB of pure wait
    c = copy.deepcopy(cfg)
    c["model"]["pretrained"] = False
    return mod.build_model(c)


def run(cfg, run_dir, run_name=RUN_NAME, checkpoint=CHECKPOINT, split=SPLIT):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = list(cfg["tasks"])

    lat = pd.read_csv(sc.resolve(cfg["paths"]["laterality_csv"], cfg))
    sides = dict(zip(lat["StudyInstanceUID"], lat["side"]))

    df = sc._eval_df(cfg, cfg["paths"]["%s_csv" % split])
    n_l = sum(1 for s in df["StudyInstanceUID"] if sides.get(s) == "left")
    n_r = sum(1 for s in df["StudyInstanceUID"] if sides.get(s) == "right")
    print("  %s: %d studies  (%d left, %d right, %d unknown)"
          % (split, len(df), n_l, n_r, len(df) - n_l - n_r), flush=True)

    model = _build_model(cfg, run_name)
    ckpt_dir = Path(run_dir) / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"]
    ck, ck_path = sc._load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    print("  checkpoint %s  (epoch %s, step %s, score %s)"
          % (ck_path.name, ck.get("epoch"), ck.get("step"), ck.get("score")), flush=True)

    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    out = {}
    for mode in MODES:
        # the ONLY thing that differs between passes. Same weights, same studies, same
        # slice picks (eval sampling is deterministic and keyed off the study id).
        cfg["input"]["laterality_flip"] = None if mode == "none" else mode
        print("=" * 70, flush=True)
        print("  pass '%s': laterality_flip=%r" % (mode, cfg["input"]["laterality_flip"]),
              flush=True)
        y_true, y_prob, _ = sc._predict_dataframe(cfg, model, df, device, None, plan=plan,
                                                  desc="flip=%s" % mode)
        m = sc.compute_metrics(y_true, y_prob, tasks)
        out[mode] = {"macro": float(m["macro_auroc"]),
                     "per": {t: float(m["per_task"][t]["auroc"]) for t in tasks}}
        print("  -> macro AUROC %.4f" % out[mode]["macro"], flush=True)
    cfg["input"]["laterality_flip"] = None

    _report(out, tasks)
    return out


def _report(out, tasks):
    base = out["none"]
    print("=" * 78, flush=True)
    print("  LATERALITY FLIP: per-finding AUROC, same weights, three input orientations",
          flush=True)
    print("=" * 78, flush=True)
    print("  %-19s %8s %8s %8s   %8s %8s" %
          ("finding", "none", "flip-R", "flip-L", "dR", "dL"), flush=True)
    for t in tasks:
        b = base["per"][t]
        r, l = out["right"]["per"][t], out["left"]["per"][t]
        tag = "  <- side-specific" if t in SIDE_SPECIFIC else ""
        print("  %-19s %8.4f %8.4f %8.4f   %+8.4f %+8.4f%s"
              % (t, b, r, l, r - b, l - b, tag), flush=True)
    print("  %-19s %8.4f %8.4f %8.4f   %+8.4f %+8.4f"
          % ("MACRO", base["macro"], out["right"]["macro"], out["left"]["macro"],
             out["right"]["macro"] - base["macro"], out["left"]["macro"] - base["macro"]),
          flush=True)

    # the discriminating number: does the damage land on the four findings whose MEANING
    # depends on the side, or is it spread evenly over all twelve?
    print("-" * 78, flush=True)
    for name, group in (("side-specific (4)", [t for t in tasks if t in SIDE_SPECIFIC]),
                        ("everything else (8)", [t for t in tasks if t not in SIDE_SPECIFIC])):
        dr = float(np.mean([out["right"]["per"][t] - base["per"][t] for t in group]))
        dl = float(np.mean([out["left"]["per"][t] - base["per"][t] for t in group]))
        print("  mean delta, %-20s  flip-R %+.4f   flip-L %+.4f" % (name, dr, dl), flush=True)
    print("=" * 78, flush=True)


try:
    import modal

    _cfg = sc.load_config(PKG_ROOT / RUN_NAME, verbose=False)
    app = modal.App("laterality-flip-eval")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(_cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(_cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={_cfg["modal"]["shards_mount"]: _shards,
                           _cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(_cfg))
    def flip_remote(run_name: str, checkpoint: str, split: str):
        cfg = sc.remote_cfg(sc.load_config(PKG_ROOT / run_name, verbose=False))
        run_dir = Path(cfg["modal"]["runs_mount"]) / run_name
        res = run(cfg, run_dir, run_name, checkpoint, split)
        out = Path(cfg["modal"]["runs_mount"]) / "others" / "laterality_flip_eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"run": run_name, "checkpoint": checkpoint,
                                   "split": split, "results": res}, indent=2),
                       encoding="utf-8")
        _runs.commit()
        print("  wrote %s" % out, flush=True)
        return res

    @app.local_entrypoint()
    def modal_main(run: str = RUN_NAME, checkpoint: str = CHECKPOINT, split: str = SPLIT):
        flip_remote.remote(run, checkpoint, split)

except ImportError:
    pass


if __name__ == "__main__":
    _c = sc.load_config(PKG_ROOT / RUN_NAME, verbose=False)
    run(_c, PKG_ROOT / RUN_NAME)
