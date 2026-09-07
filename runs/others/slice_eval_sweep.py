"""
slice_eval_sweep.py — how many slices should we feed AT INFERENCE, and does jittering
which ones help?

Every model here was trained on `n_slices: 24` picked by splitting each series into 24
equal bins and taking one slice from each. A series has a median of 30 real slices, so
24 leaves ~6 slices of headroom, and two questions follow:

  1. Feed MORE at eval than at train. 24 -> 28 -> 32 uses more of the stack, but
     pick_slices repeats slices once a series is shorter than n_want, and the repeat
     branch uses a DIFFERENT geometry (linspace over the whole stack, margin ignored)
     than the binned path. So raising n_slices is not free - it silently changes the
     sampling shape for the short series:

        n=24  17.4% of series repeat      n=30  42.0%
        n=26  26.1%                       n=32  62.3%

     At 32, most of the data is being scored under a rule the model never trained on.
     That is the thing to measure, not assume.

  2. TEST-TIME AUGMENTATION over the slice pick. `slice_sampling: random` takes one
     random slice per bin instead of the centre - same count, same order, same even
     coverage, just a different representative of each bin. Averaging k such passes is
     the only TTA this pipeline can do for free, since the model already saw exactly
     this variation during training. Note the ceiling: 17.4% of series are shorter than
     24 and get no jitter at all, so the gain is bounded by the other 82.6%.

Not to be confused with the duplicate-series TTA already tried and rejected (-0.0025):
that swapped WHICH series fills a slot, and the discarded series is genuinely worse.
This resamples within the series we already chose.

    python -m modal run runs/others/slice_eval_sweep.py --run convnext_base_relabel

Writes results/slice_sweep/<variant>_val.npz (raw logits, so a winning variant can go
straight into ensemble.py) plus a ranked table on stdout. Touches nothing else.
"""

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402

# (label, n_slices, sampling, n_passes). The first is the CONTROL: it reproduces the
# run's own recorded val score, and if it does not, nothing below it means anything.
VARIANTS = [
    ("n24-even", 24, "even", 1),
    ("n26-even", 26, "even", 1),
    ("n28-even", 28, "even", 1),
    ("n32-even", 32, "even", 1),
    ("n24-tta3", 24, "random", 3),
    ("n24-tta5", 24, "random", 5),
    ("n28-tta3", 28, "random", 3),
]


def _load_build_model(exp_dir: Path):
    """The run's own train.py, so the architecture matches what was trained."""
    sys.path.insert(0, str(exp_dir))
    spec = importlib.util.spec_from_file_location(f"{exp_dir.name}_train",
                                                  exp_dir / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model


def macro_auroc(y_true, scores, tasks):
    """Macro AUROC, targets binarized at 0.5 - val labels are LLM probabilities."""
    from sklearn.metrics import roc_auc_score
    per = {}
    for i, t in enumerate(tasks):
        col = y_true[:, i]
        m = ~np.isnan(col)
        yb = (col[m] > 0.5).astype(int)
        if len(np.unique(yb)) > 1:
            per[t] = float(roc_auc_score(yb, scores[m, i]))
    return float(np.mean(list(per.values()))), per


def run_sweep(cfg, exp_dir: Path, checkpoint: str, code_dir: Path | None = None):
    """exp_dir holds the RESULTS, code_dir holds train.py. On Modal these differ:
    the code is baked into the image at /root/runs, the checkpoints ride the volume."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_model = _load_build_model(Path(code_dir or exp_dir))
    ckpt_dir = Path(exp_dir) / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"]

    tasks = cfg["tasks"]
    df = sc._eval_df(cfg, cfg["paths"]["val_csv"])
    # the split CSVs name the id column StudyInstanceUID; the prediction npz files
    # ensemble.py reads call it study_ids. Resolve it rather than assuming either.
    id_col = next((c for c in ("StudyInstanceUID", "study", "study_id") if c in df.columns),
                  None)
    if id_col is None:
        raise SystemExit(f"no study id column in {cfg['paths']['val_csv']}: "
                         f"{list(df.columns)[:8]}")
    series = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    out_dir = Path(exp_dir) / cfg["output"]["run_dir"] / "slice_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, n_sl, sampling, passes in VARIANTS:
        vcfg = copy.deepcopy(cfg)
        vcfg["input"]["n_slices"] = n_sl
        vcfg["input"]["slice_sampling"] = sampling
        # n_slices feeds plan_slots' coverage ranking, so the slot plan is rebuilt per
        # variant rather than reused - otherwise a variant scores a different series set
        plan = sc.plan_slots(series, vcfg)

        # per_study = n_slots * n_slices is BAKED INTO THE MODEL at build time (the head
        # reshapes the flat image batch back to one row per study with it), so a variant
        # that changes n_slices needs its own model object. The weights are identical -
        # mean pooling over the slice axis has no parameters that depend on how many
        # slices there are - so the same checkpoint loads into every one of them.
        model = build_model(vcfg)
        ck, ck_path = sc._load_ckpt_weights(model, ckpt_dir, checkpoint, device)
        if label == VARIANTS[0][0]:
            print("\n  checkpoint %s  (step %s, score %s)\n"
                  % (ck_path, ck.get("step"), ck.get("score")))

        acc, y_true = None, None
        for p in range(passes):
            # the eval rng keys off reproducibility.seed (and the study id), so bumping
            # the seed is what makes pass 2 pick different slices from pass 1
            vcfg["reproducibility"]["seed"] = int(cfg["reproducibility"]["seed"]) + p
            collect = {}
            y_true, _prob, _ = sc._predict_dataframe(
                vcfg, model, df, device, plan=plan, collect=collect,
                desc="%s pass %d/%d" % (label, p + 1, passes))
            lg = np.concatenate(collect["_lg"])          # (n_studies, n_tasks) logits
            acc = lg if acc is None else acc + lg
        scores = acc / passes                            # averaged in LOGIT space
        au, per = macro_auroc(y_true, scores, tasks)
        rows.append((label, n_sl, sampling, passes, au, per))
        print("\n  >>> %-12s macro AUROC %.4f\n" % (label, au))

        np.savez_compressed(
            out_dir / ("%s_val.npz" % label),
            logits=scores.astype(np.float32), targets=y_true.astype(np.float32),
            tasks=np.array(tasks), study_ids=np.array(df[id_col].astype(str)))

    base = rows[0][4]
    print("=" * 88)
    print("  SLICE SWEEP - %s, checkpoint %s" % (Path(exp_dir).name, checkpoint))
    print("=" * 88)
    print("%-14s %7s %9s %8s %10s %12s"
          % ("variant", "slices", "sampling", "passes", "AUROC", "vs control"))
    print("-" * 88)
    for label, n_sl, sampling, passes, au, _ in rows:
        print("%-14s %7d %9s %8d %10.4f %+12.4f"
              % (label, n_sl, sampling, passes, au, au - base))
    print("=" * 88)
    print("  The control must reproduce the run's own recorded val score. Anything")
    print("  below +0.001 against it is inside the noise of a 704-study split.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_base_relabel")
    ap.add_argument("--checkpoint", default="best")
    a = ap.parse_args()
    exp_dir = PKG_ROOT / a.run
    run_sweep(sc.load_config(exp_dir, verbose=False), exp_dir, a.checkpoint)


try:
    import modal

    app = modal.App("slice-eval-sweep")
    _cfg0 = sc.load_config(PKG_ROOT / "convnext_base_relabel", verbose=False)
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(_cfg0["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(_cfg0["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={_cfg0["modal"]["shards_mount"]: _shards,
                           _cfg0["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(_cfg0))
    def sweep_remote(run: str, checkpoint: str):
        exp_dir = PKG_ROOT / run
        cfg = sc.remote_cfg(sc.load_config(exp_dir, verbose=False))
        out_dir = Path(cfg["modal"]["runs_mount"]) / run
        try:
            run_sweep(cfg, out_dir, checkpoint, code_dir=exp_dir)
        finally:
            _runs.commit()

    @app.local_entrypoint()
    def modal_main(run: str = "convnext_base_relabel", checkpoint: str = "best"):
        try:
            sweep_remote.remote(run, checkpoint)
        finally:
            sc.sync_back(sc.load_config(PKG_ROOT / run, verbose=False), "slice-swept")

except ImportError:
    pass


if __name__ == "__main__":
    main()
