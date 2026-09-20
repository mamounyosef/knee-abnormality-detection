r"""sliding_window_tta.py - many sampling grids per study, then pool them PER FINDING.

WHAT THIS IS, AND HOW IT DIFFERS FROM THE TTA WE ALREADY REJECTED. On 7 Sep a slice-TTA
sweep scored one checkpoint under three random slice picks and AVERAGED them: `n24-tta3`
came out at -0.0004, a null result, and slice TTA was closed. This is a different operation
with the same name.

  * The grids SLIDE rather than being random. input.slice_offset shifts the even sampling
    grid by a fraction of one bin, so offsets 0, 1/3, 2/3 sample three interleaved grids
    that between them cover the stack far more densely than any single grid does. Random
    picks give no such guarantee - two draws can land on nearly the same slices.
  * The passes are POOLED PER FINDING, not averaged. That is the whole point. Averaging is
    exactly what destroys a focal finding: if a fracture is visible in one grid and absent
    from two, the mean dilutes it by three while a max keeps it. The 0.941 notebook pools
    per class over its sliding windows - max for Fracture, Contusion and the menisci, top2
    for ACL and MCL, mean for Synovitis - and that per-class pooling is the mechanism we
    have never replicated at inference.

WHY THIS IS NOT THE SAME AS THE POOLING RUNS THAT FAILED. convnext_base_25d_perfinding_pool_features
applied max INSIDE training, over 24 non-overlapping windows, and lost 0 of 4 columns. Here
the model is untouched and still pools its own windows with a mean; the pooling is post hoc,
across whole predictions, on a model trained the ordinary way. Different experiment, and it
risks nothing - a bad pooling is simply not used.

ALSO TESTS REVERSE-SLICE TTA. input.slice_reverse returns each series' picks in the opposite
order. For a sagittal series that IS the laterality mirror, since the slice axis is the
left-right axis. The 0.941 notebook scores one of its checkpoints twice this way.

The GPU job runs each pass once and saves every pass's raw logits, so all the pooling
questions are answered afterwards on this PC for free.

    modal run runs/others/sliding_window_tta.py
    modal run runs/others/sliding_window_tta.py --run convnext_base_25d_18slices --offsets 5
    python runs/others/sliding_window_tta.py --pool-only     # re-pool the saved logits
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUN_NAME = "convnext_base_25d_18slices"
CHECKPOINT = "best"
SPLIT = "val"

TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
         'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# the 0.941 notebook's per-class choice over its sliding windows
NOTEBOOK_POOL = {
    "max":  ["Fracture", "Contusion", "Medial Meniscus", "Lateral Meniscus"],
    "top2": ["ACL", "MCL"],
    "mean": ["Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"],
}


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def _build_model(cfg, run_name):
    import copy
    import importlib.util
    code = PKG_ROOT / run_name / "train.py"
    spec = importlib.util.spec_from_file_location("_tta_%s" % run_name, code)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(PKG_ROOT / run_name))
    spec.loader.exec_module(mod)
    c = copy.deepcopy(cfg)
    c["model"]["pretrained"] = False        # the checkpoint overwrites every weight anyway
    return mod.build_model(c)


def passes_for(n_offsets: int, reverse: bool):
    """The sampling grids to score. Pass 0 is always the plain baseline."""
    out = [(0.0, False)]
    for k in range(1, n_offsets):
        out.append((k / float(n_offsets), False))
    if reverse:
        out += [(o, True) for o, _ in list(out)]
    return out


def run_passes(cfg, run_dir, run_name, checkpoint, split, n_offsets, reverse):
    import pandas as pd
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = sc._eval_df(cfg, cfg["paths"]["%s_csv" % split])
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    model = _build_model(cfg, run_name)
    ckpt_dir = Path(run_dir) / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"]
    ck, ck_path = sc._load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    print("  %s / %s: epoch %s step %s score %s"
          % (run_name, ck_path.name, ck.get("epoch"), ck.get("step"), ck.get("score")),
          flush=True)

    todo = passes_for(n_offsets, reverse)
    print("  %d passes over %d studies: %s"
          % (len(todo), len(df), ", ".join("o=%.2f%s" % (o, " rev" if r else "")
                                           for o, r in todo)), flush=True)

    logits, labels, y = [], [], None
    for o, rev in todo:
        cfg["input"]["slice_offset"] = o
        cfg["input"]["slice_reverse"] = rev
        tag = "offset=%.3f%s" % (o, " reverse" if rev else "")
        yt, yp, _ = sc._predict_dataframe(cfg, model, df, device, None, plan=plan, desc=tag)
        # back to logits: _predict_dataframe returns sigmoid probabilities
        lg = np.log(np.clip(yp, 1e-7, 1 - 1e-7) / (1 - np.clip(yp, 1e-7, 1 - 1e-7)))
        logits.append(lg)
        labels.append(tag)
        y = yt
        print("    %-22s macro %.4f" % (tag, sc.compute_metrics(yt, yp, cfg["tasks"])
                                        ["macro_auroc"]), flush=True)
    cfg["input"]["slice_offset"], cfg["input"]["slice_reverse"] = 0.0, False
    return np.stack(logits), np.asarray(y), labels


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    out = []
    for j in range(y.shape[1]):
        t = (y[:, j] > 0.5).astype(int)
        if t.min() != t.max():
            out.append(roc_auc_score(t, p[:, j]))
    return float(np.mean(out)) if out else float("nan")


def reduce_stack(L, kind):
    """L is (passes, studies, tasks) -> (studies, tasks)."""
    if kind == "mean":
        return L.mean(axis=0)
    if kind == "max":
        return L.max(axis=0)
    if kind == "top2":
        k = min(2, L.shape[0])
        return np.sort(L, axis=0)[-k:].mean(axis=0)
    if kind == "softmax10":
        # the notebook's beta=10 soft window pool: a smooth max
        w = np.exp(10.0 * (L - L.max(axis=0, keepdims=True)))
        return (w * L).sum(axis=0) / w.sum(axis=0)
    raise ValueError(kind)


def report(L, y, labels, out_json=None):
    print("\n" + "=" * 80)
    print("  %d passes, %d studies. Pass 0 is the plain baseline." % (L.shape[0], L.shape[1]))
    print("=" * 80)
    base = auroc(y, L[0])
    print("\n  baseline (single grid, no TTA): %.4f" % base)

    print("\n  UNIFORM POOLING, all findings the same way")
    best_uniform = ("", base)
    for kind in ("mean", "max", "top2", "softmax10"):
        v = auroc(y, reduce_stack(L, kind))
        print("    %-10s %.4f  (%+.4f)%s" % (kind, v, v - base,
                                             "  <- GAIN" if v > base + 0.0005 else ""))
        if v > best_uniform[1]:
            best_uniform = (kind, v)

    print("\n  PER-FINDING POOLING, the 0.941 notebook's assignment")
    p = np.empty_like(L[0])
    for kind, names in NOTEBOOK_POOL.items():
        red = reduce_stack(L, kind)
        for t in names:
            p[:, TASKS.index(t)] = red[:, TASKS.index(t)]
    v_nb = auroc(y, p)
    print("    notebook   %.4f  (%+.4f)%s" % (v_nb, v_nb - base,
                                              "  <- GAIN" if v_nb > base + 0.0005 else ""))

    print("\n  PER-FINDING, best pooling chosen per column (IN SAMPLE, so upper bound)")
    red = {k: reduce_stack(L, k) for k in ("mean", "max", "top2", "softmax10")}
    chosen, cols = {}, []
    print("    %-18s %8s %8s %8s %8s %8s   pick" %
          ("finding", "single", "mean", "max", "top2", "soft10"))
    for j, t in enumerate(TASKS):
        yt = (y[:, j] > 0.5).astype(int)
        if yt.min() == yt.max():
            continue
        s = {k: auroc(y[:, [j]], red[k][:, [j]]) for k in red}
        b1 = auroc(y[:, [j]], L[0][:, [j]])
        pick = max(s, key=s.get)
        chosen[t] = pick
        cols.append(s[pick])
        print("    %-18s %8.4f %8.4f %8.4f %8.4f %8.4f   %s%s"
              % (t, b1, s["mean"], s["max"], s["top2"], s["softmax10"], pick,
                 "  (+%.4f)" % (s[pick] - b1) if s[pick] > b1 else ""))
    print("    per-column argmax macro: %.4f  (%+.4f)  <- IN SAMPLE, expect shrinkage"
          % (float(np.mean(cols)), float(np.mean(cols)) - base))

    print("\n  PER-PASS, on its own")
    for i, lab in enumerate(labels):
        print("    %-22s %.4f  (%+.4f)" % (lab, auroc(y, L[i]), auroc(y, L[i]) - base))

    if out_json:
        Path(out_json).write_text(json.dumps({
            "baseline": base, "uniform_best": best_uniform,
            "notebook_per_finding": v_nb, "per_column_pick": chosen,
            "labels": labels,
        }, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)


try:
    import modal

    _cfg = sc.load_config(PKG_ROOT / RUN_NAME, verbose=False)
    app = modal.App("sliding-window-tta")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(_cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(_cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={_cfg["modal"]["shards_mount"]: _shards,
                           _cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(_cfg))
    def tta_remote(run_name: str, checkpoint: str, split: str, n_offsets: int, reverse: bool):
        cfg = sc.remote_cfg(sc.load_config(PKG_ROOT / run_name, verbose=False))
        run_dir = Path(cfg["modal"]["runs_mount"]) / run_name
        L, y, labels = run_passes(cfg, run_dir, run_name, checkpoint, split,
                                  n_offsets, reverse)
        out = Path(cfg["modal"]["runs_mount"]) / "others" / ("tta_%s_%s.npz" % (run_name, split))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, logits=L.astype(np.float32), targets=y.astype(np.float32),
                            labels=np.array(labels, dtype=object))
        _runs.commit()
        print("  wrote %s" % out, flush=True)
        report(L, y, labels)

    @app.local_entrypoint()
    def modal_main(run: str = RUN_NAME, checkpoint: str = CHECKPOINT, split: str = SPLIT,
                   offsets: int = 3, reverse: bool = True):
        tta_remote.remote(run, checkpoint, split, offsets, reverse)

except ImportError:
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-only", metavar="NPZ",
                    help="re-pool a saved tta_*.npz without touching a GPU")
    a = ap.parse_args()
    if a.pool_only:
        z = np.load(a.pool_only, allow_pickle=True)
        report(z["logits"].astype(np.float64), z["targets"], list(z["labels"]))
    else:
        print(__doc__)
