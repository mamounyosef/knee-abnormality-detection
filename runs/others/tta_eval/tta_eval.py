"""
tta_eval.py — duplicate-series test-time augmentation, scored on val + audit.

38% of studies have MORE than one series competing for a slot, and plan_slots keeps
exactly one of them and throws the rest away.
This runs the SAME checkpoint over several slot_variants - variant 0 keeps the series
normally chosen, variant 1 the runner-up - so those studies are seen through genuinely
different images, and averages the logits.

That is the useful kind of ensemble: different INPUTS, not different checkpoints.
Averaging 5 checkpoints of one run bought us nothing (-0.0002) because their errors
were correlated; two different series of the same knee are far less so.

Nothing is trained. One inference pass per variant.

    python -m modal run runs/others/tta_eval/tta_eval.py   # on Modal, beside the shards
    python runs/others/tta_eval/tta_eval.py                # here, needs local shards

Writes results/<timestamp>/ next to this script: tta_summary.json + .txt, and one .npz
of averaged logits per split in the same format as the training-time predictions, so
ensemble.py can take it as a member.
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
EXP_NAME   = "convnext_small_4slot"   # which run's checkpoint to score
CHECKPOINT = "best"                   # best | last | top1..topK | <step> | path
SETS       = ("val", "audit")
VARIANTS   = (0, 1, 2)                # slot_variant: 0 = the series plan_slots normally
                                      # picks, 1 = the runner-up, 2 = third. Reaches 33%
                                      # of studies; varying duplicate_rule instead only
                                      # reached 17.5%, because different rules often
                                      # crown the same series anyway.
BATCH_SIZE  = None                    # None -> the run's val_batch_size
NUM_WORKERS = None                    # None -> the run's val_num_workers
# ===========================================================================


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(EXP_DIR))
import shared_code as sc  # noqa: E402


def _build_model(cfg):
    """The run's own SliceModel — imported from its train.py, not reimplemented."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_run_train", EXP_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model(cfg)


def _duplicated_studies(cfg, ids):
    """Which of `ids` actually HAVE a duplicated slot — the only ones TTA can change.

    Reported separately because averaging over rules is a no-op for the rest, and
    a gain measured over the whole split understates the effect on the studies it
    can reach by roughly the ratio of the two group sizes.
    """
    import pandas as pd
    idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    idx = idx[idx.slot.isin(cfg["input"]["slots"])]
    per = idx.groupby(["study", "slot"]).size()
    dup = set(per[per > 1].index.get_level_values(0))
    return [q in dup for q in ids]


def run_tta(cfg, exp_dir):
    import copy, io
    import numpy as np
    import torch

    device = torch.device("cuda" if sc.usable_cuda() else "cpu")
    res = Path(exp_dir) / cfg["output"]["run_dir"]
    ckpt_dir = res / cfg["output"]["checkpoints_dir"]
    tasks = cfg["tasks"]

    sc._banner("DUPLICATE-SERIES TTA", "\U0001F501")
    model = _build_model(cfg)
    ck, ck_path = sc._load_ckpt_weights(model, ckpt_dir, CHECKPOINT, device)
    model = model.to(device).eval()
    print(f"  checkpoint: {ck_path.name}  (epoch {ck.get('epoch')}, step {ck.get('step')}, "
          f"score {ck.get('score')})")
    print(f"  variants  : {', '.join(str(v) for v in VARIANTS)}")

    def _loss(logits, targets):
        return logits.sum() * 0.0                     # only the logits matter here

    from datetime import datetime
    out, blobs = {"_stamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}, {}
    for split in SETS:
        per_rule, ids, y = {}, None, None
        for rule in VARIANTS:
            c = copy.deepcopy(cfg)
            c["input"]["slot_variant"] = rule
            if BATCH_SIZE:
                c["dataloader"]["val_batch_size"] = BATCH_SIZE
            if NUM_WORKERS is not None:
                c["dataloader"]["val_num_workers"] = NUM_WORKERS
            loaders = sc.make_loaders(c)
            if split not in loaders:
                print(f"  ⚠️  no {split} loader — skipped")
                break
            pack = {}
            m = sc.evaluate(model, loaders[split], _loss, c, device, collect=pack)
            per_rule[rule] = pack["logits"]
            if ids is None:
                ids = loaders[split].dataset.df["StudyInstanceUID"].tolist()
                y = pack["targets"]
            print(f"  [{split}] slot_variant={rule} macro AUROC={m['macro_auroc']:.4f}")
        if not per_rule:
            continue

        stack = np.stack(list(per_rule.values()), axis=0)       # (R, N, T)
        avg = stack.mean(axis=0)
        singles = {r: sc.compute_metrics(y, 1 / (1 + np.exp(-p)), tasks)["macro_auroc"]
                   for r, p in per_rule.items()}
        ens = sc.compute_metrics(y, 1 / (1 + np.exp(-avg)), tasks)
        base = singles[VARIANTS[0]]     # variant 0 IS the normal, non-TTA prediction

        # the studies TTA can actually change, vs the ones where every rule agrees
        dup = np.asarray(_duplicated_studies(cfg, ids), dtype=bool)
        sub = {}
        for name, mask in (("duplicated", dup), ("single-series", ~dup)):
            if mask.sum() < 20:
                continue
            b = sc.compute_metrics(y[mask], 1 / (1 + np.exp(-per_rule[list(per_rule)[0]][mask])),
                                   tasks)["macro_auroc"]
            e = sc.compute_metrics(y[mask], 1 / (1 + np.exp(-avg[mask])), tasks)["macro_auroc"]
            sub[name] = {"n": int(mask.sum()), "single": b, "tta": e, "delta": e - b}

        print(f"  [{split}] no TTA (variant 0) = {base:.4f}")
        print(f"  [{split}] TTA average      = {ens['macro_auroc']:.4f} "
              f"({ens['macro_auroc'] - base:+.4f})")
        for name, d in sub.items():
            print(f"      {name:14s} n={d['n']:4d}  {d['single']:.4f} -> {d['tta']:.4f} "
                  f"({d['delta']:+.4f})")

        out[split] = {"checkpoint": ck_path.name, "variants": list(VARIANTS),
                      "n": int(len(ids)),
                      "per_variant_macro_auroc": {str(k): v for k, v in singles.items()}, "baseline": base,
                      "tta_macro_auroc": ens["macro_auroc"],
                      "delta": ens["macro_auroc"] - base,
                      "by_group": sub,
                      "per_task": {t: ens["per_task"][t]["auroc"] for t in tasks}}

        # returned to the launcher rather than written to the volume: this is an
        # analysis artifact, not a run artifact, so it belongs beside the script.
        buf = io.BytesIO()
        np.savez_compressed(buf, logits=avg.astype(np.float32),
                            targets=np.asarray(y, np.float32),
                            study_ids=np.asarray(ids, dtype=object),
                            tasks=np.asarray(list(tasks), dtype=object),
                            meta=np.asarray(["tta", split, "0", "0",
                                             f"{ens['macro_auroc']:.6f}"], dtype=object))
        blobs[split] = buf.getvalue()

    return out, blobs


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _write(out, blobs):
    """Everything lands in tta_eval/results/, next to this script."""
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = out.pop("_stamp")
    d = RESULTS_DIR / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "tta_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "tta_summary.txt").write_text(_render(out), encoding="utf-8")
    for split, blob in blobs.items():
        (d / f"tta_{split}.npz").write_bytes(blob)
    print(f"\n  wrote -> {d}")
    return d


def _render(out) -> str:
    L = ["=" * 78, f"DUPLICATE-SERIES TTA  —  {EXP_NAME}  —  checkpoint {CHECKPOINT}",
         f"variants: {VARIANTS}", "=" * 78]
    for split, o in out.items():
        if not isinstance(o, dict):
            continue
        L += ["", f"[{split}]  n={o.get('n', '?')}",
              f"  no TTA (variant {VARIANTS[0]}) : {o['baseline']:.4f}",
              f"  TTA average             : {o['tta_macro_auroc']:.4f}  ({o['delta']:+.4f})",
              "  per variant:"]
        L += [f"     variant {k}: {v:.4f}" for k, v in o["per_variant_macro_auroc"].items()]
        if o.get("by_group"):
            L.append("  by group (TTA can only change studies with a duplicated slot):")
            for name, g in o["by_group"].items():
                L.append(f"     {name:14s} n={g['n']:4d}  {g['single']:.4f} -> "
                         f"{g['tta']:.4f}  ({g['delta']:+.4f})")
        L.append("  per finding (TTA):")
        L += [f"     {t:18s} {v:.4f}" for t, v in o["per_task"].items()]
    return "\n".join(L) + "\n"


def run_local():
    out, blobs = run_tta(sc.load_config(EXP_DIR, verbose=False), EXP_DIR)
    _write(out, blobs)


try:
    import modal

    cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App(f"{EXP_NAME}-tta")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(cfg))
    def tta_remote():
        """Scores on the GPU beside the shards; the numbers travel back as the return
        value, so nothing analysis-shaped is left behind on the runs volume."""
        rcfg = sc.remote_cfg(cfg)
        out_dir = Path(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        return run_tta(rcfg, out_dir)

    @app.local_entrypoint()
    def modal_main():
        out, blobs = tta_remote.remote()
        _write(out, blobs)

except ImportError:
    pass


if __name__ == "__main__":
    run_local()
