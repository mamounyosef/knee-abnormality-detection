"""
run_tta.py — classic test-time augmentation, three arms, no retraining.

The submission notebook uses 0.14-0.25 h of a 9 h cap, so inference budget is
essentially free. This asks whether spending some of it on TTA pays.

  arm A  SLICE RESAMPLE. Eval normally takes bin CENTRES (slice_sampling: "even").
         Re-running with "random" and a different seed picks a different slice from
         each of the 24 bins - same coverage, different pixels, identical anatomy.
         Needs no engine change: the dataset seeds itself from
         cfg["reproducibility"]["seed"].

  arm B  HORIZONTAL FLIP. hflip is disabled in training for a real reason - it swaps
         medial (inner) and lateral (outer), and 4 of the 12 findings are side-specific.
         So the flipped pass has its Medial/Lateral predictions SWAPPED BACK before
         averaging. If the model learned genuine left-right anatomy this should be
         near-lossless and add real diversity; if it scores far worse, the flip is
         breaking something and the arm is dead.

  arm C  MILD GEOMETRY + PHOTOMETRY. rotation 10deg, translate 5%, scale +-10%,
         brightness/contrast 0.15 - the exact transforms training already uses at p=0.3.
         That is the point: the model has SEEN these, so unlike the flip they are
         in-distribution, and none of them turns medial into lateral. Enabled at eval
         through the new augmentation.eval_enable knob.

    python -m modal run runs/others/tta/run_tta.py     # GPU pass, writes npz here
    python runs/others/tta/run_tta.py                  # score what is already here

Nothing outside this folder is touched.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = Path("/root/runs") if Path("/root/runs/shared_code.py").exists() else HERE.parent.parent
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402

EXP_NAME   = "convnext_base_relabel"     # the best single model
CHECKPOINT = "best"
SETS       = ("val", "audit")
N_SLICE    = 3          # arm A passes, each a different seed
N_GEOM     = 4          # arm C passes, each a different seed
EXP_DIR    = PKG_ROOT / EXP_NAME

# the two side-specific pairs a horizontal flip exchanges
FLIP_PAIRS = [("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")]


class HFlip:
    """Wraps the model so every batch is flipped on its last axis before the forward."""

    def __init__(self, model):
        self.m = model

    def __call__(self, x):
        return self.m(x.flip(-1))

    def eval(self):
        self.m.eval()
        return self

    def to(self, d):
        self.m.to(d)
        return self

    def __getattr__(self, k):
        return getattr(self.m, k)


def collect(cfg, exp_dir, device):
    import copy
    import numpy as np
    import pandas as pd

    res = Path(exp_dir) / cfg["output"]["run_dir"]
    sys.path.insert(0, str(EXP_DIR))
    from train import build_model
    model = build_model(cfg)
    ck, ck_path = sc._load_ckpt_weights(model, res / cfg["output"]["checkpoints_dir"],
                                        CHECKPOINT, device)
    model = model.to(device).eval()
    print("  checkpoint %s (epoch %s, score %s)" % (ck_path, ck.get("epoch"), ck.get("score")))

    index = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    plan = sc.plan_slots(index, cfg)
    named = {"val": cfg["paths"]["val_csv"], "audit": cfg["evaluation"]["audit"]["csv"]}
    out = {}

    def one(c, m, split, tag):
        df = sc._eval_df(c, named[split])
        pk = {}
        y, _, _ = sc._predict_dataframe(c, m, df, device, None, plan=plan,
                                        desc="%s/%s" % (split, tag), collect=pk)
        out[(split, tag)] = (np.concatenate(pk["_lg"]), y,
                             df["StudyInstanceUID"].astype(str).values)
        print("  OK %s/%s" % (split, tag))

    for split in SETS:
        one(cfg, model, split, "base")                  # the production pass
        one(cfg, HFlip(model), split, "hflip")          # arm B
        for s in range(1, N_SLICE + 1):                 # arm A
            c = copy.deepcopy(cfg)
            c["input"]["slice_sampling"] = "random"
            c["reproducibility"]["seed"] = 1000 + s
            one(c, model, split, "slice%d" % s)
        for s in range(1, N_GEOM + 1):                  # arm C
            c = copy.deepcopy(cfg)
            c["augmentation"]["eval_enable"] = True
            c["augmentation"]["hflip"] = {"p": 0.0}     # never - it swaps medial/lateral
            c["augmentation"]["slice_dropout"] = {"p": 0.0, "frac": 0.0}   # destructive
            c["reproducibility"]["seed"] = 2000 + s
            one(c, model, split, "geom%d" % s)
    return out, cfg["tasks"]


def save(out, tasks, dest):
    import numpy as np
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for (s, t), (lg, y, ids) in out.items():
        np.savez_compressed(dest / ("%s_%s.npz" % (s, t)), logits=lg, targets=y,
                            study_ids=np.array(ids, dtype=object),
                            tasks=np.array(tasks, dtype=object))
    print("  wrote %d npz -> %s" % (len(out), dest))


# ------------------------------------------------------------------ scoring (local)
def auroc(y, s):
    import numpy as np
    y, s = np.asarray(y), np.asarray(s)
    p = y > 0.5
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[o[j + 1]] == s[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[p].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def rank(x):
    import numpy as np
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def report(dest):
    import numpy as np
    dest = Path(dest)
    for split in SETS:
        fb = dest / ("%s_base.npz" % split)
        if not fb.exists():
            print("%s: missing npz - run the modal pass first" % split)
            continue
        zb = np.load(fb, allow_pickle=True)
        tasks = [str(t) for t in zb["tasks"]]
        T = zb["targets"]
        ids = [str(i) for i in zb["study_ids"]]
        pos = {i: k for k, i in enumerate(ids)}
        nT = len(tasks)

        def load(tag):
            z = np.load(dest / ("%s_%s.npz" % (split, tag)), allow_pickle=True)
            a = np.empty_like(zb["logits"])
            a[[pos[str(i)] for i in z["study_ids"]]] = z["logits"]
            if tag == "hflip":                  # undo the side swap the flip caused
                a = a.copy()
                for x, y in FLIP_PAIRS:
                    i, j = tasks.index(x), tasks.index(y)
                    a[:, [i, j]] = a[:, [j, i]]
            return np.stack([rank(a[:, k]) for k in range(nT)], 1)

        def M(R):
            return np.nanmean([auroc(T[:, k], R[:, k]) for k in range(nT)])

        base = load("base")
        slices = [load("slice%d" % s) for s in range(1, N_SLICE + 1)
                  if (dest / ("%s_slice%d.npz" % (split, s))).exists()]
        geoms = [load("geom%d" % s) for s in range(1, N_GEOM + 1)
                 if (dest / ("%s_geom%d.npz" % (split, s))).exists()]
        hflip = load("hflip") if (dest / ("%s_hflip.npz" % split)).exists() else None

        print("\n=== %s  (%d studies) ===" % (split, len(ids)))
        b = M(base)
        print("  %-34s%.4f" % ("base (production)", b))
        if hflip is not None:
            print("  %-34s%.4f" % ("hflip alone (sides swapped back)", M(hflip)))
            v = M((base + hflip) / 2)
            print("  %-34s%.4f   %+.4f" % ("base + hflip", v, v - b))
        for i, r in enumerate(slices, 1):
            print("  %-34s%.4f" % ("slice-resample %d alone" % i, M(r)))
        if slices:
            a = (base + sum(slices)) / (1 + len(slices))
            print("  %-34s%.4f   %+.4f" % ("base + all slice resamples", M(a), M(a) - b))
        for i, r in enumerate(geoms, 1):
            print("  %-34s%.4f" % ("geometry %d alone" % i, M(r)))
        if geoms:
            for k in range(1, len(geoms) + 1):
                g = (base + sum(geoms[:k])) / (1 + k)
                print("  %-34s%.4f   %+.4f" % ("base + %d geometry" % k, M(g), M(g) - b))
        if slices and geoms:
            a = (base + sum(slices) + sum(geoms)) / (1 + len(slices) + len(geoms))
            print("  %-34s%.4f   %+.4f" % ("base + slices + geometry", M(a), M(a) - b))


# ------------------------------------------------------------------ modal
try:
    import modal

    _cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App("knee-tta")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(_cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(_cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={_cfg["modal"]["shards_mount"]: _shards,
                           _cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(_cfg))
    def remote():
        """Returns the npz as bytes - the container cannot write to the local disk."""
        import io
        import zipfile
        import torch
        rcfg = sc.remote_cfg(_cfg)
        exp_dir = Path(_cfg["modal"]["runs_mount"]) / _cfg["experiment"]["name"]
        out, tasks = collect(rcfg, exp_dir, torch.device("cuda"))
        tmp = Path("/tmp/tta")
        save(out, tasks, tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.glob("*.npz"):
                z.write(f, f.name)
        return buf.getvalue()

    @app.local_entrypoint()
    def modal_main():
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(remote.remote())) as z:
            z.extractall(HERE)
        print("\npulled the predictions to %s" % HERE)
        report(HERE)

except ImportError:
    pass


if __name__ == "__main__":
    report(HERE)
