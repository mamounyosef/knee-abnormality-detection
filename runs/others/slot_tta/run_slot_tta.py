"""
run_slot_tta.py — is a discarded duplicate series worth a second forward pass?

33% of studies have >1 series competing for a slot (val 38%, audit 43%). We keep the
`coverage` winner and throw the rest away. This scores the SAME checkpoint twice —
slot_variant 0 (the winner) and slot_variant 1 (the runner-up) — and asks whether
averaging the two beats either alone. No retraining, no engine change: `slot_variant`
already exists in plan_slots.

    python -m modal run runs/others/slot_tta/run_slot_tta.py     # GPU pass, writes npz here
    python runs/others/slot_tta/run_slot_tta.py                  # score what is already here

Nothing outside this folder is touched.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = Path("/root/runs") if Path("/root/runs/shared_code.py").exists() else HERE.parent.parent
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402

EXP_NAME = "convnext_small_relabel"
CHECKPOINT = "best"
SETS = ("val", "audit")
VARIANTS = (0, 1)

EXP_DIR = PKG_ROOT / EXP_NAME


def _build_model(cfg):
    """The run's own train.py owns the architecture — never re-declare it here."""
    sys.path.insert(0, str(EXP_DIR))
    from train import build_model
    return build_model(cfg)


def score_variants(cfg, exp_dir, device):
    """-> {(set, variant): (logits, targets, ids)}"""
    import numpy as np, pandas as pd, torch
    res = Path(exp_dir) / cfg["output"]["run_dir"]
    model = _build_model(cfg)
    ck, ck_path = sc._load_ckpt_weights(model, res / cfg["output"]["checkpoints_dir"],
                                        CHECKPOINT, device)
    model = model.to(device).eval()
    print(f"  checkpoint {ck_path}  (epoch {ck.get('epoch')}, step {ck.get('step')}, "
          f"score {ck.get('score')})")

    index = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    named = {"val": cfg["paths"]["val_csv"], "audit": cfg["evaluation"]["audit"]["csv"]}
    out = {}
    for v in VARIANTS:
        cfg["input"]["slot_variant"] = v
        plan = sc.plan_slots(index, cfg)
        for s in SETS:
            df = sc._eval_df(cfg, named[s])
            pk = {}
            y_true, _, _ = sc._predict_dataframe(cfg, model, df, device, None, plan=plan,
                                                 desc=f"{s}/v{v}", collect=pk)
            ids = df["StudyInstanceUID"].astype(str).values
            out[(s, v)] = (np.concatenate(pk["_lg"]), y_true, ids)
            print(f"  ✅ {s} variant {v}: {out[(s,v)][0].shape}")
    return out, cfg["tasks"]


def save(out, tasks, dest: Path):
    import numpy as np
    dest.mkdir(parents=True, exist_ok=True)
    for (s, v), (lg, yt, ids) in out.items():
        np.savez_compressed(dest / f"{s}_v{v}.npz", logits=lg, targets=yt,
                            study_ids=np.array(ids, dtype=object),
                            tasks=np.array(tasks, dtype=object))
    print(f"  💾 wrote {len(out)} npz -> {dest}")


# ---------------------------------------------------------------- scoring (local, no GPU)
def auroc(y, s):
    import numpy as np
    y, s = np.asarray(y), np.asarray(s)
    p = y > 0.5                                  # the engine's convention: strictly greater
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[o[j + 1]] == s[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[p].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def rank(x):
    import numpy as np
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def report(dest: Path):
    import numpy as np, pandas as pd
    index = pd.read_csv(HERE.parent.parent.parent / "data" / "07_series_index.csv")
    slots = ["sag_plain", "sag_fluid", "cor_fluid", "ax_fluid"]
    g = index[index.slot.isin(slots)].groupby(["study", "slot"]).size()
    contested = set(g[g > 1].index.get_level_values(0).astype(str))

    print(f"\n{'set':<8}{'n':>6}{'contested':>11}{'variant0':>11}{'variant1':>11}"
          f"{'MEAN':>11}{'delta':>9}")
    print("-" * 67)
    for s in SETS:
        f0, f1 = dest / f"{s}_v0.npz", dest / f"{s}_v1.npz"
        if not (f0.exists() and f1.exists()):
            print(f"{s:<8} missing npz - run the modal pass first")
            continue
        z0, z1 = np.load(f0, allow_pickle=True), np.load(f1, allow_pickle=True)
        ids = [str(i) for i in z0["study_ids"]]
        pos = {i: k for k, i in enumerate(ids)}
        idx = [pos[str(i)] for i in z1["study_ids"]]
        T = z0["targets"]
        L0 = z0["logits"]
        L1 = np.empty_like(L0); L1[idx] = z1["logits"]
        nT = len(z0["tasks"])
        R0 = np.stack([rank(L0[:, k]) for k in range(nT)], 1)
        R1 = np.stack([rank(L1[:, k]) for k in range(nT)], 1)
        RM = (R0 + R1) / 2
        a0 = np.nanmean([auroc(T[:, k], R0[:, k]) for k in range(nT)])
        a1 = np.nanmean([auroc(T[:, k], R1[:, k]) for k in range(nT)])
        am = np.nanmean([auroc(T[:, k], RM[:, k]) for k in range(nT)])
        nc = sum(1 for i in ids if i in contested)
        print(f"{s:<8}{len(ids):>6}{nc:>7} ({100*nc/len(ids):3.0f}%)"
              f"{a0:>11.4f}{a1:>11.4f}{am:>11.4f}{am-a0:>+9.4f}")
    print("\ndelta = MEAN minus variant0. variant0 is the current production pick, so a")
    print("positive delta is what averaging over the discarded duplicate buys.")


# ---------------------------------------------------------------- modal
try:
    import modal

    _cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App("slot-tta")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(_cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(_cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={_cfg["modal"]["shards_mount"]: _shards,
                           _cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(_cfg))
    def remote():
        """Returns the npz files as bytes — the container cannot write to the local disk."""
        import io, zipfile, torch
        rcfg = sc.remote_cfg(_cfg)
        exp_dir = Path(_cfg["modal"]["runs_mount"]) / _cfg["experiment"]["name"]
        out, tasks = score_variants(rcfg, exp_dir, torch.device("cuda"))
        tmp = Path("/tmp/slot_tta"); save(out, tasks, tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.glob("*.npz"):
                z.write(f, f.name)
        return buf.getvalue()

    @app.local_entrypoint()
    def modal_main():
        import io, zipfile
        blob = remote.remote()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(HERE)
        print(f"\n⬇️  pulled the predictions to {HERE}")
        report(HERE)

except ImportError:
    pass


if __name__ == "__main__":
    report(HERE)
